"""
IC-Light 阴影后处理器
使用 IC-Light foreground-conditioned 模型为换装结果图重新打光
"""
import os
from typing import Optional, List
from PIL import Image

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class ICLightProcessor:
    """
    使用 IC-Light foreground-conditioned 模型为换装结果图重新打光
    官方仓库：https://github.com/lllyasviel/IC-Light
    """

    SD15_BASE = "runwayml/stable-diffusion-v1-5"
    VAE_ID = "stabilityai/sd-vae-ft-mse"

    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        """
        Args:
            ckpt_path: IC-Light 模型权重路径
            device: 运行设备
        """
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.pipe = None

    def _load(self):
        """懒加载 IC-Light 模型"""
        if self.pipe is not None:
            return

        if torch is None:
            raise ImportError("请安装 torch: pip install torch")

        try:
            from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("请安装 diffusers 和 safetensors: pip install diffusers safetensors")

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"IC-Light 权重不存在：{self.ckpt_path}\n"
                "请从 https://huggingface.co/lllyasviel/ic-light 下载 ic-light-fc.safetensors"
            )

        print(f"[IC-Light] 加载模型 ({self.device}) ...")
        from diffusers import UNet2DConditionModel, DDIMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer

        # 使用 SD1.5 作为 base，替换 UNet 权重
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.SD15_BASE,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )

        # 加载 IC-Light 权重到 UNet
        ic_state = load_file(self.ckpt_path, device=self.device)
        with torch.no_grad():
            old_conv = pipe.unet.conv_in
            new_conv = torch.nn.Conv2d(8, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding)
            new_conv = new_conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
            new_conv.weight.zero_()
            new_conv.weight[:, :4, :, :].copy_(old_conv.weight)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
            pipe.unet.conv_in = new_conv
            pipe.unet.config["in_channels"] = 8

        missing, unexpected = pipe.unet.load_state_dict(ic_state, strict=False)
        print(f"[IC-Light] UNet 权重加载完成 (missing={len(missing)}, unexpected={len(unexpected)})")

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.enable_attention_slicing()
        
        self.pipe = pipe
        print("[IC-Light] 模型加载完成")

    def _make_fg_latent(self, image: Image.Image) -> "torch.Tensor":
        """将前景图转为 4 通道 latent（让 IC-Light fc 拼接用）"""
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        return transform(image).unsqueeze(0).to(
            self.device, 
            torch.float16 if self.device != "cpu" else torch.float32
        )

    def process(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        negative_prompt: str = "overexposed, dark, blurry, noisy, low quality",
        strength: float = 0.35,
        num_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = 42,
        light_direction: str = "top_left",
        specular_boost: float = 1.0,
    ) -> Image.Image:
        """
        对单张图像做 IC-Light 打光处理。
        
        Args:
            image: 输入图像
            prompt: 可选的自定义 prompt
            negative_prompt: 负面 prompt
            strength: 重绘强度
            num_steps: 推理步数
            guidance_scale: 引导系数
            seed: 随机种子
            light_direction: 光源方向
            specular_boost: 高光增强系数
            
        Returns:
            处理后的 PIL Image
        """
        self._load()

        # 若未传入 prompt，根据光源和高光参数自动构建
        if prompt is None:
            from src.models.detectors import build_iclight_prompt
            params = build_iclight_prompt(
                light_direction=light_direction,
                material="general",
                specular_boost=specular_boost,
            )
            prompt = params["prompt"]
            negative_prompt = params["negative_prompt"]
            guidance_scale = params["guidance_scale"]

        orig_size = image.size
        fg_tensor = self._make_fg_latent(image)

        # 编码前景为 latent（4 通道），用于在 UNet 输入时进行拼接
        with torch.no_grad():
            fg_latent = self.pipe.vae.encode(
                fg_tensor.to(self.pipe.vae.dtype)
            ).latent_dist.sample() * 0.18215

        # 使用 Wrapper 动态拼接 4+4 通道
        class ICLightUNetWrapper(nn.Module):
            def __init__(self, unet, fg_latent):
                super().__init__()
                self.unet = unet
                self.fg_latent = fg_latent
                self.config = unet.config
                self.add_embedding = getattr(unet, "add_embedding", None)

            def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
                repeated_fg = self.fg_latent.repeat(sample.shape[0], 1, 1, 1)
                concatenated_input = torch.cat([sample, repeated_fg], dim=1)
                return self.unet(concatenated_input, timestep, encoder_hidden_states, **kwargs)

        orig_unet = self.pipe.unet
        self.pipe.unet = ICLightUNetWrapper(orig_unet, fg_latent)

        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image.resize((512, 512)),
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        finally:
            self.pipe.unet = orig_unet

        return result.resize(orig_size, Image.LANCZOS)

    def batch_process(
        self, 
        images: List[Image.Image], 
        **kwargs
    ) -> List[Image.Image]:
        """批量处理多张图"""
        results = []
        for i, img in enumerate(images):
            if img is None:
                results.append(None)
                continue
            print(f"[IC-Light] 处理第 {i+1}/{len(images)} 张 ...")
            try:
                results.append(self.process(img, seed=kwargs.get("seed", 42) + i, **{
                    k: v for k, v in kwargs.items() if k != "seed"
                }))
            except Exception as e:
                print(f"[IC-Light] 第 {i+1} 张处理失败：{e}")
                results.append(img)
        return results
