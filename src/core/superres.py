"""
Real-ESRGAN 超分处理器
基于 Real-ESRGAN 的 4 倍超分，针对服装面料纹理优化
"""
import os
from typing import Optional, List
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


class SuperResolutionProcessor:
    """
    基于 Real-ESRGAN 的 4 倍超分处理器
    
    默认模型：realesr-general-x4v3
      - 使用 SRVGGNetCompact 架构，推理速度快
      - 对真实世界面料纹理（编织、丝绸、皮纹）还原效果最佳
      - 运行时若权重不存在，则自动下载至 ckpt/ 目录
    """

    _SR_MODEL_URLS = {
        "realesr-general-x4v3": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            "realesr-general-x4v3.pth",
        ),
        "RealESRGAN_x4plus": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "RealESRGAN_x4plus.pth",
        ),
    }

    def __init__(
        self,
        model_name: str = "realesr-general-x4v3",
        ckpt_dir: Optional[str] = None,
        device: Optional[str] = None,
        tile: int = 512,
        tile_pad: int = 32,
    ):
        """
        Args:
            model_name: 模型名称
            ckpt_dir: 权重目录
            device: 运行设备
            tile: tile 大小，显存不足时自动减半
            tile_pad: tile 填充大小
        """
        self.model_name = model_name
        self.ckpt_dir = ckpt_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "ckpt")
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.tile = tile
        self.tile_pad = tile_pad
        self.upsampler = None

    def _load(self):
        """懒加载 Real-ESRGAN 模型"""
        if self.upsampler is not None:
            return

        if torch is None:
            raise ImportError("请安装 torch: pip install torch")

        # 兼容性补丁
        try:
            import sys, types, torchvision
            if not hasattr(torchvision.transforms, 'functional_tensor'):
                from torchvision.transforms import functional as TF
                mod = types.ModuleType("torchvision.transforms.functional_tensor")
                mod.rgb_to_grayscale = TF.rgb_to_grayscale
                sys.modules["torchvision.transforms.functional_tensor"] = mod
        except Exception:
            pass

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            raise ImportError(
                "请安装 Real-ESRGAN 以启用超分功能:\n"
                "  pip install realesrgan basicsr"
            )

        url, fname = self._SR_MODEL_URLS[self.model_name]
        model_path = os.path.join(self.ckpt_dir, fname)

        # 权重不存在则自动下载
        if not os.path.exists(model_path):
            print(f"[SR] 权重文件不存在，自动下载：{url}")
            try:
                import urllib.request
                os.makedirs(self.ckpt_dir, exist_ok=True)
                urllib.request.urlretrieve(
                    url, model_path,
                    reporthook=lambda b, bs, t: print(
                        f"\r  下载中... {min(b*bs, t)*100 // max(t, 1)}%", end=""
                    )
                )
                print()
                print(f"[SR] 下载完成：{model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"权重下载失败：{e}\n"
                    f"请手动下载 {url}\n"
                    f"并放入 {self.ckpt_dir}/"
                ) from e

        print(f"[SR] 加载模型：{self.model_name} ({self.device}) ...")

        if self.model_name == "realesr-general-x4v3":
            try:
                from realesrgan.archs.srvgg_arch import SRVGGNetCompact
                model = SRVGGNetCompact(
                    num_in_ch=3, num_out_ch=3,
                    num_feat=64, num_conv=32,
                    upscale=4, act_type="prelu",
                )
            except ImportError:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3,
                    num_feat=64, num_block=23, num_grow_ch=32, scale=4
                )
        else:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=0,
            half=(self.device != "cpu"),
        )
        print(f"[SR] {self.model_name} 加载完成")

    def upscale(self, image: Image.Image) -> Image.Image:
        """对单张图像做 4 倍超分"""
        self._load()
        img_np = np.array(image.convert("RGB"))
        img_bgr = img_np[:, :, ::-1]
        
        try:
            out_bgr, _ = self.upsampler.enhance(img_bgr, outscale=4)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[SR] OOM，自动减小 tile 重试 ({self.tile} → {self.tile//2}) ...")
                torch.cuda.empty_cache()
                original_tile = self.upsampler.tile
                self.upsampler.tile = self.tile // 2
                out_bgr, _ = self.upsampler.enhance(img_bgr, outscale=4)
                self.upsampler.tile = original_tile
            else:
                raise
        
        out_rgb = out_bgr[:, :, ::-1]
        return Image.fromarray(out_rgb)

    def batch_upscale(
        self, 
        images: List[Image.Image], 
        progress_callback=None
    ) -> List[Image.Image]:
        """批量超分，失败时返回原图"""
        results = []
        for i, img in enumerate(images):
            if img is None:
                results.append(None)
                continue
            w, h = img.size
            print(f"[SR] 超分 {i+1}/{len(images)}: {w}×{h} → {w*4}×{h*4}")
            try:
                results.append(self.upscale(img))
            except Exception as e:
                print(f"[SR] 第 {i+1} 张超分失败：{e}，返回原图")
                results.append(img)
            if progress_callback:
                progress_callback(i + 1, len(images))
            if self.device != "cpu":
                torch.cuda.empty_cache()
        return results


# 需要导入 numpy
import numpy as np
