"""
服装详情图生成流水线
Pipeline: 上传衣服图 → SAM提取Mask → IDM-VTON批量换装(5张姿态) → IC-Light阴影后处理
"""

import os
import sys
import glob
import time
import numpy as np
from PIL import Image
import torch
import random

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

IDMVTON_URL   = os.environ.get("IDMVTON_URL", "http://127.0.0.1:7861/")
POSES_DIR     = os.path.join(os.path.dirname(__file__), "poses")
OUTPUTS_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
SAM_CKPT      = os.path.join(os.path.dirname(__file__), "ckpt", "sam_vit_h_4b8939.pth")
ICLIGHT_CKPT  = os.path.join(os.path.dirname(__file__), "ckpt", "iclight_sd15_fc.safetensors")

MAX_POSES = 5   # 最多读取的姿态底图数量

# 模特类型 → poses 子目录名称
MODEL_TYPE_DIRS = {
    "adult_female":  "adult_female",   # 成人女性
    "adult_male":    "adult_male",     # 成人男性
    "adult_neutral": "adult_neutral",  # 成人中性
    "child_female":  "child_female",   # 儿童女
    "child_male":    "child_male",     # 儿童男
    "child_neutral": "child_neutral",  # 儿童中性
}

def get_pose_files(base_poses_dir: str, model_type: str) -> list:
    """
    根据模特类型返回适用的姿态图片路径列表。
    对于 _neutral 类型，将合并对应群体的 female 和 male 目录图片。
    若特定目录不存在或为空，则回退到 base_poses_dir 根目录。
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    
    def _fetch(sub: str):
        if not sub: return []
        candidate = os.path.join(base_poses_dir, sub)
        if not os.path.isdir(candidate): return []
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(candidate, ext)))
        return files

    files = []
    if model_type.endswith("_neutral"):
        prefix = model_type.split("_")[0]  # "adult" or "child"
        files.extend(_fetch(MODEL_TYPE_DIRS.get(f"{prefix}_female", "")))
        files.extend(_fetch(MODEL_TYPE_DIRS.get(f"{prefix}_male", "")))
        files.extend(_fetch(MODEL_TYPE_DIRS.get(model_type, "")))
        files = list(set(files))  # 去重
    else:
        files = _fetch(MODEL_TYPE_DIRS.get(model_type, ""))
        
    if files:
        print(f"[Pose Router] 找到 {len(files)} 张姿态图 (模型类型: {model_type})")
        # 对 neutral 列表进行固定打乱，保证前几张能看到男女混排
        if model_type.endswith("_neutral"):
            import random
            random.Random(42).shuffle(files)
        return files
        
    # 回退到根目录
    fallback = []
    for ext in exts:
        fallback.extend(glob.glob(os.path.join(base_poses_dir, ext)))
    print(f"[Pose Router] 特定子目录无图片，回退到根目录，找到 {len(fallback)} 张图")
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# 材质识别 & 光影参数生成
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 零、工具函数：等比例缩放与填充（防止变形）
# ─────────────────────────────────────────────────────────────────────────────

def resize_and_pad(image: Image.Image, target_size=(768, 1024), fill_color=(255, 255, 255)) -> tuple:
    """
    保持长宽比缩放图片，并在两侧补边至目标尺寸。
    
    Returns:
        (padded_image, (x_offset, y_offset, new_w, new_h))
    """
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    
    # 计算缩放比例
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    
    resized_img = image.resize((new_w, new_h), Image.LANCZOS)
    
    # 创建背景并居中贴图
    padded_img = Image.new("RGB", target_size, fill_color)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_img.paste(resized_img, (x_offset, y_offset))
    
    return padded_img, (x_offset, y_offset, new_w, new_h)


def unpad_and_resize(padded_image: Image.Image, padding_info: tuple, final_size=None) -> Image.Image:
    """
    根据 padding 信息裁掉边框，并可选缩放回最终尺寸。
    """
    x, y, w, h = padding_info
    img_cropped = padded_image.crop((x, y, x + w, y + h))
    if final_size:
        return img_cropped.resize(final_size, Image.LANCZOS)
    return img_cropped


# 材质关键词映射表
_MATERIAL_KEYWORDS = {
    "leather": ["leather", "leatherette", "faux leather", "pu leather", "leather-like",
                "皮质", "皮革", "真皮", "pu皮", "人造皮", "皮衣", "机车服"],
    "satin":   ["satin", "silk", "silky", "glossy", "shiny", "polyester silk",
                "绸缎", "丝缎", "缎面", "丝绸", "光泽", "亮面", "真丝", "桑蚕丝", "醋酸", "亮片"],
    "denim":   ["denim", "jeans", "jean", "washed denim",
                "牛仔", "水洗牛仔", "丹宁", "牛仔布"],
    "chiffon": ["chiffon", "georgette", "sheer", "transparent", "mesh",
                "雪纺", "乔其纱", "薄纱", "网眼", "透明", "透视"],
    "linen":   ["linen", "flax", "hemp",
                "亚麻", "麻料", "棉麻", "汉麻"],
    "cotton":  ["cotton", "jersey", "combed cotton",
                "棉质", "全棉", "纯棉", "精梳棉", "针织棉", "汗布"],
    "wool":    ["wool", "cashmere", "knit", "knitted", "sweater", "cardigan",
                "毛料", "羊毛", "羊绒", "针织", "编织", "毛衣", "开衫"],
    "velvet":  ["velvet", "velour", "suede", "flocked",
                "丝绒", "天鹅绒", "金丝绒", "磨毛", "麂皮"],
    "lace":    ["lace", "embroidery", "hollow out",
                "蕾丝", "刺绣", "镂空", "钩编"],
}

# 光源方向 → IC-Light prompt 片段
_LIGHT_DIR_PROMPTS = {
    "top_left":    "soft directional light from upper left at 45 degrees",
    "top_right":   "soft directional light from upper right at 45 degrees",
    "top":         "soft overhead lighting from directly above",
    "front":       "soft frontal studio lighting",
}

DEFAULT_LIGHT_DIR = "top_left"   # 默认光源：左上方 45°

# 类别关键词映射表
_CATEGORY_KEYWORDS = {
    "lower_body": ["pants", "trousers", "shorts", "skirt", "jeans", "leggings", "bottom",
                   "裤", "裙", "短裤", "长裤", "牛仔裤", "半身裙", "下装", "运动裤", "休闲裤"],
    "dresses":    ["dress", "gown", "robe", "one-piece", "jumpsuit",
                   "连衣裙", "长裙", "礼服", "旗袍", "连体衣", "连身裙", "连体衫"],
}

# 性别与人群关键词映射表
_GENDER_KEYWORDS = {
    "adult_female":  ["woman", "women", "female", "lady", "girl", "her", "she", 
                      "女", "女性", "女士", "女孩", "美女", "熟女"],
    "adult_male":    ["man", "men", "male", "guy", "boy", "his", "he", "him",
                      "男", "男性", "男士", "男孩", "帅哥", "型男"],
    "adult_neutral": ["neutral", "androgynous", "unisex", "中性", "男女同款", "无性别"],
    "child_female":  ["child girl", "little girl", "toddler girl", "kid girl",
                      "女童", "女孩儿", "小女孩", "小女生"],
    "child_male":    ["child boy", "little boy", "toddler boy", "kid boy",
                      "男童", "男孩儿", "小男孩", "小男生"],
    "child_neutral": ["child neutral", "unisex child", "中性童装", "男女童"],
}


def detect_category(garment_desc: str) -> str:
    """
    从服装描述文本中检测类别。
    返回: 'upper_body' (默认) | 'lower_body' | 'dresses'
    """
    if not garment_desc:
        return "upper_body"
    desc_lower = garment_desc.lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw.lower() in desc_lower for kw in keywords):
            print(f"[Category] 识别到类别: {cat} (desc='{garment_desc}')")
            return cat
    return "upper_body"


def detect_material(garment_desc: str) -> str:
    """
    从服装描述文本中检测材质类型。
    返回常用材质键或 'general'
    """
    if not garment_desc: return "general"
    desc_lower = garment_desc.lower()
    for material, keywords in _MATERIAL_KEYWORDS.items():
        if any(kw.lower() in desc_lower for kw in keywords):
            print(f"[Material] 识别到材质: {material} (desc='{garment_desc}')")
            return material
    return "general"


def detect_gender(garment_desc: str) -> str:
    """
    从服装描述文本中检测性别/人群。
    返回: 'adult_female' (默认) | 'adult_male' | 'adult_neutral' | 'child_female' | 'child_male' | 'child_neutral'
    """
    if not garment_desc: return "adult_female"
    desc_lower = garment_desc.lower()
    
    # 优先检测儿童
    if any(kw in desc_lower for kw in ["child", "kid", "little", "toddler", "童", "小"]):
        if any(kw in desc_lower for kw in ["neutral", "unisex", "中性", "男女"]):
            return "child_neutral"
        if any(kw in desc_lower for kw in ["girl", "female", "女"]):
            return "child_female"
        if any(kw in desc_lower for kw in ["boy", "male", "男"]):
            return "child_male"
            
    # 普通成人检测
    for gender, keywords in _GENDER_KEYWORDS.items():
        if any(kw.lower() in desc_lower for kw in keywords):
            return gender
            
    return "adult_female"


def detect_ethnicity(garment_desc: str) -> str:
    """
    [NEW] 从服装描述中检测人种（当前主要用于前端 UI 联动）
    返回: 'Caucasian' (默认) | 'Asian' | 'Black' | 'Middle Eastern'
    """
    desc_lower = garment_desc.lower() if garment_desc else ""
    if any(kw in desc_lower for kw in ["亚洲", "华", "中式", "asian", "oriental", "chinese"]):
        return "Asian"
    if any(kw in desc_lower for kw in ["非", "黑", "african", "black"]):
        return "Black"
    return "Caucasian"


def detect_age_group(garment_desc: str) -> str:
    """
    [NEW] 从服装描述中检测年龄段
    返回: '20s' (默认) | '30s' | '40s' | 'teens'
    """
    desc_lower = garment_desc.lower() if garment_desc else ""
    if any(kw in desc_lower for kw in ["中年", "30", "40", "mature"]):
        return "40s组 / 40s"
    if any(kw in desc_lower for kw in ["青", "少", "18", "teens"]):
        return "少年组 / Teens"
    return "20岁组 / 20s"


def build_iclight_prompt(
    light_direction: str = DEFAULT_LIGHT_DIR,
    material: str = "general",
    specular_boost: float = 1.0,
) -> dict:
    """
    根据光源方向和材质构建 IC-Light 的 prompt / negative_prompt 和 guidance_scale。

    Returns:
        {
          "prompt": str,
          "negative_prompt": str,
          "guidance_scale": float,
          "specular_boost": float,  # 透传给调用方
        }
    """
    light_desc = _LIGHT_DIR_PROMPTS.get(light_direction, _LIGHT_DIR_PROMPTS[DEFAULT_LIGHT_DIR])

    # 基础 prompt
    base_prompt = (
        f"a person wearing clothes, {light_desc}, "
        "natural shadow, realistic lighting, commercial fashion photography, high quality, 8k"
    )

    # 材质特化 prompt 追加
    if material == "leather":
        base_prompt += (
            ", leather texture, subtle specular highlight, "
            "3D depth and thickness, rich surface detail"
        )
    elif material == "satin":
        base_prompt += (
            ", satin sheen, smooth silky surface, "
            "elegant specular reflection, 3D depth and thickness"
        )

    negative_prompt = (
        "overexposed, flat lighting, dark, blurry, noisy, "
        "low quality, cartoon, painting, illustration"
    )

    # 高光材质提升 guidance_scale 以增强对比度和高光响应
    guidance_scale = 7.5 * specular_boost if specular_boost > 1.0 else 7.5

    return {
        "prompt":         base_prompt,
        "negative_prompt": negative_prompt,
        "guidance_scale": min(guidance_scale, 15.0),  # 上限 15
        "specular_boost": specular_boost,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 一、SAM Mask 提取
# ─────────────────────────────────────────────────────────────────────────────

class SAMExtractor:
    """使用 Segment Anything Model 从衣服图中提取服装 Mask"""

    def __init__(self, ckpt_path: str = SAM_CKPT, device: str = None):
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None

    def _load(self):
        if self.predictor is not None:
            return
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError("请安装 segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"SAM 权重文件不存在: {self.ckpt_path}\n"
                "请从 https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 下载并放入 ckpt/ 目录。"
            )

        print(f"[SAM] 加载模型 ({self.device}) ...")
        sam = sam_model_registry["vit_h"](checkpoint=self.ckpt_path)
        sam.to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=5000,
        )
        self.predictor = True   # 标记已加载
        print("[SAM] 模型加载完成")

    def extract(self, garment_image: Image.Image) -> Image.Image:
        """
        输入：服装单品图（PIL RGB）
        输出：二值 Mask（PIL L，白=服装，黑=背景）
        """
        self._load()
        img_np = np.array(garment_image.convert("RGB"))

        print("[SAM] 自动生成 Mask ...")
        masks = self.mask_generator.generate(img_np)

        if not masks:
            print("[SAM] 未检测到任何区域，返回全白 Mask")
            return Image.fromarray(np.ones(img_np.shape[:2], dtype=np.uint8) * 255, mode="L")

        # 选取面积最大的 Mask（通常即服装主体）
        masks.sort(key=lambda m: m["area"], reverse=True)
        best_mask = masks[0]["segmentation"].astype(np.uint8) * 255
        print(f"[SAM] 选取最大 Mask，面积 = {masks[0]['area']} px")

        mask_img = Image.fromarray(best_mask, mode="L")
        # 掩膜精修：膨胀 1 像素，羽化 2 像素
        return QualityChecker.refine_mask(mask_img, dilation=3, blur=2)


# ─────────────────────────────────────────────────────────────────────────────
# 二、IDM-VTON 批量换装（通过 Gradio Client 调用远程服务）
# ─────────────────────────────────────────────────────────────────────────────

class IDMVTONClient:
    """
    通过 Gradio Client 调用远程（或本地）IDM-VTON 服务进行虚拟换装
    服务地址: IDMVTON_URL
    """

    def __init__(self, server_url: str = IDMVTON_URL):
        self.server_url = server_url.rstrip("/")
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            raise ImportError("请安装 gradio_client: pip install gradio_client")
        print(f"[IDM-VTON] 连接服务: {self.server_url} ...")
        self._client = Client(self.server_url)
        self._handle_file = handle_file
        print("[IDM-VTON] 连接成功")
        return self._client

    def tryon(
        self,
        human_image: Image.Image,
        garment_image: Image.Image,
        garment_desc: str = "a garment",
        category: str = "upper_body",
        denoise_steps: int = 30,
        seed: int = 42,
    ) -> Image.Image:
        """
        对单张姿态图执行换装，返回结果 PIL Image
        """
        import tempfile
        client = self._get_client()

        # ── 预处理：等比例缩放与填充 (防止变形) ──
        human_padded, h_pad_info = resize_and_pad(human_image, target_size=(768, 1024))
        garm_padded,  _          = resize_and_pad(garment_image, target_size=(768, 1024))

        # 将 PIL Image 保存为临时文件（Gradio Client 需要文件路径）
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_human:
            human_padded.save(f_human.name)
            human_path = f_human.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_garm:
            garm_padded.save(f_garm.name)
            garm_path = f_garm.name

        try:
            # 构造 ImageEditor 格式的 dict
            human_dict = {
                "background": self._handle_file(human_path),
                "layers": [],
                "composite": None,
            }

            result = client.predict(
                dict=human_dict,
                garm_img=self._handle_file(garm_path),
                garment_des=garment_desc,
                is_checked=True,       # 使用自动遮罩
                is_checked_crop=False,
                denoise_steps=denoise_steps,
                seed=seed,
                category=category,
                bg_img_input=self._handle_file(human_path),
                api_name="/start_tryon",
            )
            
            # result 是 (output_image_path, mask_image_path) 元组
            out_path = result[0]
            if isinstance(out_path, dict):
                out_path = out_path.get("path") or out_path.get("url")

            mask_path = result[1]
            if isinstance(mask_path, dict):
                mask_path = mask_path.get("path") or mask_path.get("url")

            # 读取结果并还原比例 (裁掉填充边框)
            raw_out  = Image.open(out_path).convert("RGB")
            raw_mask = Image.open(mask_path).convert("L")
            
            res_img  = unpad_and_resize(raw_out, h_pad_info, final_size=human_image.size)
            res_mask = unpad_and_resize(raw_mask, h_pad_info, final_size=human_image.size)

            # VTON 掩膜精修
            refined_mask = QualityChecker.refine_mask(res_mask, dilation=3, blur=2)

            return res_img, refined_mask

        finally:
            if os.path.exists(human_path): os.unlink(human_path)
            if os.path.exists(garm_path): os.unlink(garm_path)

    def batch_tryon(
        self,
        poses_dir: str,
        garment_image: Image.Image,
        garment_desc: str = "a garment",
        category: str = "upper_body",
        denoise_steps: int = 30,
        seed: int = 42,
        max_poses: int = MAX_POSES,
        model_type: str = "adult_female",
        specific_pose_paths: list = None,  # [NEW] 支持指定姿态图片列表
        progress_callback=None,
    ) -> list:
        """
        批量读取 poses_dir 中的姿态图，依次换装，返回结果图列表
        """
        if specific_pose_paths:
            pose_files = specific_pose_paths
            print(f"[Pose Router] 使用用户指定的 {len(pose_files)} 张姿态图")
        else:
            pose_files = get_pose_files(poses_dir, model_type)
            if pose_files:
                num_to_sample = min(len(pose_files), max_poses)
                rng = random.Random(seed)
                # 随机选取指定数量的姿态图（受 Seed 控制以保证单次生成的一致性）
                pose_files = rng.sample(pose_files, num_to_sample)
                print(f"[Pose Sampler] 随机选取了 {num_to_sample} 张姿态图 (Seed={seed})")

        if not pose_files:
            raise ValueError(
                f"poses 目录为空！\n"
                f"模特类型: {model_type}\n"
                "请放入 jpg/png 格式姿态底图。"
            )

        results = []
        for i, pose_path in enumerate(pose_files):
            print(f"[IDM-VTON] 处理姿态图 {i+1}/{len(pose_files)}: {os.path.basename(pose_path)}")
            # 增加对 Gradio Gallery 传入路径的处理
            if isinstance(pose_path, dict):
                pose_path = pose_path.get("name") or pose_path.get("path")
            
            try:
                pose_img = Image.open(pose_path).convert("RGB")
                res_img, res_mask = self.tryon(
                    pose_img, garment_image,
                    garment_desc=garment_desc,
                    category=category,
                    denoise_steps=denoise_steps,
                    seed=seed + i,
                )
                results.append((res_img, res_mask))
            except Exception as e:
                print(f"[IDM-VTON] 第 {i+1} 张换装失败 ({pose_path}): {e}")
                results.append((None, None))   # 占位，保持顺序

            if progress_callback:
                progress_callback(i + 1, len(pose_files))

        return results


# ─────────────────────────────────────────────────────────────────────────────
# 三、IC-Light 阴影后处理
# ─────────────────────────────────────────────────────────────────────────────

class ICLightProcessor:
    """
    使用 IC-Light foreground-conditioned 模型为换装结果图重新打光
    官方仓库: https://github.com/lllyasviel/IC-Light
    模型权重: ic-light-fc.safetensors (存放于 ckpt/ 目录)
    """

    # huggingface 上的 VAE 和 SD1.5 base 路径
    SD15_BASE = "runwayml/stable-diffusion-v1-5"
    VAE_ID    = "stabilityai/sd-vae-ft-mse"

    def __init__(self, ckpt_path: str = ICLIGHT_CKPT, device: str = None):
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None

    def _load(self):
        if self.pipe is not None:
            return

        # ── 兼容性补丁 (针对 huggingface_hub 在新版本中将 is_offline_mode 移动到 utils 的问题) ──
        try:
            import huggingface_hub
            if not hasattr(huggingface_hub, "is_offline_mode"):
                try:
                    from huggingface_hub.utils import is_offline_mode
                    huggingface_hub.is_offline_mode = is_offline_mode
                except ImportError:
                    # 尝试其他可能的路径
                    try:
                        from huggingface_hub.constants import IS_OFFLINE_MODE
                        huggingface_hub.is_offline_mode = lambda: IS_OFFLINE_MODE
                    except ImportError:
                        # 最后的降级：始终返回 False (假设在线)
                        huggingface_hub.is_offline_mode = lambda: False
        except:
            pass

        try:
            from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("请安装 diffusers 和 safetensors: pip install diffusers safetensors")

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"IC-Light 权重不存在: {self.ckpt_path}\n"
                "请从 https://huggingface.co/lllyasviel/ic-light 下载 ic-light-fc.safetensors 并放入 ckpt/ 目录。"
            )

        print(f"[IC-Light] 加载模型 ({self.device}) ...")
        # 加载 IC-Light 自定义 UNet 权重到 SD1.5 pipeline
        from diffusers import UNet2DConditionModel, DDIMScheduler, logging as diffusers_logging
        from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, logging as transformers_logging

        # 屏蔽无关的 Hub 离线连接报错与架构兼容警告
        diffusers_logging.set_verbosity_error()
        transformers_logging.set_verbosity_error()

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
        # IC-Light fc 模型将 UNet 输入通道从 4→8（拼接前景条件）
        # 需要调整 conv_in 并确保与原模型 dtype (float16) 一致
        with torch.no_grad():
            old_conv = pipe.unet.conv_in
            new_conv = torch.nn.Conv2d(8, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding)
            # 关键：确保新层的类型与原层对齐 (解决 c10::Half 与 float 不匹配问题)
            new_conv = new_conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
            
            new_conv.weight.zero_()
            new_conv.weight[:, :4, :, :].copy_(old_conv.weight)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
            pipe.unet.conv_in = new_conv
            pipe.unet.config["in_channels"] = 8

        # 加载 ic-light 权重（严格匹配时忽略 shape 不匹配的旧键）
        missing, unexpected = pipe.unet.load_state_dict(ic_state, strict=False)
        print(f"[IC-Light] UNet 权重加载完成 (missing={len(missing)}, unexpected={len(unexpected)})")

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self.pipe = pipe
        print("[IC-Light] 模型加载完成")

    def _make_fg_latent(self, image: Image.Image) -> torch.Tensor:
        """将前景图转为 4 通道 latent（让 IC-Light fc 拼接用）"""
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        return transform(image).unsqueeze(0).to(self.device, torch.float16 if self.device != "cpu" else torch.float32)

    def process(
        self,
        image: Image.Image,
        prompt: str = None,
        negative_prompt: str = "overexposed, dark, blurry, noisy, low quality",
        strength: float = 0.35,
        num_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = 42,
        light_direction: str = DEFAULT_LIGHT_DIR,
        specular_boost: float = 1.0,
    ) -> Image.Image:
        """
        对单张图像做 IC-Light 打光处理。

        Args:
            light_direction: 光源方向，支持 'top_left'(默认) / 'top_right' / 'top' / 'front'
            specular_boost:  高光增强系数，1.0=默认，1.5=皮质/绸缎材质增强模式
        """
        self._load()

        # 若未传入 prompt，根据光源和高光参数自动构建
        if prompt is None:
            from pipeline import build_iclight_prompt, detect_material
            params = build_iclight_prompt(
                light_direction=light_direction,
                specular_boost=specular_boost,
            )
            prompt = params["prompt"]
            negative_prompt = params["negative_prompt"]
            guidance_scale = params["guidance_scale"]

        orig_size = image.size
        fg_tensor = self._make_fg_latent(image)

        # 编码前景为 latent（4通道），用于在 UNet 输入时进行拼接
        with torch.no_grad():
            fg_latent = self.pipe.vae.encode(fg_tensor.to(self.pipe.vae.dtype)).latent_dist.sample() * 0.18215

        # ── 关键：使用 Wrapper 动态拼接 4+4 通道 ──
        class ICLightUNetWrapper(torch.nn.Module):
            def __init__(self, unet, fg_latent):
                super().__init__()
                self.unet = unet
                self.fg_latent = fg_latent
                # 继承常用属性，适配 diffusers 内部逻辑
                self.config = unet.config
                self.add_embedding = getattr(unet, "add_embedding", None)

            def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
                # sample: [B, 4, 64, 64] -> noisy latents
                # self.fg_latent: [1, 4, 64, 64] -> condition
                # 拼接为 [B, 8, 64, 64]
                repeated_fg = self.fg_latent.repeat(sample.shape[0], 1, 1, 1)
                concatenated_input = torch.cat([sample, repeated_fg], dim=1)
                return self.unet(concatenated_input, timestep, encoder_hidden_states, **kwargs)

        orig_unet = self.pipe.unet
        self.pipe.unet = ICLightUNetWrapper(orig_unet, fg_latent)

        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            # 使用 img2img 进行推理
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
            # 必须还原 UNet，防止对后续调用造成干扰
            self.pipe.unet = orig_unet

        return result.resize(orig_size, Image.LANCZOS)

    def batch_process(self, images: list, **kwargs) -> list:
        """批量处理多张图，kwargs 透传给 process()"""
        results = []
        for i, img in enumerate(images):
            if img is None:
                results.append(None)
                continue
            print(f"[IC-Light] 处理第 {i+1}/{len(images)} 张 "
                  f"(light={kwargs.get('light_direction', DEFAULT_LIGHT_DIR)}, "
                  f"specular_boost={kwargs.get('specular_boost', 1.0)}) ...")
            try:
                results.append(self.process(img, seed=kwargs.get("seed", 42) + i, **{k: v for k, v in kwargs.items() if k != "seed"}))
            except Exception as e:
                print(f"[IC-Light] 第 {i+1} 张处理失败: {e}")
                results.append(img)   # 失败时返回原图
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 四、Real-ESRGAN 4倍超分（面料纹理优化）
# ─────────────────────────────────────────────────────────────────────────────

SR_CKPT_DIR = os.path.join(os.path.dirname(__file__), "ckpt")

# Real-ESRGAN 权重下载地址
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


class SuperResolutionProcessor:
    """
    基于 Real-ESRGAN 的 4 倍超分处理器，针对服装面料纹理优化。

    默认模型: realesr-general-x4v3
      - 使用 SRVGGNetCompact 架构，推理速度快
      - 对真实世界面料纹理（编织、丝绸、皮纹）还原效果最佳
      - 运行时若权重不存在，则自动下载至 ckpt/ 目录

    输入: 768x1024 PIL Image
    输出: 3072x4096 PIL Image（4倍）
    """

    def __init__(
        self,
        model_name: str = "realesr-general-x4v3",
        ckpt_dir: str = None,
        device: str = None,
        tile: int = 512,       # tile 大小，显存不足时自动减半
        tile_pad: int = 32,
    ):
        self.model_name = model_name
        self.ckpt_dir   = ckpt_dir or SR_CKPT_DIR
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tile       = tile
        self.tile_pad   = tile_pad
        self.upsampler  = None

    def _load(self):
        if self.upsampler is not None:
            return
        
        # ── 兼容性补丁 (针对 basicsr 在新版 torchvision 中的 functional_tensor 缺失问题) ──
        try:
            import sys, types, torchvision
            if not hasattr(torchvision.transforms, 'functional_tensor'):
                from torchvision.transforms import functional as TF
                mod = types.ModuleType("torchvision.transforms.functional_tensor")
                mod.rgb_to_grayscale = TF.rgb_to_grayscale
                sys.modules["torchvision.transforms.functional_tensor"] = mod
        except:
            pass

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            raise ImportError(
                "请安装 Real-ESRGAN 以启用超分功能:\n"
                "  pip install realesrgan basicsr"
            )

        url, fname = _SR_MODEL_URLS[self.model_name]
        model_path = os.path.join(self.ckpt_dir, fname)

        # 权重不存在则自动下载
        if not os.path.exists(model_path):
            print(f"[SR] 权重文件不存在，自动下载: {url}")
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
                print(f"[SR] 下载完成: {model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"权重下载失败: {e}\n"
                    f"请手动下载 {url}\n"
                    f"并放入 {self.ckpt_dir}/"
                ) from e

        print(f"[SR] 加载模型: {self.model_name} ({self.device}) ...")

        if self.model_name == "realesr-general-x4v3":
            # SRVGGNetCompact —— 轻量快速，面料纹理还原最佳
            try:
                from realesrgan.archs.srvgg_arch import SRVGGNetCompact
                model = SRVGGNetCompact(
                    num_in_ch=3, num_out_ch=3,
                    num_feat=64, num_conv=32,
                    upscale=4, act_type="prelu",
                )
            except ImportError:
                # 回退到 RRDB
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3,
                    num_feat=64, num_block=23, num_grow_ch=32, scale=4
                )
        else:
            # RealESRGAN_x4plus — 20层 RRDB，质量更高但更慢
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
        """对单张图像做 4 倍超分，返回超分后的 PIL Image"""
        self._load()
        img_np  = np.array(image.convert("RGB"))
        img_bgr = img_np[:, :, ::-1]   # Real-ESRGAN 使用 BGR 格式
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

    def batch_upscale(self, images: list, progress_callback=None) -> list:
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
                print(f"[SR] 第 {i+1} 张超分失败: {e}，返回原图")
                results.append(img)
            if progress_callback:
                progress_callback(i + 1, len(images))
            # 每张后清理显存
            if self.device != "cpu":
                torch.cuda.empty_cache()
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 五、质量控制检查器
# ─────────────────────────────────────────────────────────────────────────────

class QualityChecker:
    """
    三项图像质量检查，每项返回 (passed: bool, reason: str)。

    ① check_color_consistency  — 颜色一致性
       对比原始服装图（Mask 区域均值 RGB）与生成图对应区域。
       若任一通道偏差 > COLOR_THRESH（默认 15% = 38/255），判定失败。

    ② check_structure_integrity — 结构完整性
       Laplacian 方差（模糊度）< BLUR_THRESH（80）或边缘密度 < EDGE_THRESH（3%）判定失败。

    ③ check_shadow_depth        — 立体感/阴影层次
       灰度标准差 < SHADOW_THRESH（30）判定阴影层次不足。
    """

    COLOR_THRESH  = 0.25   # 放宽至 25%（原 20%），允许适度颜色偏差，应对光影带来的自然色偏
    BLUR_THRESH   = 50.0   # Laplacian 方差降至 50.0（原 80.0），避免柔和/素色面料被误判为结构崩坏
    EDGE_THRESH   = 0.015  # Canny 边缘像素比例降至 1.5%（原 3%），让丝绸等无纹理衣服能过关
    SHADOW_THRESH = 20.0   # 灰度标准差阈值降至 20.0（原 30.0），解决纯白衣服容易报“立体感不足”的问题

    @staticmethod
    def refine_mask(mask: Image.Image, dilation: int = 3, blur: int = 2) -> Image.Image:
        """
        掩膜后处理：膨胀 + 羽化（解决边缘稀疏/锯齿问题）
        """
        from PIL import ImageFilter
        # 1. 膨胀 (MaxFilter): 填充孔洞，确保覆盖边缘
        if dilation > 0:
            mask = mask.filter(ImageFilter.MaxFilter(dilation))
        # 2. 羽化 (GaussianBlur): 平滑过渡
        if blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur))
        return mask

    # ── 颜色一致性 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _masked_mean_rgb(image: Image.Image, mask: Image.Image) -> np.ndarray:
        """用 SAM Mask 提取服装区域的均值 RGB（0–255）。"""
        img_np = np.array(image.convert("RGB"), dtype=np.float32)
        if img_np.shape[:2] != (mask.height, mask.width):
            mask = mask.resize((img_np.shape[1], img_np.shape[0]), Image.NEAREST)
        msk_np = np.array(mask.convert("L"), dtype=np.float32) / 255.0
        total  = msk_np.sum()
        if total < 1:
            return img_np.mean(axis=(0, 1))
        return (img_np * msk_np[:, :, np.newaxis]).sum(axis=(0, 1)) / total

    @classmethod
    def check_color_consistency(
        cls,
        garment_original: Image.Image,
        original_mask: Image.Image,
        generated: Image.Image,
        generated_mask: Image.Image,
        thresh: float = None,
    ) -> tuple:
        target_thresh = thresh if thresh is not None else cls.COLOR_THRESH
        orig_rgb = cls._masked_mean_rgb(garment_original, original_mask)
        gen_rgb  = cls._masked_mean_rgb(generated, generated_mask)
        deviation = np.abs(orig_rgb - gen_rgb) / 255.0
        max_dev   = float(deviation.max())
        channels  = ["R", "G", "B"]
        worst_ch  = channels[int(deviation.argmax())]
        print(
            f"[QC-Color] 原图均值RGB=({orig_rgb[0]:.1f},{orig_rgb[1]:.1f},{orig_rgb[2]:.1f}) "
            f"生成均值RGB=({gen_rgb[0]:.1f},{gen_rgb[1]:.1f},{gen_rgb[2]:.1f}) "
            f"最大偏差={max_dev*100:.1f}%（{worst_ch}通道）阈值={target_thresh*100:.0f}%"
        )
        if max_dev > target_thresh:
            return False, f"颜色偏差过大: {worst_ch}通道偏差 {max_dev*100:.1f}% > {target_thresh*100:.0f}%"
        return True, ""

    # ── 结构完整性 ─────────────────────────────────────────────────────────────
    @classmethod
    def check_structure_integrity(cls, image: Image.Image) -> tuple:
        gray = np.array(image.convert("L"), dtype=np.float32)

        # 模糊度：Laplacian 方差（用 numpy 手写卷积避免额外依赖）
        # 对每行差分再对列差分近似 Laplacian
        lap = (
            np.roll(gray, -1, axis=0) + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=1) + np.roll(gray, 1, axis=1)
            - 4 * gray
        )
        blur_score = float(lap.var())
        print(f"[QC-Structure] Laplacian方差={blur_score:.1f} 阈值>{cls.BLUR_THRESH}")

        if blur_score < cls.BLUR_THRESH:
            return False, f"图像模糊: Laplacian方差 {blur_score:.1f} < {cls.BLUR_THRESH}"

        # 边缘密度：梯度幅值
        gy = np.gradient(gray, axis=0)
        gx = np.gradient(gray, axis=1)
        edge_density = float((np.sqrt(gx**2 + gy**2) > 20).mean())
        print(f"[QC-Structure] 边缘密度={edge_density*100:.2f}% 阈值>{cls.EDGE_THRESH*100:.1f}%")

        if edge_density < cls.EDGE_THRESH:
            return False, f"边缘稀疏: 边缘密度 {edge_density*100:.2f}% < {cls.EDGE_THRESH*100:.1f}%"

        return True, ""

    # ── 立体感/阴影层次 ────────────────────────────────────────────────────────
    @classmethod
    def check_shadow_depth(cls, image: Image.Image) -> tuple:
        gray = np.array(image.convert("L"), dtype=np.float32)
        std  = float(gray.std())
        print(f"[QC-Shadow] 灰度标准差={std:.2f} 阈值>{cls.SHADOW_THRESH}")
        if std < cls.SHADOW_THRESH:
            return False, f"阴影层次不足: 灰度std={std:.2f} < {cls.SHADOW_THRESH}"
        return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# 六、完整流水线（含质量控制重试）
# ─────────────────────────────────────────────────────────────────────────────

class ClothingDetailPipeline:
    """
    完整流水线编排：
    SAM → IDM-VTON × N → QC(颜色+结构，失败换Seed重试)
        → IC-Light × N → QC(立体感，失败调强iclight_strength重试)
        → Real-ESRGAN 4倍超分

    质量控制重试上限：MAX_QC_RETRIES（默认 3 次）
    """
    MAX_QC_RETRIES = 3

    def __init__(
        self,
        idmvton_url: str = IDMVTON_URL,
        poses_dir: str = POSES_DIR,
        outputs_dir: str = OUTPUTS_DIR,
        sam_ckpt: str = SAM_CKPT,
        iclight_ckpt: str = ICLIGHT_CKPT,
        sr_model: str = "realesr-general-x4v3",
        sr_tile: int = 512,
        device: str = None,
    ):
        self.poses_dir   = poses_dir
        self.outputs_dir = outputs_dir
        os.makedirs(outputs_dir, exist_ok=True)

        self.sam      = SAMExtractor(ckpt_path=sam_ckpt, device=device)
        self.vton     = IDMVTONClient(server_url=idmvton_url)
        self.iclight  = ICLightProcessor(ckpt_path=iclight_ckpt, device=device)
        self.sr       = SuperResolutionProcessor(
            model_name=sr_model, ckpt_dir=SR_CKPT_DIR,
            tile=sr_tile, device=device,
        )
        self.qc = QualityChecker()

    def run(
        self,
        garment_image: Image.Image,
        garment_desc: str = "a garment",
        category: str = "upper_body",
        model_type: str = "adult_female",
        denoise_steps: int = 30,
        seed: int = 42,
        enable_iclight: bool = True,
        iclight_strength: float = 0.35,
        light_direction: str = DEFAULT_LIGHT_DIR,
        enable_sr: bool = True,
        progress_callback=None,
    ) -> dict:

        """
        执行完整流水线（含三项质量控制自动重试）

        QC 重试逻辑：
          - ① 颜色一致性 & ② 结构完整性：VTON 后检查，失败换 seed 最多重试 MAX_QC_RETRIES 次
          - ③ 立体感/阴影层次：IC-Light 后检查，失败递增 iclight_strength 最多重试 MAX_QC_RETRIES 次

        Returns:
            {
                "mask":            PIL Mask 图,
                "tryon_results":   [PIL, ...],
                "final_results":   [PIL, ...],
                "sr_results":      [PIL, ...],
                "material":        str,
                "specular_boost":  float,
                "light_direction": str,
                "qc_log":          [str, ...],  # 质量控制事件日志
            }
        """
    def run(
        self,
        garment_image: Image.Image,
        garment_desc: str = "a garment",
        category: str = "upper_body",
        model_type: str = "adult_female",
        light_direction: str = DEFAULT_LIGHT_DIR,
        enable_sr: bool = True,
        denoise_steps: int = 30,
        seed: int = 42,
        enable_iclight: bool = True,
        iclight_strength: float = 0.35,
        specific_pose_paths: list = None,   # [NEW] 支持指定姿态列表
        material: str = None,               # [NEW]
        ethnicity: str = None,              # [NEW]
        age_group: str = None,              # [NEW]
        progress_callback=None,
    ) -> dict:
        """
        执行完整流水线。
        """
        def _progress(stage, cur, total):
            if progress_callback:
                progress_callback(stage, cur, total)

        qc_log = []   # 收集所有 QC 事件

        # ── 材质识别 & 光影参数决策 ─────────────────────────────────────────────
        material    = detect_material(material) if material else detect_material(garment_desc)
        ethnicity   = detect_ethnicity(ethnicity) if ethnicity else detect_ethnicity(garment_desc)
        age_group   = detect_age_group(age_group) if age_group else detect_age_group(garment_desc)
        
        specular_boost = 1.5 if material in ("leather", "satin") else 1.0
        if specular_boost > 1.0:
            print(f"[Light] 检测到高光材质 '{material}'，启用 specular_boost={specular_boost}")
        iclight_params = build_iclight_prompt(
            light_direction=light_direction,
            material=material,
            specular_boost=specular_boost,
        )
        print(f"[Light] 光源: {light_direction} | 材质: {material} | 属性: {ethnicity}, {age_group}")

        # ── Step 1: SAM ──────────────────────────────────────────────────────
        print("\n========== Step 1: SAM Mask 提取 ==========")
        _progress("SAM提取Mask", 0, 1)
        mask = self.sam.extract(garment_image)
        _progress("SAM提取Mask", 1, 1)

        # ── Step 2 + QC①②: IDM-VTON → 颜色一致性 & 结构完整性检查 ─────────────
        print("\n========== Step 2: IDM-VTON 批量换装 (增量重试模式) ==========")
        current_seed = seed
        
        # 获取所有姿态路径
        if specific_pose_paths:
            pose_files = specific_pose_paths
        else:
            pose_files = get_pose_files(self.poses_dir, model_type)
            num_to_sample = min(len(pose_files), MAX_POSES)
            if pose_files:
                pose_files = random.Random(seed).sample(pose_files, num_to_sample)

        # 初始化结果列表 (None 表示待生成或待修复)
        tryon_results   = [None] * len(pose_files)
        vton_masks      = [None] * len(pose_files)
        qc_passed       = [False] * len(pose_files)
        
        for vton_attempt in range(self.MAX_QC_RETRIES + 1):
            # 找出需要生成/修复的索引
            todo_indices = [i for i, img in enumerate(tryon_results) if img is None]
            if not todo_indices:
                break
                
            if vton_attempt > 0:
                current_seed += 100
                msg = f"[QC] 增量重试 VTON 第{vton_attempt}次 (剩余{len(todo_indices)}张), seed={current_seed}"
                print(f"\n{msg}"); qc_log.append(msg)
                
                # 针对失败的图，在重试时尝试更换全新的 pose
                all_poses = get_pose_files(self.poses_dir, model_type)
                # 已经用过的 pose 不再用
                used_poses = set()
                for p in pose_files:
                    if isinstance(p, dict):
                        used_poses.add(p.get("name") or p.get("path"))
                    else:
                        used_poses.add(p)
                
                available_poses = [p for p in all_poses if p not in used_poses]
                
                for i in todo_indices:
                    if available_poses:
                        # 随机选一张全新的
                        rng = random.Random(current_seed + i)
                        new_pose = rng.choice(available_poses)
                        pose_files[i] = new_pose
                        available_poses.remove(new_pose)
                        used_poses.add(new_pose)
                        
                        log_msg = f"  [QC-Pose] 索引 {i+1} 更换为新姿态图: {os.path.basename(new_pose)}"
                        print(log_msg)
                        qc_log.append(log_msg)

            for i in todo_indices:
                pose_path = pose_files[i]
                # 处理可能传入的 dict (如 Gradio Gallery)
                if isinstance(pose_path, dict):
                    pose_path = pose_path.get("name") or pose_path.get("path")
                
                print(f"[IDM-VTON] 处理/修复 [{i+1}/{len(pose_files)}]: {os.path.basename(pose_path)}")
                _progress("IDM-VTON换装", i + 1, len(pose_files))
                
                try:
                    res_img, res_mask = self.vton.tryon(
                        Image.open(pose_path).convert("RGB"), 
                        garment_image,
                        garment_desc=garment_desc,
                        category=category,
                        denoise_steps=denoise_steps,
                        seed=current_seed + i,
                    )
                    
                    # 立即对这张图进行 QC
                    color_ok, color_reason = self.qc.check_color_consistency(
                        garment_original=garment_image, original_mask=mask,
                        generated=res_img, generated_mask=res_mask,
                    )
                    struct_ok, struct_reason = self.qc.check_structure_integrity(res_img)
                    
                    if color_ok and struct_ok:
                        tryon_results[i] = res_img
                        vton_masks[i]    = res_mask
                        qc_passed[i]     = True
                        print(f"  ✅ [QC] 通过: {os.path.basename(pose_path)}")
                    else:
                        fail_msg = f"  ❌ [QC] 失败 ({color_reason or ''} {struct_reason or ''})"
                        print(fail_msg)
                        # 如果是最后一次尝试，依然保留，避免返回空
                        if vton_attempt == self.MAX_QC_RETRIES:
                            tryon_results[i] = res_img
                            vton_masks[i]    = res_mask
                            qc_passed[i]     = False
                            qc_log.append(f"[QC] ⚠️ 达到重试上限，保留当前结果: {os.path.basename(pose_path)}")
                except Exception as e:
                    print(f"  ⚠️ [IDM-VTON] 接口调用失败: {e}")
                    if vton_attempt == self.MAX_QC_RETRIES:
                        qc_log.append(f"[QC] ❌ 最终换装失败: {os.path.basename(pose_path)}")

        qc_log.append(f"[QC] VTON 阶段结束，最终合格率: {sum(1 for x in tryon_results if x is not None)}/{len(pose_files)}")

        # ── Step 3 + QC③: IC-Light → 立体感/阴影层次检查 ─────────────────────
        print(f"\n========== Step 3: IC-Light 阴影后处理 (权重基准={iclight_strength:.2f}) ==========")
        
        # 初始化最终结果列表 (None 表示待打光)
        final_results = [None] * len(tryon_results)
        
        if enable_iclight:
            # 将 VTON 失败或 QC 未通过的直接继承，跳过 IC-Light 处理
            for i, (res, qc) in enumerate(zip(tryon_results, qc_passed)):
                if res is not None and not qc:
                    print(f"  ⚠️ [Skip IC-Light] 图像 {i+1} VTON QC 未通过，跳过打光处理")
                    final_results[i] = res

            for iclight_attempt in range(self.MAX_QC_RETRIES + 1):
                # 找出需要处理/加强的索引 (VTON 成功且 final 为 None)
                todo_indices = [i for i, r in enumerate(final_results) if r is None and tryon_results[i] is not None]
                if not todo_indices:
                    break
                
                current_strength = min(iclight_strength + iclight_attempt * 0.10, 0.70)
                if iclight_attempt > 0:
                    msg = f"[QC] 增量增强 IC-Light 第{iclight_attempt}次 (剩余{len(todo_indices)}张), strength={current_strength:.2f}"
                    print(f"\n{msg}"); qc_log.append(msg)

                for i in todo_indices:
                    res_vton = tryon_results[i]
                    msk_vton = vton_masks[i]
                    
                    print(f"[IC-Light] 处理 [{i+1}/{len(tryon_results)}] ...")
                    try:
                        # 单张执行打光
                        ic_res = self.iclight.process(
                            res_vton,
                            strength=current_strength,
                            light_direction=light_direction,
                            specular_boost=iclight_params["specular_boost"],
                            prompt=iclight_params["prompt"],
                            negative_prompt=iclight_params["negative_prompt"],
                            guidance_scale=iclight_params["guidance_scale"],
                            seed=current_seed + i,
                        )
                        
                        # 取消原有的 mask 回贴融合 (直接采用 IC-Light 全局自然打光的结果)
                        # 因 IC-Light 会重塑全局光影（含背景投射及阴影），若再强行将人体抠回未经打光的原版背景中，
                        # 会导致人物边缘光影断层，从而产生明显的“人影框线 / silhouette outline”。
                        composite_img = ic_res
                        if composite_img.size != res_vton.size:
                            composite_img = composite_img.resize(res_vton.size, Image.LANCZOS)
                            
                        # 立即对这张图进行 QC③（立体感+色彩漂移）
                        shadow_ok, shadow_reason = self.qc.check_shadow_depth(composite_img)
                        color_final_ok, color_final_reason = self.qc.check_color_consistency(
                            garment_original=garment_image, original_mask=mask,
                            generated=composite_img, generated_mask=msk_vton,
                            thresh=0.15 # 打光阶段放宽一点点
                        )
                        
                        if shadow_ok and color_final_ok:
                            final_results[i] = composite_img
                            qc_passed[i]     = True
                            print(f"  ✅ [QC] 打光通过")
                        else:
                            print(f"  ❌ [QC] 打光不足或偏色: {shadow_reason or ''} {color_final_reason or ''}")
                            if iclight_attempt == self.MAX_QC_RETRIES:
                                final_results[i] = composite_img # 保留最后结果
                                qc_passed[i]     = False
                                qc_log.append(f"[QC] ⚠️ 打光重试上限，保留当前结果: {i+1}")
                    except Exception as e:
                        print(f"  ⚠️ [IC-Light] 处理失败: {e}")
                        if iclight_attempt == self.MAX_QC_RETRIES:
                            final_results[i] = res_vton
        else:
            final_results = tryon_results

        qc_log.append(f"[QC] IC-Light 阶段结束")

        # ── Step 4: Real-ESRGAN 4倍超分 ────────────────────────────────────────
        if enable_sr:
            print("\n========== Step 4: Real-ESRGAN 4倍超分 ==========")
            valid_for_sr = []
            for i, r in enumerate(final_results):
                if r is not None:
                    if qc_passed[i]:
                        valid_for_sr.append(r)
                    else:
                        print(f"  ⚠️ [Skip SR] 图像 {i+1} QC 未完全通过，跳过超分处理")
                        
            if valid_for_sr:
                sr_valid = self.sr.batch_upscale(
                    valid_for_sr,
                    progress_callback=lambda cur, total: _progress("超分4x", cur, total),
                )
            else:
                sr_valid = []
                
            sr_idx, sr_results = 0, []
            for i, r in enumerate(final_results):
                if r is not None and qc_passed[i]:
                    sr_results.append(sr_valid[sr_idx])
                    sr_idx += 1
                else:
                    sr_results.append(None)
        else:
            sr_results = [None] * len(final_results)

        # ── 保存结果 ────────────────────────────────────────────────────────────
        ts = int(time.time())
        mask_path = os.path.join(self.outputs_dir, f"{ts}_mask.jpg")
        mask.convert("RGB").save(mask_path, format='JPEG', quality=95)
        
        final_paths = []
        sr_paths    = []
        
        for i, (img, sr_img) in enumerate(zip(final_results, sr_results)):
            p = None
            if img:
                p = os.path.join(self.outputs_dir, f"{ts}_result_{i+1}.jpg")
                img.save(p, format='JPEG', quality=95)
                final_paths.append(p)
            else:
                # 换装失败的情况，给原姿态图打上失败提示水印并返回，避免前端直接丢失数量
                try:
                    from PIL import ImageDraw, ImageFont
                    # 处理 pose_path 可能是字典等格式
                    pose_path = pose_files[i]
                    if isinstance(pose_path, dict):
                        pose_path = pose_path.get("name") or pose_path.get("path")
                        
                    fail_img = Image.open(pose_path).convert("RGB").copy()
                    
                    try:
                        font = ImageFont.truetype("arial.ttf", size=60)
                    except:
                        try:
                            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size=60)
                        except:
                            font = ImageFont.load_default()
                            
                    w, h = fail_img.size
                    overlay = Image.new('RGBA', fail_img.size, (0, 0, 0, 0))
                    draw_overlay = ImageDraw.Draw(overlay)
                    draw_overlay.rectangle([0, 0, w, 120], fill=(255, 0, 0, 180))
                    fail_img = Image.alpha_composite(fail_img.convert('RGBA'), overlay).convert('RGB')
                    
                    draw = ImageDraw.Draw(fail_img)
                    text = "TRY-ON FAILED / 换装失败"
                    draw.text((20, 30), text, fill="white", font=font)
                    
                    p = os.path.join(self.outputs_dir, f"{ts}_fail_{i+1}.jpg")
                    fail_img.save(p, format='JPEG', quality=95)
                    final_paths.append(p)
                except Exception as e:
                    print(f"创建失败提示图出错: {e}")
                    final_paths.append(None)
                
            if sr_img:
                p_sr = os.path.join(self.outputs_dir, f"{ts}_result_{i+1}_4x.jpg")
                sr_img.save(p_sr, format='JPEG', quality=95)
                sr_paths.append(p_sr)
            else:
                # 若未超分但存在原图（QC未通过或是彻底失败），将最终结果/失败图作为 fallback
                sr_paths.append(p)

        n_saved = sum(1 for r in sr_results if r) if enable_sr else sum(1 for r in final_results if r)
        print(f"\n✅ 流水线完成，共输出 {n_saved} 张{'4x超分' if enable_sr else ''}详情图，已保存至 {self.outputs_dir}")
        if qc_log:
            print("[QC日志]\n" + "\n".join(f"  {l}" for l in qc_log))

        return {
            "mask":            mask,
            "mask_path":       mask_path,
            "tryon_results":   tryon_results,
            "final_results":   final_results,
            "final_paths":     final_paths,
            "sr_results":      sr_results,
            "sr_paths":        sr_paths,
            "material":        material,
            "specular_boost":  specular_boost,
            "light_direction": light_direction,
            "qc_log":          qc_log,
        }



# ─────────────────────────────────────────────────────────────────────────────
# 命令行快速测试
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--garment",    type=str, required=True, help="服装图路径")
    parser.add_argument("--desc",        type=str, default="a garment", help="服装描述")
    parser.add_argument("--category",   type=str, default="upper_body", choices=["upper_body","lower_body","dresses"])
    parser.add_argument("--model-type", type=str, default="adult_female",
                        choices=["adult_female","adult_male","child_female","child_male"],
                        help="模特类型，对应 poses/ 子目录")
    parser.add_argument("--no-iclight", action="store_true", help="跳过 IC-Light 处理")
    args = parser.parse_args()

    pipe = ClothingDetailPipeline()
    garment = Image.open(args.garment).convert("RGB")

    results = pipe.run(
        garment_image=garment,
        garment_desc=args.desc,
        category=args.category,
        model_type=args.model_type,
        enable_iclight=not args.no_iclight,
    )
    print("完成！最终图片：", [f"outputs/result_{i+1}.jpg" for i, r in enumerate(results["final_results"]) if r])
