"""
IDM-VTON 虚拟换装客户端
通过 Gradio Client 调用远程 IDM-VTON 服务
"""
import os
import tempfile
from typing import Tuple, Optional, Dict, Any
from PIL import Image

from src.utils.image_ops import resize_and_pad


class IDMVTONClient:
    """
    通过 Gradio Client 调用远程（或本地）IDM-VTON 服务进行虚拟换装
    """

    def __init__(self, server_url: str):
        """
        Args:
            server_url: IDM-VTON 服务地址
        """
        self.server_url = server_url.rstrip("/")
        self._client = None
        self._handle_file = None

    def _get_client(self):
        """懒加载 Gradio Client"""
        if self._client is not None:
            return self._client
        
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            raise ImportError("请安装 gradio_client: pip install gradio_client")
        
        print(f"[IDM-VTON] 连接服务：{self.server_url} ...")
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
    ) -> Tuple[Image.Image, Image.Image]:
        """
        对单张姿态图执行换装
        
        Args:
            human_image: 人体姿态图
            garment_image: 服装图
            garment_desc: 服装描述
            category: 服装类别
            denoise_steps: 去噪步数
            seed: 随机种子
            
        Returns:
            (结果图，Mask 图)
        """
        client = self._get_client()

        # 预处理：等比例缩放与填充 (防止变形)
        human_padded, _ = resize_and_pad(human_image, target_size=(768, 1024))
        garm_padded, _ = resize_and_pad(garment_image, target_size=(768, 1024))

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

            # 转回 PIL Image
            out_img = Image.open(out_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("L")

            # 还原 padding 前的尺寸
            out_img = unpad_and_resize(out_img, _, final_size=human_image.size)
            mask_img = unpad_and_resize(mask_img, _, final_size=human_image.size)

            return out_img, mask_img

        finally:
            # 清理临时文件
            try:
                os.unlink(human_path)
                os.unlink(garm_path)
            except Exception:
                pass


def get_pose_files(base_poses_dir: str, model_type: str) -> list:
    """
    根据模特类型返回适用的姿态图片路径列表。
    
    Args:
        base_poses_dir: 姿态图根目录
        model_type: 模特类型 (如 'adult_female', 'child_neutral' 等)
        
    Returns:
        姿态图路径列表
    """
    from src.config import settings
    
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    
    def _fetch(sub: str):
        if not sub:
            return []
        candidate = os.path.join(base_poses_dir, sub)
        if not os.path.isdir(candidate):
            return []
        files = []
        for ext in exts:
            import glob
            files.extend(glob.glob(os.path.join(candidate, ext)))
        return files

    files = []
    if model_type.endswith("_neutral"):
        prefix = model_type.split("_")[0]  # "adult" or "child"
        files.extend(_fetch(settings.model_type_dirs.get(f"{prefix}_female", "")))
        files.extend(_fetch(settings.model_type_dirs.get(f"{prefix}_male", "")))
        files.extend(_fetch(settings.model_type_dirs.get(model_type, "")))
        files = list(set(files))  # 去重
    else:
        files = _fetch(settings.model_type_dirs.get(model_type, ""))
        
    if files:
        print(f"[Pose Router] 找到 {len(files)} 张姿态图 (模型类型：{model_type})")
        if model_type.endswith("_neutral"):
            import random
            random.Random(42).shuffle(files)
        return files
        
    # 回退到根目录
    fallback = []
    import glob
    for ext in exts:
        fallback.extend(glob.glob(os.path.join(base_poses_dir, ext)))
    print(f"[Pose Router] 特定子目录无图片，回退到根目录，找到 {len(fallback)} 张图")
    return fallback


# 需要在模块级别导入以避免循环依赖
def unpad_and_resize(padded_image: Image.Image, padding_info: tuple, final_size=None) -> Image.Image:
    """从 utils 导入的辅助函数占位，实际使用来自 utils.image_ops"""
    from src.utils.image_ops import unpad_and_resize as _unpad
    return _unpad(padded_image, padding_info, final_size)
