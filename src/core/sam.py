"""
SAM (Segment Anything Model) 掩膜提取器
从服装单品图中提取服装 Mask
"""
import os
import numpy as np
from PIL import Image
from typing import Optional

try:
    import torch
except ImportError:
    torch = None


class SAMExtractor:
    """使用 Segment Anything Model 从衣服图中提取服装 Mask"""

    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        """
        Args:
            ckpt_path: SAM 模型权重路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.mask_generator = None

    def _load(self):
        """懒加载 SAM 模型"""
        if self.mask_generator is not None:
            return
        
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "请安装 segment-anything: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"SAM 权重文件不存在：{self.ckpt_path}\n"
                "请从 https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth "
                "下载并放入 ckpt/ 目录。"
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
        print("[SAM] 模型加载完成")

    def extract(self, garment_image: Image.Image) -> Image.Image:
        """
        从服装图提取二值 Mask
        
        Args:
            garment_image: 服装单品图（PIL RGB）
            
        Returns:
            二值 Mask（PIL L，白=服装，黑=背景）
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
        # 掩膜精修：膨胀 3 像素，羽化 2 像素
        from src.models.quality import QualityChecker
        return QualityChecker.refine_mask(mask_img, dilation=3, blur=2)
