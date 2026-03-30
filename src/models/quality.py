"""
质量控制检查器
三项图像质量检查：颜色一致性、结构完整性、阴影层次
"""
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Optional


class QualityChecker:
    """
    三项图像质量检查，每项返回 (passed: bool, reason: str)。

    ① check_color_consistency  — 颜色一致性
       对比原始服装图（Mask 区域均值 RGB）与生成图对应区域。
       若任一通道偏差 > COLOR_THRESH（默认 25%），判定失败。

    ② check_structure_integrity — 结构完整性
       Laplacian 方差（模糊度）< BLUR_THRESH（50.0）或边缘密度 < EDGE_THRESH（1.5%）判定失败。

    ③ check_shadow_depth        — 立体感/阴影层次
       灰度标准差 < SHADOW_THRESH（20.0）判定阴影层次不足。
    """

    COLOR_THRESH = 0.25   # 颜色偏差阈值 (25%)
    BLUR_THRESH = 50.0    # Laplacian 方差阈值
    EDGE_THRESH = 0.015   # 边缘像素比例阈值 (1.5%)
    SHADOW_THRESH = 20.0  # 灰度标准差阈值

    @staticmethod
    def refine_mask(mask: Image.Image, dilation: int = 3, blur: int = 2) -> Image.Image:
        """
        掩膜后处理：膨胀 + 羽化（解决边缘稀疏/锯齿问题）
        
        Args:
            mask: 输入二值 Mask
            dilation: 膨胀大小
            blur: 高斯模糊半径
            
        Returns:
            处理后的 Mask
        """
        # 1. 膨胀 (MaxFilter): 填充孔洞，确保覆盖边缘
        if dilation > 0:
            mask = mask.filter(ImageFilter.MaxFilter(dilation))
        # 2. 羽化 (GaussianBlur): 平滑过渡
        if blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur))
        return mask

    @staticmethod
    def _masked_mean_rgb(image: Image.Image, mask: Image.Image) -> np.ndarray:
        """用 SAM Mask 提取服装区域的均值 RGB（0–255）。"""
        img_np = np.array(image.convert("RGB"), dtype=np.float32)
        if img_np.shape[:2] != (mask.height, mask.width):
            mask = mask.resize((img_np.shape[1], img_np.shape[0]), Image.NEAREST)
        msk_np = np.array(mask.convert("L"), dtype=np.float32) / 255.0
        total = msk_np.sum()
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
        thresh: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        检查颜色一致性
        
        Args:
            garment_original: 原始服装图
            original_mask: 原始服装 Mask
            generated: 生成图
            generated_mask: 生成图 Mask
            thresh: 可选的自定义阈值
            
        Returns:
            (passed, reason)
        """
        target_thresh = thresh if thresh is not None else cls.COLOR_THRESH
        orig_rgb = cls._masked_mean_rgb(garment_original, original_mask)
        gen_rgb = cls._masked_mean_rgb(generated, generated_mask)
        deviation = np.abs(orig_rgb - gen_rgb) / 255.0
        max_dev = float(deviation.max())
        channels = ["R", "G", "B"]
        worst_ch = channels[int(deviation.argmax())]
        print(
            f"[QC-Color] 原图均值 RGB=({orig_rgb[0]:.1f},{orig_rgb[1]:.1f},{orig_rgb[2]:.1f}) "
            f"生成均值 RGB=({gen_rgb[0]:.1f},{gen_rgb[1]:.1f},{gen_rgb[2]:.1f}) "
            f"最大偏差={max_dev*100:.1f}%（{worst_ch}通道）阈值={target_thresh*100:.0f}%"
        )
        if max_dev > target_thresh:
            return False, f"颜色偏差过大：{worst_ch}通道偏差 {max_dev*100:.1f}% > {target_thresh*100:.0f}%"
        return True, ""

    @classmethod
    def check_structure_integrity(cls, image: Image.Image) -> Tuple[bool, str]:
        """
        检查结构完整性（模糊度和边缘密度）
        
        Args:
            image: 待检查图像
            
        Returns:
            (passed, reason)
        """
        gray = np.array(image.convert("L"), dtype=np.float32)

        # 模糊度：Laplacian 方差（用 numpy 手写卷积避免额外依赖）
        lap = (
            np.roll(gray, -1, axis=0) + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=1) + np.roll(gray, 1, axis=1)
            - 4 * gray
        )
        blur_score = float(lap.var())
        print(f"[QC-Structure] Laplacian 方差={blur_score:.1f} 阈值>{cls.BLUR_THRESH}")

        if blur_score < cls.BLUR_THRESH:
            return False, f"图像模糊：Laplacian 方差 {blur_score:.1f} < {cls.BLUR_THRESH}"

        # 边缘密度：梯度幅值
        gy = np.gradient(gray, axis=0)
        gx = np.gradient(gray, axis=1)
        edge_density = float((np.sqrt(gx**2 + gy**2) > 20).mean())
        print(f"[QC-Structure] 边缘密度={edge_density*100:.2f}% 阈值>{cls.EDGE_THRESH*100:.1f}%")

        if edge_density < cls.EDGE_THRESH:
            return False, f"边缘稀疏：边缘密度 {edge_density*100:.2f}% < {cls.EDGE_THRESH*100:.1f}%"

        return True, ""

    @classmethod
    def check_shadow_depth(cls, image: Image.Image) -> Tuple[bool, str]:
        """
        检查阴影层次（立体感）
        
        Args:
            image: 待检查图像
            
        Returns:
            (passed, reason)
        """
        gray = np.array(image.convert("L"), dtype=np.float32)
        std = float(gray.std())
        print(f"[QC-Shadow] 灰度标准差={std:.2f} 阈值>{cls.SHADOW_THRESH}")
        if std < cls.SHADOW_THRESH:
            return False, f"阴影层次不足：灰度 std={std:.2f} < {cls.SHADOW_THRESH}"
        return True, ""
