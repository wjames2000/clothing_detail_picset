"""
质量控制检查器
三项图像质量检查：颜色一致性、结构完整性、阴影层次
优化版本：增强颜色保真度，确保生成图片尽可能接近原始服装
"""
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Optional
import cv2


class QualityChecker:
    """
    三项图像质量检查（优化版），每项返回 (passed: bool, reason: str)。

    ① check_color_consistency  — 颜色一致性（增强版）
       使用 Lab 色彩空间对比原始服装图与生成图，Lab 空间更符合人眼感知。
       若 ΔE*ab（CIE76 公式）> COLOR_DE_THRESH（默认 15.0），判定失败。
       同时保留 RGB 通道偏差检查作为辅助，任一通道偏差 > COLOR_RGB_THRESH（15%）也失败。

    ② check_structure_integrity — 结构完整性
       Laplacian 方差（模糊度）< BLUR_THRESH（50.0）或边缘密度 < EDGE_THRESH（1.5%）判定失败。

    ③ check_shadow_depth        — 立体感/阴影层次
       灰度标准差 < SHADOW_THRESH（20.0）判定阴影层次不足。
       
    ④ apply_color_correction     — 颜色校正（新增）
       对生成图像进行直方图匹配，使其颜色分布更接近原始服装。
    """

    # Lab 色彩空间 ΔE 阈值 (CIE76 公式，15 表示人眼可察觉的最小差异)
    COLOR_DE_THRESH = 15.0
    
    # RGB 通道偏差阈值 (从 25% 降至 15%，更严格)
    COLOR_RGB_THRESH = 0.15
    
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
    def _rgb2lab(rgb: np.ndarray) -> np.ndarray:
        """
        将 RGB 数组转换为 Lab 色彩空间 (使用 OpenCV 标准转换)
        
        Args:
            rgb: RGB 数组，值域 [0, 255]
            
        Returns:
            Lab 数组，L:[0,100], a:[-128,127], b:[-128,127]
        """
        # OpenCV 需要 BGR 格式且值域归一化到 [0,1]
        # 先确保是 float32 并归一化
        rgb_normalized = rgb.astype(np.float32) / 255.0
        # OpenCV 的 cvtColor 期望 BGR 顺序输入
        bgr_normalized = rgb_normalized[:, :, ::-1]
        # 转换到 Lab，OpenCV 输出 L:[0,100], a:[-128,127], b:[-128,127]
        lab = cv2.cvtColor(bgr_normalized, cv2.COLOR_RGB2LAB)
        return lab

    @staticmethod
    def _masked_mean_lab(image: Image.Image, mask: Image.Image) -> np.ndarray:
        """用 Mask 提取服装区域的均值 Lab 值。"""
        img_np = np.array(image.convert("RGB"), dtype=np.float32)
        if img_np.shape[:2] != (mask.height, mask.width):
            mask = mask.resize((img_np.shape[1], img_np.shape[0]), Image.NEAREST)
        msk_np = np.array(mask.convert("L"), dtype=np.float32) / 255.0
        
        # 计算加权 Lab 均值
        lab_img = QualityChecker._rgb2lab(img_np)
        total = msk_np.sum()
        if total < 1:
            return lab_img.mean(axis=(0, 1))
        return (lab_img * msk_np[:, :, np.newaxis]).sum(axis=(0, 1)) / total

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

    @staticmethod
    def _delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> float:
        """
        计算 CIE76 ΔE*ab 色差公式
        ΔE = sqrt((L1-L2)² + (a1-a2)² + (b1-b2)²)
        
        Args:
            lab1, lab2: Lab 颜色值 [L, a, b]
            
        Returns:
            ΔE 值
        """
        return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))

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
        检查颜色一致性（增强版：Lab + RGB 双重检查）
        
        Args:
            garment_original: 原始服装图
            original_mask: 原始服装 Mask
            generated: 生成图
            generated_mask: 生成图 Mask
            thresh: 可选的自定义 RGB 阈值（Lab 阈值固定为 COLOR_DE_THRESH）
            
        Returns:
            (passed, reason)
        """
        target_rgb_thresh = thresh if thresh is not None else cls.COLOR_RGB_THRESH
        
        # ① Lab 色彩空间 ΔE 检查（主要）
        orig_lab = cls._masked_mean_lab(garment_original, original_mask)
        gen_lab = cls._masked_mean_lab(generated, generated_mask)
        delta_e = cls._delta_e_cie76(orig_lab, gen_lab)
        
        print(
            f"[QC-Color] 原图 Lab=({orig_lab[0]:.1f},{orig_lab[1]:.1f},{orig_lab[2]:.1f}) "
            f"生成 Lab=({gen_lab[0]:.1f},{gen_lab[1]:.1f},{gen_lab[2]:.1f}) "
            f"ΔE={delta_e:.2f} 阈值<{cls.COLOR_DE_THRESH}"
        )
        
        if delta_e > cls.COLOR_DE_THRESH:
            return False, f"颜色偏差过大：ΔE={delta_e:.2f} > {cls.COLOR_DE_THRESH}（人眼可察觉）"
        
        # ② RGB 通道偏差检查（辅助）
        orig_rgb = cls._masked_mean_rgb(garment_original, original_mask)
        gen_rgb = cls._masked_mean_rgb(generated, generated_mask)
        deviation = np.abs(orig_rgb - gen_rgb) / 255.0
        max_dev = float(deviation.max())
        channels = ["R", "G", "B"]
        worst_ch = channels[int(deviation.argmax())]
        
        print(
            f"[QC-Color] 原图 RGB=({orig_rgb[0]:.1f},{orig_rgb[1]:.1f},{orig_rgb[2]:.1f}) "
            f"生成 RGB=({gen_rgb[0]:.1f},{gen_rgb[1]:.1f},{gen_rgb[2]:.1f}) "
            f"最大偏差={max_dev*100:.1f}%（{worst_ch}通道）阈值<{target_rgb_thresh*100:.0f}%"
        )
        
        if max_dev > target_rgb_thresh:
            return False, f"颜色偏差过大：{worst_ch}通道偏差 {max_dev*100:.1f}% > {target_rgb_thresh*100:.0f}%"
        
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

    @staticmethod
    def apply_color_correction(
        generated: Image.Image,
        garment_original: Image.Image,
        original_mask: Image.Image,
        generated_mask: Image.Image,
        correction_strength: float = 0.7,
    ) -> Image.Image:
        """
        对生成图像进行颜色校正，使其更接近原始服装颜色
        
        使用直方图匹配 + 颜色转移相结合的方法：
        1. 提取原图和生成图的服装区域
        2. 计算颜色统计信息（均值和标准差）
        3. 使用 Reinhard 颜色转移算法调整生成图颜色
        4. 将校正后的颜色与原生成图混合
        
        Args:
            generated: 生成图
            garment_original: 原始服装图
            original_mask: 原始服装 Mask
            generated_mask: 生成图 Mask
            correction_strength: 校正强度 (0.0-1.0)，0 表示不校正，1 表示完全匹配
            
        Returns:
            颜色校正后的图像
        """
        # 转换为 numpy 数组
        gen_np = np.array(generated.convert("RGB"), dtype=np.float32)
        orig_np = np.array(garment_original.convert("RGB"), dtype=np.float32)
        
        # 调整 mask 尺寸以匹配图像
        if generated_mask.size != generated.size:
            generated_mask = generated_mask.resize(generated.size, Image.NEAREST)
        if original_mask.size != garment_original.size:
            original_mask = original_mask.resize(garment_original.size, Image.NEAREST)
        
        gen_msk = np.array(generated_mask.convert("L"), dtype=np.float32) / 255.0
        orig_msk = np.array(original_mask.convert("L"), dtype=np.float32) / 255.0
        
        # 扩展 mask 维度
        gen_msk_3d = gen_msk[:, :, np.newaxis]
        orig_msk_3d = orig_msk[:, :, np.newaxis]
        
        # 计算服装区域的 RGB 均值和标准差
        def compute_stats(img, mask):
            masked = img * mask
            total = mask.sum()
            if total < 1:
                return img.mean(axis=(0, 1)), img.std(axis=(0, 1))
            mean = masked.sum(axis=(0, 1)) / total
            # 计算标准差
            var = ((img - mean) ** 2 * mask).sum(axis=(0, 1)) / total
            std = np.sqrt(var + 1e-6)  # 避免除零
            return mean, std
        
        orig_mean, orig_std = compute_stats(orig_np, orig_msk_3d)
        gen_mean, gen_std = compute_stats(gen_np, gen_msk_3d)
        
        # Reinhard 颜色转移
        # corrected = (gen - gen_mean) * (orig_std / gen_std) + orig_mean
        ratio = orig_std / (gen_std + 1e-6)
        corrected = (gen_np - gen_mean) * ratio + orig_mean
        
        # 限制值域
        corrected = np.clip(corrected, 0, 255)
        
        # 混合校正结果和原生成图
        result = gen_np * (1 - correction_strength) + corrected * correction_strength
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result)

