"""
质量控制检查器（增强版）
多项图像质量检查：
1. 颜色一致性：均值 RGB + 直方图相关性 + KL 散度 + Lab ΔE
2. 材质特定阈值：denim(22%), cotton(25%), silk(30%), leather(35%) 等
3. 纹理保留度：SSIM + 梯度方向一致性
4. 特征相似度：ORB 特征点匹配 (logo、图案等)
5. 结构完整性：Laplacian 方差 + 边缘密度
6. 阴影层次：灰度标准差
"""
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Optional, Dict, Any
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy


class QualityChecker:
    """
    多项图像质量检查（增强版），每项返回 (passed: bool, reason: str)。

    ① check_color_consistency  — 颜色一致性（多指标融合）
       - 均值 RGB 偏差检查（支持材质特定阈值）
       - 直方图相关性检查（Bhattacharyya 距离）
       - KL 散度检查（颜色分布相似性）
       - Lab 色彩空间 ΔE 检查（CIE76 公式）
       
    ② check_texture_preservation — 纹理保留度（新增）
       - SSIM 结构相似性检查
       - 梯度方向一致性检查
       
    ③ check_feature_similarity   — 特征点匹配（新增）
       - ORB 特征点检测和匹配
       - 用于 logo、图案等细节保真度检查
       
    ④ check_structure_integrity  — 结构完整性
       Laplacian 方差（模糊度）+ 边缘密度检查

    ⑤ check_shadow_depth        — 立体感/阴影层次
       灰度标准差检查
       
    ⑥ apply_color_correction     — 颜色校正
       Reinhard 颜色转移算法
    """

    # ========== 基础阈值配置 ==========
    # Lab 色彩空间 ΔE 阈值 (CIE76 公式)
    COLOR_DE_THRESH = 15.0
    
    # 直方图相关性阈值（Bhattacharyya 距离，越小越好）
    HIST_BHATTACHARYYA_THRESH = 0.15
    
    # KL 散度阈值（颜色分布差异）
    KL_DIVERGENCE_THRESH = 0.5
    
    # 纹理 SSIM 阈值
    TEXTURE_SSIM_THRESH = 0.75
    
    # 梯度方向一致性阈值
    GRADIENT_ORIENTATION_THRESH = 0.70
    
    # ORB 特征点匹配率阈值
    FEATURE_MATCH_RATIO_THRESH = 0.30
    
    BLUR_THRESH = 50.0    # Laplacian 方差阈值
    EDGE_THRESH = 0.015   # 边缘像素比例阈值 (1.5%)
    SHADOW_THRESH = 20.0  # 灰度标准差阈值

    # ========== 材质特定阈值配置 ==========
    # RGB 通道偏差阈值（按材质分类，单位：%）
    MATERIAL_RGB_THRESHOLDS: Dict[str, float] = {
        "denim": 0.22,      # 牛仔布容差 22%
        "cotton": 0.25,     # 棉质容差 25%
        "silk": 0.30,       # 丝绸容差 30%
        "satin": 0.30,      # 缎面容差 30%
        "leather": 0.35,    # 皮革容差 35%
        "wool": 0.25,       # 羊毛容差 25%
        "linen": 0.25,      # 亚麻容差 25%
        "chiffon": 0.28,    # 雪纺容差 28%
        "velvet": 0.30,     # 丝绒容差 30%
        "lace": 0.28,       # 蕾丝容差 28%
        "general": 0.20,    # 通用容差 20%
    }
    
    # Lab ΔE 材质特定阈值
    MATERIAL_DE_THRESHOLDS: Dict[str, float] = {
        "denim": 18.0,      # 牛仔布 ΔE 阈值
        "cotton": 16.0,     # 棉质 ΔE 阈值
        "silk": 20.0,       # 丝绸 ΔE 阈值（高反光材质允许更大偏差）
        "satin": 20.0,      # 缎面 ΔE 阈值
        "leather": 22.0,    # 皮革 ΔE 阈值
        "wool": 17.0,       # 羊毛 ΔE 阈值
        "linen": 17.0,      # 亚麻 ΔE 阈值
        "chiffon": 18.0,    # 雪纺 ΔE 阈值
        "velvet": 19.0,     # 丝绒 ΔE 阈值
        "lace": 18.0,       # 蕾丝 ΔE 阈值
        "general": 15.0,    # 通用 ΔE 阈值
    }

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

    @staticmethod
    def _compute_histogram(image: Image.Image, mask: Image.Image, bins: int = 32) -> np.ndarray:
        """
        计算 Mask 区域的 RGB 直方图
        
        Args:
            image: 输入图像
            mask: Mask 图像
            bins: 直方图 bin 数量
            
        Returns:
            归一化的 RGB 直方图 (3, bins)
        """
        img_np = np.array(image.convert("RGB"), dtype=np.float32)
        if img_np.shape[:2] != (mask.height, mask.width):
            mask = mask.resize((img_np.shape[1], img_np.shape[0]), Image.NEAREST)
        msk_np = np.array(mask.convert("L"), dtype=np.float32) / 255.0
        
        histograms = []
        for c in range(3):
            channel_data = img_np[:, :, c][msk_np > 0.5]
            if len(channel_data) < 10:
                hist = np.ones(bins) / bins
            else:
                hist, _ = np.histogram(channel_data, bins=bins, range=(0, 256), density=True)
                hist = hist + 1e-10  # 避免零值
                hist = hist / hist.sum()  # 归一化
            histograms.append(hist)
        
        return np.array(histograms)

    @staticmethod
    def _bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        计算两个直方图的 Bhattacharyya 距离
        
        Args:
            hist1, hist2: 归一化直方图 (3, bins)
            
        Returns:
            Bhattacharyya 距离 (0-1，越小越相似)
        """
        bc_coeff = 0.0
        for c in range(3):
            bc_coeff += np.sqrt(hist1[c] * hist2[c]).sum()
        bc_coeff /= 3.0
        return -np.log(bc_coeff + 1e-10)

    @staticmethod
    def _kl_divergence(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        计算两个直方图的 KL 散度（对称化）
        
        Args:
            hist1, hist2: 归一化直方图 (3, bins)
            
        Returns:
            对称 KL 散度
        """
        kl_12 = 0.0
        kl_21 = 0.0
        for c in range(3):
            kl_12 += entropy(hist1[c], hist2[c])
            kl_21 += entropy(hist2[c], hist1[c])
        return (kl_12 + kl_21) / 2.0

    @classmethod
    def check_color_consistency(
        cls,
        garment_original: Image.Image,
        original_mask: Image.Image,
        generated: Image.Image,
        generated_mask: Image.Image,
        material: str = "general",
        thresh: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        检查颜色一致性（多指标融合：均值 RGB + 直方图相关性 + KL 散度 + Lab ΔE）
        
        Args:
            garment_original: 原始服装图
            original_mask: 原始服装 Mask
            generated: 生成图
            generated_mask: 生成图 Mask
            material: 材质类型（用于选择特定阈值）
            thresh: 可选的自定义 RGB 阈值（优先级高于材质阈值）
            
        Returns:
            (passed, reason)
        """
        # 获取材质特定阈值
        de_thresh = cls.MATERIAL_DE_THRESHOLDS.get(material, cls.MATERIAL_DE_THRESHOLDS["general"])
        rgb_thresh = thresh if thresh is not None else cls.MATERIAL_RGB_THRESHOLDS.get(material, cls.MATERIAL_RGB_THRESHOLDS["general"])
        
        results = {}
        failures = []
        
        # ========== ① Lab 色彩空间 ΔE 检查 ==========
        orig_lab = cls._masked_mean_lab(garment_original, original_mask)
        gen_lab = cls._masked_mean_lab(generated, generated_mask)
        delta_e = cls._delta_e_cie76(orig_lab, gen_lab)
        results["delta_e"] = delta_e
        
        print(
            f"[QC-Color] Lab ΔE 检查：原图 Lab=({orig_lab[0]:.1f},{orig_lab[1]:.1f},{orig_lab[2]:.1f}) "
            f"生成 Lab=({gen_lab[0]:.1f},{gen_lab[1]:.1f},{gen_lab[2]:.1f}) "
            f"ΔE={delta_e:.2f} 阈值<{de_thresh} (材质:{material})"
        )
        
        if delta_e > de_thresh:
            failures.append(f"Lab ΔE={delta_e:.2f} > {de_thresh}")
        
        # ========== ② RGB 通道偏差检查 ==========
        orig_rgb = cls._masked_mean_rgb(garment_original, original_mask)
        gen_rgb = cls._masked_mean_rgb(generated, generated_mask)
        deviation = np.abs(orig_rgb - gen_rgb) / 255.0
        max_dev = float(deviation.max())
        channels = ["R", "G", "B"]
        worst_ch = channels[int(deviation.argmax())]
        results["rgb_deviation"] = max_dev
        
        print(
            f"[QC-Color] RGB 均值检查：原图 RGB=({orig_rgb[0]:.1f},{orig_rgb[1]:.1f},{orig_rgb[2]:.1f}) "
            f"生成 RGB=({gen_rgb[0]:.1f},{gen_rgb[1]:.1f},{gen_rgb[2]:.1f}) "
            f"最大偏差={max_dev*100:.1f}%（{worst_ch}通道）阈值<{rgb_thresh*100:.0f}%"
        )
        
        if max_dev > rgb_thresh:
            failures.append(f"RGB {worst_ch}通道偏差={max_dev*100:.1f}% > {rgb_thresh*100:.0f}%")
        
        # ========== ③ 直方图 Bhattacharyya 距离检查 ==========
        orig_hist = cls._compute_histogram(garment_original, original_mask)
        gen_hist = cls._compute_histogram(generated, generated_mask)
        bhattacharyya_dist = cls._bhattacharyya_distance(orig_hist, gen_hist)
        results["bhattacharyya"] = bhattacharyya_dist
        
        print(
            f"[QC-Color] 直方图 Bhattacharyya 距离={bhattacharyya_dist:.4f} 阈值<{cls.HIST_BHATTACHARYYA_THRESH}"
        )
        
        if bhattacharyya_dist > cls.HIST_BHATTACHARYYA_THRESH:
            failures.append(f"直方图 Bhattacharyya 距离={bhattacharyya_dist:.4f} > {cls.HIST_BHATTACHARYYA_THRESH}")
        
        # ========== ④ KL 散度检查 ==========
        kl_div = cls._kl_divergence(orig_hist, gen_hist)
        results["kl_divergence"] = kl_div
        
        print(
            f"[QC-Color] KL 散度={kl_div:.4f} 阈值<{cls.KL_DIVERGENCE_THRESH}"
        )
        
        if kl_div > cls.KL_DIVERGENCE_THRESH:
            failures.append(f"KL 散度={kl_div:.4f} > {cls.KL_DIVERGENCE_THRESH}")
        
        # ========== 综合判断 ==========
        if failures:
            return False, f"颜色不一致：{' | '.join(failures)}"
        
        return True, ""

    @classmethod
    def check_texture_preservation(
        cls,
        garment_original: Image.Image,
        original_mask: Image.Image,
        generated: Image.Image,
        generated_mask: Image.Image,
    ) -> Tuple[bool, str]:
        """
        检查纹理保留度（SSIM + 梯度方向一致性）
        
        Args:
            garment_original: 原始服装图
            original_mask: 原始服装 Mask
            generated: 生成图
            generated_mask: 生成图 Mask
            
        Returns:
            (passed, reason)
        """
        failures = []
        
        # ========== ① SSIM 结构相似性检查 ==========
        img_orig = np.array(garment_original.convert("L"), dtype=np.float32) / 255.0
        img_gen = np.array(generated.convert("L"), dtype=np.float32) / 255.0
        
        # 调整尺寸以匹配
        if img_orig.shape != img_gen.shape:
            from PIL import Image
            gen_resized = generated.resize((garment_original.width, garment_original.height), Image.LANCZOS)
            img_gen = np.array(gen_resized.convert("L"), dtype=np.float32) / 255.0
        
        # 调整 mask 尺寸
        if original_mask.size != (img_orig.shape[1], img_orig.shape[0]):
            orig_mask_resized = original_mask.resize((img_orig.shape[1], img_orig.shape[0]), Image.NEAREST)
        else:
            orig_mask_resized = original_mask
        
        if generated_mask.size != (img_gen.shape[1], img_gen.shape[0]):
            gen_mask_resized = generated_mask.resize((img_gen.shape[1], img_gen.shape[0]), Image.NEAREST)
        else:
            gen_mask_resized = generated_mask
        
        msk_orig = np.array(orig_mask_resized.convert("L"), dtype=np.float32) / 255.0 > 0.5
        msk_gen = np.array(gen_mask_resized.convert("L"), dtype=np.float32) / 255.0 > 0.5
        
        # 计算共同区域
        common_mask = msk_orig & msk_gen
        if common_mask.sum() < 100:
            print(f"[QC-Texture] SSIM 检查：Mask 重叠区域过小，跳过")
        else:
            # 计算 SSIM（使用滑动窗口）- 指定 data_range
            ssim_score, _ = ssim(img_orig, img_gen, full=True, data_range=1.0)
            
            print(f"[QC-Texture] SSIM 结构相似性={ssim_score:.4f} 阈值>{cls.TEXTURE_SSIM_THRESH}")
            
            if ssim_score < cls.TEXTURE_SSIM_THRESH:
                failures.append(f"SSIM={ssim_score:.4f} < {cls.TEXTURE_SSIM_THRESH}")
        
        # ========== ② 梯度方向一致性检查 ==========
        # 计算原图梯度
        gy_orig, gx_orig = np.gradient(img_orig)
        angle_orig = np.arctan2(gy_orig, gx_orig)
        
        # 计算生成图梯度
        gy_gen, gx_gen = np.gradient(img_gen)
        angle_gen = np.arctan2(gy_gen, gx_gen)
        
        # 在 mask 区域内比较梯度方向
        common_pixels = common_mask.sum()
        if common_pixels > 100:
            # 计算角度差异（考虑周期性）
            angle_diff = np.abs(angle_orig - angle_gen)
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            
            # 计算一致的比例（角度差小于 45 度视为一致）
            consistent_ratio = (angle_diff[common_mask] < np.pi / 4).mean()
            
            print(f"[QC-Texture] 梯度方向一致性={consistent_ratio:.4f} 阈值>{cls.GRADIENT_ORIENTATION_THRESH}")
            
            if consistent_ratio < cls.GRADIENT_ORIENTATION_THRESH:
                failures.append(f"梯度方向一致性={consistent_ratio:.4f} < {cls.GRADIENT_ORIENTATION_THRESH}")
        else:
            print(f"[QC-Texture] 梯度方向检查：Mask 重叠区域过小，跳过")
        
        if failures:
            return False, f"纹理保留不足：{' | '.join(failures)}"
        
        return True, ""

    @classmethod
    def check_feature_similarity(
        cls,
        garment_original: Image.Image,
        original_mask: Image.Image,
        generated: Image.Image,
        generated_mask: Image.Image,
        min_match_ratio: float = 0.30,
    ) -> Tuple[bool, str]:
        """
        检查特征点相似度（ORB 特征匹配，用于 logo、图案等细节保真度）
        
        Args:
            garment_original: 原始服装图
            original_mask: 原始服装 Mask
            generated: 生成图
            generated_mask: 生成图 Mask
            min_match_ratio: 最小匹配率阈值
            
        Returns:
            (passed, reason)
        """
        # 转换为 OpenCV 格式
        img_orig = cv2.cvtColor(np.array(garment_original.convert("RGB")), cv2.COLOR_RGB2BGR)
        img_gen = cv2.cvtColor(np.array(generated.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # 调整尺寸以匹配
        if img_orig.shape[:2] != img_gen.shape[:2]:
            img_gen = cv2.resize(img_gen, (img_orig.shape[1], img_orig.shape[0]))
        
        # 调整 mask
        if original_mask.size != (img_orig.shape[1], img_orig.shape[0]):
            orig_mask_cv = cv2.resize(np.array(original_mask.convert("L")), (img_orig.shape[1], img_orig.shape[0]))
        else:
            orig_mask_cv = np.array(original_mask.convert("L"))
        
        if generated_mask.size != (img_gen.shape[1], img_gen.shape[0]):
            gen_mask_cv = cv2.resize(np.array(generated_mask.convert("L")), (img_gen.shape[1], img_gen.shape[0]))
        else:
            gen_mask_cv = np.array(generated_mask.convert("L"))
        
        # 二值化 mask
        _, orig_mask_bin = cv2.threshold(orig_mask_cv, 128, 255, cv2.THRESH_BINARY)
        _, gen_mask_bin = cv2.threshold(gen_mask_cv, 128, 255, cv2.THRESH_BINARY)
        
        # 创建 ORB 检测器
        orb = cv2.ORB_create(nfeatures=500)
        
        # 检测关键点和描述子
        kp1, des1 = orb.detectAndCompute(img_orig, orig_mask_bin)
        kp2, des2 = orb.detectAndCompute(img_gen, gen_mask_bin)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            print(f"[QC-Feature] ORB 特征点过少 (orig={len(kp1)}, gen={len(kp2)})，跳过检查")
            return True, ""
        
        # BFMatcher 进行特征匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # 计算匹配率
        match_ratio = len(matches) / max(len(kp1), len(kp2))
        
        print(
            f"[QC-Feature] ORB 特征点：原图={len(kp1)}, 生成图={len(kp2)}, "
            f"匹配数={len(matches)}, 匹配率={match_ratio:.4f} 阈值>{min_match_ratio}"
        )
        
        if match_ratio < min_match_ratio:
            return False, f"特征点匹配率={match_ratio:.4f} < {min_match_ratio}（logo/图案可能丢失）"
        
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

