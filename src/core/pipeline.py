"""
服装详情图生成流水线
整合 SAM、IDM-VTON、IC-Light、Real-ESRGAN 的完整流程
"""
import os
import time
import random
from typing import Dict, List, Optional, Callable, Any
from PIL import Image

from src.config import settings
from src.core.sam import SAMExtractor
from src.core.idmvton import IDMVTONClient, get_pose_files
from src.core.iclight import ICLightProcessor
from src.core.superres import SuperResolutionProcessor
from src.models.detectors import (
    detect_material, detect_gender, detect_category,
    detect_ethnicity, detect_age_group, build_iclight_prompt,
)
from src.models.quality import QualityChecker


class ClothingDetailPipeline:
    """
    完整流水线编排：
    SAM → IDM-VTON × N → QC(颜色 + 结构，失败换 Seed 重试)
        → IC-Light × N → QC(立体感，失败调强 iclight_strength 重试)
        → Real-ESRGAN 4 倍超分
    """

    def __init__(
        self,
        idmvton_url: Optional[str] = None,
        poses_dir: Optional[str] = None,
        outputs_dir: Optional[str] = None,
        sam_ckpt: Optional[str] = None,
        iclight_ckpt: Optional[str] = None,
        sr_model: str = "realesr-general-x4v3",
        sr_tile: int = 512,
        device: Optional[str] = None,
    ):
        """
        Args:
            idmvton_url: IDM-VTON 服务地址
            poses_dir: 姿态图目录
            outputs_dir: 输出目录
            sam_ckpt: SAM 模型权重路径
            iclight_ckpt: IC-Light 模型权重路径
            sr_model: 超分模型名称
            sr_tile: 超分 tile 大小
            device: 运行设备
        """
        self.poses_dir = poses_dir or settings.poses_dir
        self.outputs_dir = outputs_dir or settings.outputs_dir
        os.makedirs(self.outputs_dir, exist_ok=True)

        self.sam = SAMExtractor(ckpt_path=sam_ckpt or settings.sam_ckpt, device=device)
        self.vton = IDMVTONClient(server_url=idmvton_url or settings.idmvton_url)
        self.iclight = ICLightProcessor(ckpt_path=iclight_ckpt or settings.iclight_ckpt, device=device)
        self.sr = SuperResolutionProcessor(
            model_name=sr_model, 
            ckpt_dir=settings.ckpt_dir,
            tile=sr_tile, 
            device=device,
        )
        self.qc = QualityChecker()

    def run(
        self,
        garment_image: Image.Image,
        garment_desc: str = "a garment",
        category: str = "upper_body",
        model_type: str = "adult_female",
        light_direction: str = "top_left",
        enable_sr: bool = True,
        denoise_steps: int = 30,
        seed: int = 42,
        enable_iclight: bool = True,
        iclight_strength: float = 0.35,
        specific_pose_paths: Optional[List] = None,
        material: Optional[str] = None,
        ethnicity: Optional[str] = None,
        age_group: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        执行完整流水线
        
        Args:
            garment_image: 服装图
            garment_desc: 服装描述
            category: 服装类别
            model_type: 模特类型
            light_direction: 光源方向
            enable_sr: 是否启用超分
            denoise_steps: 去噪步数
            seed: 随机种子
            enable_iclight: 是否启用 IC-Light
            iclight_strength: IC-Light 强度
            specific_pose_paths: 指定的姿态图路径列表
            material: 材质类型（可选，自动检测）
            ethnicity: 种族（可选，自动检测）
            age_group: 年龄段（可选，自动检测）
            progress_callback: 进度回调函数 stage, cur, total
            
        Returns:
            包含结果图和元数据的字典
        """
        def _progress(stage: str, cur: int, total: int):
            if progress_callback:
                progress_callback(stage, cur, total)

        qc_log = []

        # ── 材质识别 & 光影参数决策 ─────────────────────────────────────────────
        material = detect_material(material) if material else detect_material(garment_desc)
        ethnicity = detect_ethnicity(ethnicity) if ethnicity else detect_ethnicity(garment_desc)
        age_group = detect_age_group(age_group) if age_group else detect_age_group(garment_desc)

        specular_boost = 1.5 if material in ("leather", "satin") else 1.0
        if specular_boost > 1.0:
            print(f"[Light] 检测到高光材质 '{material}'，启用 specular_boost={specular_boost}")
        
        iclight_params = build_iclight_prompt(
            light_direction=light_direction,
            material=material,
            specular_boost=specular_boost,
        )
        print(f"[Light] 光源：{light_direction} | 材质：{material} | 属性：{ethnicity}, {age_group}")

        # ── Step 1: SAM ──────────────────────────────────────────────────────
        print("\n========== Step 1: SAM Mask 提取 ==========")
        _progress("SAM 提取 Mask", 0, 1)
        mask = self.sam.extract(garment_image)
        _progress("SAM 提取 Mask", 1, 1)

        # ── Step 2 + QC①②: IDM-VTON → 颜色一致性 & 结构完整性检查 ─────────────
        print("\n========== Step 2: IDM-VTON 批量换装 (增量重试模式) ==========")
        current_seed = seed

        # 获取姿态路径
        if specific_pose_paths:
            pose_files = specific_pose_paths
        else:
            pose_files = get_pose_files(self.poses_dir, model_type)
            num_to_sample = min(len(pose_files), settings.max_poses)
            if pose_files:
                pose_files = random.Random(seed).sample(pose_files, num_to_sample)

        # 初始化结果列表
        tryon_results = [None] * len(pose_files)
        vton_masks = [None] * len(pose_files)
        qc_passed = [False] * len(pose_files)

        for vton_attempt in range(settings.max_qc_retries + 1):
            todo_indices = [i for i, img in enumerate(tryon_results) if img is None]
            if not todo_indices:
                break

            if vton_attempt > 0:
                current_seed += 100
                msg = f"[QC] 增量重试 VTON 第{vton_attempt}次 (剩余{len(todo_indices)}张), seed={current_seed}"
                print(f"\n{msg}"); qc_log.append(msg)

            for i in todo_indices:
                pose_path = pose_files[i]
                if isinstance(pose_path, dict):
                    pose_path = pose_path.get("name") or pose_path.get("path")

                print(f"[IDM-VTON] 处理/修复 [{i+1}/{len(pose_files)}]: {os.path.basename(pose_path)}")
                _progress("IDM-VTON 换装", i + 1, len(pose_files))

                try:
                    res_img, res_mask = self.vton.tryon(
                        Image.open(pose_path).convert("RGB"),
                        garment_image,
                        garment_desc=garment_desc,
                        category=category,
                        denoise_steps=denoise_steps,
                        seed=current_seed + i,
                    )

                    # QC 检查
                    color_ok, color_reason = self.qc.check_color_consistency(
                        garment_original=garment_image, original_mask=mask,
                        generated=res_img, generated_mask=res_mask,
                    )
                    struct_ok, struct_reason = self.qc.check_structure_integrity(res_img)

                    if color_ok and struct_ok:
                        tryon_results[i] = res_img
                        vton_masks[i] = res_mask
                        qc_passed[i] = True
                        print(f"  ✅ [QC] 通过：{os.path.basename(pose_path)}")
                    else:
                        fail_msg = f"  ❌ [QC] 失败 ({color_reason or ''} {struct_reason or ''})"
                        print(fail_msg)
                        
                        # 如果颜色不一致但结构完整，尝试颜色校正
                        if not color_ok and struct_ok and vton_attempt == settings.max_qc_retries:
                            print(f"  🔧 [Color Correction] 尝试自动颜色校正...")
                            corrected_img = self.qc.apply_color_correction(
                                generated=res_img,
                                garment_original=garment_image,
                                original_mask=mask,
                                generated_mask=res_mask,
                                correction_strength=0.8,
                            )
                            
                            # 重新检查校正后的颜色
                            color_corrected_ok, _ = self.qc.check_color_consistency(
                                garment_original=garment_image, original_mask=mask,
                                generated=corrected_img, generated_mask=res_mask,
                            )
                            
                            if color_corrected_ok:
                                tryon_results[i] = corrected_img
                                vton_masks[i] = res_mask
                                qc_passed[i] = True
                                print(f"  ✅ [Color Correction] 颜色校正成功！")
                            else:
                                tryon_results[i] = res_img
                                vton_masks[i] = res_mask
                                qc_passed[i] = False
                                print(f"  ⚠️ [Color Correction] 颜色校正效果有限，保留原图")
                        
                        if vton_attempt == settings.max_qc_retries and not qc_passed[i]:
                            tryon_results[i] = res_img
                            vton_masks[i] = res_mask
                            qc_passed[i] = False
                            qc_log.append(f"[QC] ⚠️ 达到重试上限，保留当前结果")
                except Exception as e:
                    print(f"  ⚠️ [IDM-VTON] 接口调用失败：{e}")
                    if vton_attempt == settings.max_qc_retries:
                        qc_log.append(f"[QC] ❌ 最终换装失败")

        qc_log.append(f"[QC] VTON 阶段结束，最终合格率：{sum(1 for x in tryon_results if x is not None)}/{len(pose_files)}")

        # ── Step 3 + QC③: IC-Light → 立体感/阴影层次检查 ─────────────────────
        print(f"\n========== Step 3: IC-Light 阴影后处理 (权重基准={iclight_strength:.2f}) ==========")
        final_results = [None] * len(tryon_results)

        if enable_iclight:
            for i, (res, qc) in enumerate(zip(tryon_results, qc_passed)):
                if res is not None and not qc:
                    print(f"  ⚠️ [Skip IC-Light] 图像 {i+1} VTON QC 未通过，跳过打光处理")
                    final_results[i] = res

            for iclight_attempt in range(settings.max_qc_retries + 1):
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

                        composite_img = ic_res
                        if composite_img.size != res_vton.size:
                            composite_img = composite_img.resize(res_vton.size, Image.LANCZOS)

                        # QC③检查
                        shadow_ok, shadow_reason = self.qc.check_shadow_depth(composite_img)
                        color_final_ok, color_final_reason = self.qc.check_color_consistency(
                            garment_original=garment_image, original_mask=mask,
                            generated=composite_img, generated_mask=msk_vton,
                            thresh=0.15
                        )

                        if shadow_ok and color_final_ok:
                            final_results[i] = composite_img
                            qc_passed[i] = True
                            print(f"  ✅ [QC] 打光通过")
                        else:
                            print(f"  ❌ [QC] 打光不足或偏色：{shadow_reason or ''} {color_final_reason or ''}")
                            if iclight_attempt == settings.max_qc_retries:
                                final_results[i] = composite_img
                                qc_passed[i] = False
                                qc_log.append(f"[QC] ⚠️ 打光重试上限，保留当前结果")
                    except Exception as e:
                        print(f"  ⚠️ [IC-Light] 处理失败：{e}")
                        if iclight_attempt == settings.max_qc_retries:
                            final_results[i] = res_vton
        else:
            final_results = tryon_results

        qc_log.append(f"[QC] IC-Light 阶段结束")

        # ── Step 4: Real-ESRGAN 4 倍超分 ────────────────────────────────────────
        if enable_sr:
            print("\n========== Step 4: Real-ESRGAN 4 倍超分 ==========")
            valid_for_sr = [r for i, r in enumerate(final_results) if r is not None and qc_passed[i]]

            if valid_for_sr:
                sr_valid = self.sr.batch_upscale(
                    valid_for_sr,
                    progress_callback=lambda cur, total: _progress("超分 4x", cur, total),
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
        sr_paths = []

        for i, (img, sr_img) in enumerate(zip(final_results, sr_results)):
            p = None
            if img:
                p = os.path.join(self.outputs_dir, f"{ts}_result_{i+1}.jpg")
                img.save(p, format='JPEG', quality=95)
                final_paths.append(p)
            else:
                final_paths.append(None)

            if sr_img:
                p_sr = os.path.join(self.outputs_dir, f"{ts}_result_{i+1}_4x.jpg")
                sr_img.save(p_sr, format='JPEG', quality=95)
                sr_paths.append(p_sr)
            else:
                sr_paths.append(p)

        n_saved = sum(1 for r in sr_results if r) if enable_sr else sum(1 for r in final_results if r)
        print(f"\n✅ 流水线完成，共输出 {n_saved} 张{'4x 超分' if enable_sr else ''}详情图，已保存至 {self.outputs_dir}")
        if qc_log:
            print("[QC 日志]\n" + "\n".join(f"  {l}" for l in qc_log))

        return {
            "mask": mask,
            "mask_path": mask_path,
            "tryon_results": tryon_results,
            "final_results": final_results,
            "final_paths": final_paths,
            "sr_results": sr_results,
            "sr_paths": sr_paths,
            "material": material,
            "specular_boost": specular_boost,
            "light_direction": light_direction,
            "qc_log": qc_log,
        }
