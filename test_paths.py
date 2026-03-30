#!/usr/bin/env python3
"""测试路径配置是否正确"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings

print("=" * 60)
print("路径配置测试")
print("=" * 60)

print(f"\n1. base_dir: {settings.base_dir}")
print(f"   期望：/path/to/clothing_detail_picset")
print(f"   ✓ 正确" if os.path.basename(settings.base_dir) == "clothing_detail_picset" else "   ✗ 错误")

print(f"\n2. poses_dir: {settings.poses_dir}")
print(f"   期望：{settings.base_dir}/poses")
print(f"   ✓ 正确" if settings.poses_dir == os.path.join(settings.base_dir, "poses") else "   ✗ 错误")

print(f"\n3. ckpt_dir: {settings.ckpt_dir}")
print(f"   期望：{settings.base_dir}/ckpt")
print(f"   ✓ 正确" if settings.ckpt_dir == os.path.join(settings.base_dir, "ckpt") else "   ✗ 错误")

print(f"\n4. outputs_dir: {settings.outputs_dir}")
print(f"   期望：{settings.base_dir}/outputs")
print(f"   ✓ 正确" if settings.outputs_dir == os.path.join(settings.base_dir, "outputs") else "   ✗ 错误")

print(f"\n5. sam_ckpt: {settings.sam_ckpt}")
print(f"   期望：{settings.base_dir}/ckpt/sam_vit_h_4b8939.pth")
expected_sam = os.path.join(settings.base_dir, "ckpt", "sam_vit_h_4b8939.pth")
print(f"   ✓ 正确" if settings.sam_ckpt == expected_sam else "   ✗ 错误")

print(f"\n6. iclight_ckpt: {settings.iclight_ckpt}")
print(f"   期望：{settings.base_dir}/ckpt/iclight_sd15_fc.safetensors")
expected_iclight = os.path.join(settings.base_dir, "ckpt", "iclight_sd15_fc.safetensors")
print(f"   ✓ 正确" if settings.iclight_ckpt == expected_iclight else "   ✗ 错误")

# 检查目录是否存在
print("\n" + "=" * 60)
print("目录存在性检查")
print("=" * 60)

for dir_name, dir_path in [
    ("poses", settings.poses_dir),
    ("ckpt", settings.ckpt_dir),
    ("outputs", settings.outputs_dir),
]:
    exists = os.path.exists(dir_path)
    print(f"{dir_name:10s}: {'✓ 存在' if exists else '✗ 不存在'} - {dir_path}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
