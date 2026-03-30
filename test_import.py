#!/usr/bin/env python3
"""测试 app.py 的导入是否成功"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("正在测试导入...")

try:
    from src.core.pipeline import ClothingDetailPipeline
    print("✅ 成功导入 ClothingDetailPipeline")
except Exception as e:
    print(f"❌ 导入 ClothingDetailPipeline 失败：{e}")
    sys.exit(1)

try:
    from src.config import settings
    print("✅ 成功导入 settings")
except Exception as e:
    print(f"❌ 导入 settings 失败：{e}")
    sys.exit(1)

try:
    from src.core.idmvton import get_pose_files
    print("✅ 成功导入 get_pose_files")
except Exception as e:
    print(f"❌ 导入 get_pose_files 失败：{e}")
    sys.exit(1)

try:
    from src.models.detectors import (
        detect_material, detect_gender, detect_category,
        detect_ethnicity, detect_age_group,
    )
    print("✅ 成功导入检测函数")
except Exception as e:
    print(f"❌ 导入检测函数失败：{e}")
    sys.exit(1)

# 现在尝试导入整个 app 模块
try:
    import app
    print("✅ 成功导入 app 模块")
except Exception as e:
    print(f"❌ 导入 app 模块失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 所有导入测试通过！")
