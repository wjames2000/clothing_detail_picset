#!/usr/bin/env python3
"""修复 app.py 中的变量引用问题"""

import re

# 读取文件
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换 get_pipeline 函数中的变量引用
old_code = """            _pipeline = ClothingDetailPipeline(
                idmvton_url=IDMVTON_URL,
                poses_dir=POSES_DIR,
                outputs_dir=OUTPUTS_DIR,
            )"""

new_code = """            _pipeline = ClothingDetailPipeline(
                idmvton_url=settings.idmvton_url,
                poses_dir=settings.poses_dir,
                outputs_dir=settings.outputs_dir,
            )"""

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ 成功替换 get_pipeline 函数中的变量引用")
else:
    print("⚠️  未找到需要替换的旧代码片段")

# 写回文件
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("📝 文件修复完成")
