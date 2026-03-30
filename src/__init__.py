"""
数字工坊 / THE DIGITAL ATELIER
服装详情图生成系统 - 重构版本

项目结构:
    src/
    ├── __init__.py          # 包初始化
    ├── config/              # 配置模块
    │   ├── __init__.py
    │   └── settings.py      # 全局配置
    ├── core/                # 核心流水线
    │   ├── __init__.py
    │   ├── pipeline.py      # 主流水线编排
    │   ├── sam.py           # SAM 掩膜提取
    │   ├── idmvton.py       # IDM-VTON 换装
    │   ├── iclight.py       # IC-Light 光影处理
    │   └── superres.py      # 超分处理
    ├── models/              # 数据模型与工具
    │   ├── __init__.py
    │   ├── detectors.py     # 材质/类别/性别检测
    │   └── quality.py       # 质量控制
    └── utils/               # 通用工具
        ├── __init__.py
        └── image_ops.py     # 图像操作工具

app.py                     # Gradio 前端界面
requirements.txt           # 依赖
"""

__version__ = "2.0.0"
__author__ = "THE DIGITAL ATELIER"
