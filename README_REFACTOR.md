# 代码重构说明

## 重构概述

本次重构将原有的单体 `pipeline.py` (1469 行) 和 `app.py` (503 行) 拆分为模块化、可维护的包结构。

## 新目录结构

```
/workspace/
├── app.py                      # Gradio 前端界面（已更新导入路径）
├── requirements.txt            # 依赖列表
├── README.md                   # 项目说明
└── src/                        # 核心代码包
    ├── __init__.py             # 包初始化与版本信息
    ├── config/                 # 配置模块
    │   ├── __init__.py
    │   └── settings.py         # 全局配置类 (dataclass)
    ├── core/                   # 核心流水线组件
    │   ├── __init__.py
    │   ├── pipeline.py         # 主流水线编排
    │   ├── sam.py              # SAM 掩膜提取器
    │   ├── idmvton.py          # IDM-VTON 换装客户端
    │   ├── iclight.py          # IC-Light 光影处理器
    │   └── superres.py         # Real-ESRGAN 超分处理器
    ├── models/                 # 数据模型与工具
    │   ├── __init__.py
    │   ├── detectors.py        # 材质/类别/性别检测器
    │   └── quality.py          # 质量控制检查器
    └── utils/                  # 通用工具
        ├── __init__.py
        └── image_ops.py        # 图像操作工具函数
```

## 主要改进

### 1. 模块化设计
- **单一职责原则**: 每个模块只负责一个特定功能
- **高内聚低耦合**: 模块间通过清晰的接口通信
- **易于测试**: 独立模块便于单元测试

### 2. 配置管理
- 使用 `dataclass` 提供类型安全的配置
- 集中管理所有配置项
- 支持环境变量覆盖

### 3. 类型注解
- 全面的类型提示
- 更好的 IDE 支持
- 减少运行时错误

### 4. 代码质量
- 统一的文档字符串规范
- 清晰的函数参数说明
- 合理的异常处理

## 模块说明

### src.config.settings
```python
from src.config import settings

# 访问配置
print(settings.idmvton_url)
print(settings.poses_dir)
print(settings.max_qc_retries)
```

### src.models.detectors
```python
from src.models.detectors import (
    detect_material, detect_category, 
    detect_gender, detect_ethnicity, detect_age_group,
    build_iclight_prompt
)

material = detect_material("丝绸连衣裙")  # 返回 "satin"
category = detect_category("牛仔裤")      # 返回 "lower_body"
```

### src.models.quality
```python
from src.models.quality import QualityChecker
from PIL import Image

img = Image.open("test.jpg")
passed, reason = QualityChecker.check_structure_integrity(img)
```

### src.core.pipeline
```python
from src.core import ClothingDetailPipeline
from PIL import Image

pipeline = ClothingDetailPipeline()
garment = Image.open("garment.jpg")

results = pipeline.run(
    garment_image=garment,
    garment_desc="法式复古真丝连衣裙",
    model_type="adult_female",
)
```

## 向后兼容性

原 `pipeline.py` 中的主要功能和接口保持不变：
- `ClothingDetailPipeline` 类的 `run()` 方法签名保持一致
- 检测函数 (`detect_*`) 行为不变
- 质量控制逻辑保持原样

## 迁移指南

### 原导入路径 → 新导入路径

| 原路径 | 新路径 |
|--------|--------|
| `from pipeline import ClothingDetailPipeline` | `from src.core import ClothingDetailPipeline` |
| `from pipeline import detect_material` | `from src.models.detectors import detect_material` |
| `from pipeline import get_pose_files` | `from src.core.idmvton import get_pose_files` |
| `POSES_DIR` | `settings.poses_dir` |
| `OUTPUTS_DIR` | `settings.outputs_dir` |
| `IDMVTON_URL` | `settings.idmvton_url` |

## 运行测试

```bash
# 测试配置模块
python -c "from src.config import settings; print(settings)"

# 测试检测器
python -c "from src.models.detectors import detect_material; print(detect_material('丝绸'))"

# 测试质量控制
python -c "from src.models.quality import QualityChecker; from PIL import Image; img = Image.new('RGB', (100, 100)); print(QualityChecker.check_structure_integrity(img))"

# 测试流水线导入
python -c "from src.core.pipeline import ClothingDetailPipeline; print('OK')"
```

## 下一步

1. 添加单元测试
2. 完善日志系统
3. 添加性能监控
4. 支持更多检测模型
