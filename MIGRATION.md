# 目录结构迁移说明

## 迁移概述

为了简化部署和维护，已将数据目录迁移到项目根目录。

## 变更内容

### 1. 目录结构调整

**调整前：**
```
clothing_detail_picset/
├── src/
│   ├── poses/          ← 姿态图片
│   └── ckpt/           ← 模型权重（如果存在）
└── outputs/            ← 输出目录
```

**调整后：**
```
clothing_detail_picset/
├── poses/              ← 姿态图片（已迁移到根目录）
├── ckpt/               ← 模型权重（已迁移到根目录）
├── outputs/            ← 输出目录
└── src/                ← 源代码（不再包含数据目录）
```

### 2. 配置文件更新

**文件：** `src/config/settings.py`

**变更：**
- `base_dir` 计算路径从 2 级父目录改为 3 级父目录
- 确保所有子目录路径都基于项目根目录

```python
# 修改前
base_dir = os.path.dirname(os.path.dirname(__file__))  # 指向 src/

# 修改后
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # 指向项目根目录
```

### 3. 模块路径更新

**文件：** `src/core/superres.py`

**变更：**
- `SuperResolutionProcessor.__init__` 优先使用配置文件中的 `ckpt_dir`
- 保持向后兼容，支持手动指定路径

```python
# 新逻辑
if ckpt_dir:
    self.ckpt_dir = ckpt_dir
else:
    from src.config import settings
    self.ckpt_dir = settings.ckpt_dir  # 使用项目根目录的 ckpt/
```

## 迁移步骤

### 已完成的操作

1. ✅ 更新 `src/config/settings.py` 中的 `base_dir` 计算
2. ✅ 更新 `src/core/superres.py` 中的默认路径逻辑
3. ✅ 验证所有路径引用都通过配置文件

### 需要手动执行的操作

#### 1. 确认目录存在

确保项目根目录下有以下目录：
```bash
poses/
ckpt/
outputs/
```

#### 2. 删除旧目录（可选）

如果确认根目录已有这些目录，可以删除 `src/poses`：
```bash
rm -rf src/poses
```

#### 3. 测试路径配置

运行测试脚本验证路径：
```bash
python test_paths.py
```

预期输出：
```
============================================================
路径配置测试
============================================================

1. base_dir: /path/to/clothing_detail_picset
   期望：/path/to/clothing_detail_picset
   ✓ 正确

2. poses_dir: /path/to/clothing_detail_picset/poses
   期望：/path/to/clothing_detail_picset/poses
   ✓ 正确

... (所有检查都应该显示 ✓ 正确)

============================================================
目录存在性检查
============================================================
poses     : ✓ 存在 - /path/to/clothing_detail_picset/poses
ckpt      : ✓ 存在 - /path/to/clothing_detail_picset/ckpt
outputs   : ✓ 存在 - /path/to/clothing_detail_picset/outputs

============================================================
测试完成！
============================================================
```

## 优势

1. **部署简化**：数据目录与代码分离，便于版本控制
2. **维护方便**：更新代码时不会覆盖用户数据
3. **清晰结构**：项目结构更符合 Python 包规范
4. **向后兼容**：保留了旧版本的兼容性处理

## 注意事项

1. 首次迁移后，请确保 `ckpt/` 目录中包含所需的模型权重文件
2. 如果使用 Docker 或其他容器化部署，需要更新卷挂载路径
3. CI/CD 流程中可能需要调整构建步骤

## 回滚方案

如果需要回滚到旧版本：

1. 恢复 `src/config/settings.py`：
   ```python
   base_dir = os.path.dirname(os.path.dirname(__file__))
   ```

2. 恢复 `src/core/superres.py`：
   ```python
   self.ckpt_dir = ckpt_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "ckpt")
   ```

3. 将 `poses/` 和 `ckpt/` 移回 `src/` 目录
