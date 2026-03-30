# 服装详情图智能生成系统 (Clothing Detail Picset)

本项目是一款针对电商场景设计的**商用级服装详情图生成工具**。基于 SAM、IDM-VTON、IC-Light 和 Real-ESRGAN 四大核心模型，实现从原始服装图到高质量试穿详情图的全自动生产。

---

## 🌟 核心特性

### 1. 商用级 Gradio UI 重构

- **沉浸式布局**：采用「左控右显」设计，对齐主流商用 AI 工具风格。
- **智能交互**：支持多图上传（最多 6 张）、组图需求智能描述、实时进度反馈。
- **生成预设**：预设白底精修、3D 立体、细节特写等多种电商常用图组。

### 2. 全自动化生产流水线

- **精准提取**：利用 SAM 自动识别并提取服装 Mask。
- **真实试穿**：集成 IDM-VTON 实现高保真度的人物姿态试穿。
- **光影增强**：通过 IC-Light 进行阴影后处理，显著提升成品的立体感与真实度。

### 3. 强制 4x 超分 (极致面料纹理)

- **原生超分**：强制启用 Real-ESRGAN 4 倍超分算法，输出分辨率高达 **3072x4096**。
- **纹理保护**：专门针对皮革、绸缎、编织等材质进行算法优化，保留极致的面料细节。

### 4. 智能质量控制 (QC) 与自愈重试

内置三项自动化检查逻辑，有效过滤生成瑕疵：

- **颜色一致性**：自动对比原图 RGB 偏差，偏差过大 (>15%) 自动更换 Seed 重刷。
- **结构完整性**：自动检测模特肢体畸变或边缘模糊并重试。
- **立体感自测**：检测阴影标准差，阴影不足时自动阶梯式调强 IC-Light 强度。

---

## 📂 目录结构

```text
clothing_detail_picset/
├── poses/           # 📥 放入 1-5 张人物姿态底图 (jpg/png)
├── outputs/         # 📤 生成的 4x 超分成品保存在此
├── ckpt/            # 🧠 模型权重存放目录
│   ├── sam_vit_h_4b8939.pth
│   ├── ic-light-fc.safetensors
│   └── realesr-general-x4v3.pth  # (自动下载)
├── assets/          # 🎨 UI 图标及静态资源
├── pipeline.py      # ⚙️ 核心逻辑：流水线编排 + QC 检查器
├── app.py           # 🖥️ 前端入口：Gradio 界面与事件绑定
└── requirements.txt # 📦 依赖列表
```

---

## 🛠️ 部署指南

### 1. 环境准备

推荐使用 Python 3.10+ 及 CUDA 11.8+ 环境。

```bash
# 1. 克隆并进入目录
cd clothing_detail_picset

# 2. 安装基础依赖
pip install -r requirements.txt

# 3. 安装专有组件
pip install git+https://github.com/facebookresearch/segment-anything.git
# 若需运行超分
pip install realesrgan
```

### 2. 模型权重下载

请将权重放置于 `ckpt/` 目录下（若不存在请手动创建）。

- **SAM (Segment Anything)**:
  [下载 sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- **IC-Light**:
  ```
  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download lllyasviel/ic-light iclight_sd15_fc.safetensors --local-dir ckpt/
  ```
- **Real-ESRGAN**:
  首次运行会自动下载相关模型权重。

### 3. 服务配置

若使用远程 IDM-VTON 服务，请设置环境变量：

```bash
export IDMVTON_URL="http://your-server-ip:7860/"
```

### 4. 启动应用

```bash
python app.py
```

访问地址: `http://localhost:7861` (内网) 或 `http://0.0.0.0:7861` (外网)

---

## 🧪 命令行调用 (开发者)

```bash
# 执行完整流程（含 4x 超分与质检）
python pipeline.py --garment test.jpg --desc "黑色真皮夹克" --category upper_body
```

---

## ⚡ 性能提示

- **显存占用**：建议使用 16GB+ 显存设备 (如 RTX 3090/4090)。
- **处理时间**：单张图完整耗时约 45s-90s (含超分与潜在的 QC 重试)。
