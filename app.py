"""
服装详情图生成 —— Gradio 前端界面（THE DIGITAL ATELIER 重新设计版）
"""

import os
import sys
import threading
import glob
import random
import gradio as gr
from PIL import Image

from pipeline import (
    ClothingDetailPipeline, POSES_DIR, OUTPUTS_DIR, IDMVTON_URL,
    MODEL_TYPE_DIRS, get_pose_files,
    detect_material, detect_gender, detect_category, 
    detect_ethnicity, detect_age_group, # [NEW] 导入新识别函数
    DEFAULT_LIGHT_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
# 全局单例流水线
# ─────────────────────────────────────────────────────────────────────────────

_pipeline: ClothingDetailPipeline = None
_pipeline_lock = threading.Lock()

def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = ClothingDetailPipeline(
                idmvton_url=IDMVTON_URL,
                poses_dir=POSES_DIR,
                outputs_dir=OUTPUTS_DIR,
            )
    return _pipeline

# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def get_image_from_gradio(img_input):
    """从 Gradio 各种可能的输入格式中提取 PIL Image"""
    if img_input is None: return None
    
    import numpy as np
    first_img = img_input
    # 如果是 Gallery 选中的图
    if isinstance(first_img, (list, tuple)) and len(first_img) > 0:
        first_img = first_img[0]
        
    while isinstance(first_img, (dict, list, tuple)) and not isinstance(first_img, np.ndarray):
        if isinstance(first_img, dict):
            val = first_img.get("image") or first_img.get("path") or first_img.get("name")
            if val: first_img = val
            else: break
        elif isinstance(first_img, (list, tuple)):
            first_img = first_img[0]
        else: break

    if isinstance(first_img, str):
        return Image.open(first_img).convert("RGB")
    elif isinstance(first_img, Image.Image):
        return first_img.convert("RGB")
    elif isinstance(first_img, np.ndarray):
        return Image.fromarray(first_img).convert("RGB")
    return None

def generate_v3(
    garment_input, garment_desc, category, material, gender, ethnicity, age_group, 
    selected_poses, light_direction, denoise_steps, seed, enable_sr, history_state,
    progress=gr.Progress(track_tqdm=True)
):
    """主生成入口：适配新 UI 参数"""
    
    garment = get_image_from_gradio(garment_input)
    if not garment:
        raise gr.Error("请先在 Step 01 上传一张主服装图！")

    # 处理特定姿态选择
    specific_pose_paths = None
    if selected_poses and len(selected_poses) > 0:
        # selected_poses 是 Gallery 返回的格式 [(path, label), ...]
        specific_pose_paths = []
        for p in selected_poses:
            path = p[0] if isinstance(p, (list, tuple)) else p
            if isinstance(path, dict): path = path.get("name") or path.get("path")
            specific_pose_paths.append(path)

    pipe = get_pipeline()

    # 进度映射
    _STAGE_BASE  = {"SAM提取Mask": 0.00, "IDM-VTON换装": 0.05, "IC-Light后处理": 0.65, "超分4x": 0.85}
    _STAGE_WIDTH = {"SAM提取Mask": 0.05, "IDM-VTON换装": 0.60, "IC-Light后处理": 0.20, "超分4x": 0.15}

    def onprogress(stage, cur, total):
        frac = cur / max(total, 1)
        base  = _STAGE_BASE.get(stage, 0.0)
        width = _STAGE_WIDTH.get(stage, 0.05)
        progress(base + frac * width, desc=f"{stage} ({cur}/{total})")

    # 动态计算 model_type
    is_child = "少年" in age_group or "Teens" in age_group
    prefix = "child" if is_child else "adult"
    if "女" in gender or "Female" in gender:
        suffix = "female"
    elif "男" in gender or "Male" in gender:
        suffix = "male"
    else:
        suffix = "neutral"
    model_type = f"{prefix}_{suffix}"

    # 执行流水线
    results = pipe.run(
        garment_image=garment,
        garment_desc=garment_desc or "a garment",
        category=category,
        model_type=model_type,
        light_direction=light_direction,
        enable_sr=enable_sr,
        denoise_steps=int(denoise_steps),
        seed=int(seed),
        specific_pose_paths=specific_pose_paths,
        material=material,
        ethnicity=ethnicity,
        age_group=age_group,
        progress_callback=onprogress,
    )

    # 收集输出图
    display_imgs = results.get("sr_paths") or results.get("final_paths") or []
    display_imgs = [p for p in display_imgs if p is not None]

    if not display_imgs:
        raise gr.Error("生成失败，请检查后端服务日志。")

    hist = history_state or []
    # 历史区按用户要求不能与当前生成区（主预览区）重复，
    # 因此本次 history_gallery 渲染的内容只包含先前的记录，不包含本次
    history_out = hist
    
    # 将本次结果存入状态，留待下一次点生成时作为“历史记录”展示
    new_state = display_imgs + hist
    
    return display_imgs, history_out, new_state

def auto_detect_ui(desc: str):
    if not desc or not desc.strip():
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    
    mat  = detect_material(desc)
    eth  = detect_ethnicity(desc)
    cat  = detect_category(desc)
    age  = detect_age_group(desc)
    gen  = detect_gender(desc)

    # 映射回 UI 中文选项
    mat_map = {
        "cotton": "棉质 / Cotton", "linen": "亚麻 / Linen", 
        "leather": "皮质 / Leather", "satin": "绸缎 / Satin",
        "denim": "牛仔 / Denim", "wool": "羊毛 / Wool"
    }
    eth_map = {
        "Caucasian": "白人 / Caucasian", "Asian": "亚洲人 / Asian",
        "Black": "黑人 / Black", "Mixed": "混血 / Mixed"
    }
    age_map = {
        "20s": "20岁组 / 20s", "30s": "30岁组 / 30s", "40s": "40岁组 / 40s", "Teens": "少年组 / Teens"
    }
    gen_map = {
        "adult_female": "女 / Female", "child_female": "女 / Female",
        "adult_male": "男 / Male", "child_male": "男 / Male",
        "adult_neutral": "中性 / Neutral", "child_neutral": "中性 / Neutral"
    }

    return (
        gr.update(value=cat),
        gr.update(value=mat_map.get(mat, "棉质 / Cotton")),
        gr.update(value=gen_map.get(gen, "女 / Female")),
        gr.update(value=eth_map.get(eth, "白人 / Caucasian")),
        gr.update(value=age_map.get(age, "20岁组 / 20s"))
    )

# ─────────────────────────────────────────────────────────────────────────────
# CSS 样式 (保持原样)
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── 全局变量与基础设置 ── */
:root, .dark, body.dark {
    --primary: #4f46e5;
    --primary-glow: rgba(79, 70, 229, 0.4);
    --sidebar-width: 260px;
    --card-bg: rgba(255, 255, 255, 0.7);
    --card-border: rgba(255, 255, 255, 0.9);
    --bg-main: #f3f4f6;
    --text-title: #0f172a;
    --text-body: #475569;
    
    /* 强行覆盖 Gradio 黑夜模式导致的输入框和容器变黑 */
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --block-background-fill: #ffffff !important;
    --input-background-fill: #f8fafc !important;
    --panel-background-fill: #ffffff !important;
    
    --body-text-color: #0f172a !important;
    --body-text-color-subdued: #475569 !important;
    --block-title-text-color: #0f172a !important;
    --block-label-text-color: #475569 !important;
    --input-text-color: #0f172a !important;
    
    --border-color-primary: #e2e8f0 !important;
    --block-border-color: #e2e8f0 !important;
    --input-border-color: #cbd5e1 !important;
}

body, .gradio-container {
    background-color: var(--bg-main) !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(79, 70, 229, 0.05) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(124, 58, 237, 0.05) 0px, transparent 50%) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* 隐藏 Gradio 默认页脚 */
footer { display: none !important; }

/* ── 侧边栏样式 ── */
#sidebar-col {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
    height: 100vh !important;
    padding: 24px 16px !important;
    position: fixed !important;
    left: 0; top: 0;
    width: var(--sidebar-width) !important;
    z-index: 1000;
}

#logo-area { margin-bottom: 40px; }
#logo-text { font-size: 20px; font-weight: 800; color: var(--primary); letter-spacing: -0.02em; line-height: 1.1; }
#studio-text { font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600; margin-top: 10px; }

.side-nav-btn {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; border-radius: 12px;
    color: #4b5563; font-weight: 600; font-size: 14px;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer; margin-bottom: 4px;
}
.side-nav-btn:hover { background: #f3f4f6; color: var(--primary); }
.side-nav-btn.active { background: var(--primary); color: white; box-shadow: 0 4px 12px var(--primary-glow); }

#user-card {
    position: absolute; bottom: 24px; left: 16px; right: 16px;
    padding: 16px; background: #f8fafc; border-radius: 14px;
    display: flex; align-items: center; gap: 12px;
}

/* ── 主内容区 ── */
#main-content-col {
    margin-left: var(--sidebar-width) !important;
    padding: 0 !important;
}

#top-bar {
    height: 64px; background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(8px); border-bottom: 1px solid #e5e7eb;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 40px; position: sticky; top: 0; z-index: 900;
}

/* ── 配置卡片 (步骤 01-04) ── */
.step-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px; padding: 16px; margin-bottom: 0px;
    box-shadow: 0 8px 12px -3px rgba(0, 0, 0, 0.03);
}

.step-header {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px; font-weight: 700; color: var(--primary);
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;
}

/* ── 预览显示 (带 AI 叠加层) ── */
#preview-container {
    position: relative; border-radius: 20px; overflow: hidden;
    background: #ffffff; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08);
}

#ai-overlay {
    position: absolute; top: 16px; right: 16px;
    background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(4px);
    padding: 8px 12px; border-radius: 10px;
    display: flex; flex-direction: column; gap: 4px; z-index: 10;
}
.history-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
#core-workspace { padding: 24px !important; gap: 24px !important; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="数字工坊 / THE DIGITAL ATELIER",
    head="""
    <script>
        // 强制禁用 Dark Mode (Gradio 将其挂载在 html 标签上)
        const observer = new MutationObserver(() => {
            if (document.documentElement.classList.contains('dark')) {
                document.documentElement.classList.remove('dark');
                document.documentElement.style.colorScheme = 'light';
            }
        });
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
        
        // 确保初始加载时也立刻移除
        document.addEventListener("DOMContentLoaded", () => {
            document.documentElement.classList.remove('dark');
            document.documentElement.style.colorScheme = 'light';
        });
    </script>
    """
) as demo:

    # ─── 0. 此处定义一个隐藏的 State 用于模拟积分或项目状态 ───
    credits = gr.State(84)

    with gr.Row(equal_height=False):
        
        # ─── 1. 左侧边栏 (Sidebar) ───
        with gr.Column(elem_id="sidebar-col", scale=0):
            gr.HTML("""
                <div id="logo-area">
                    <div id="logo-text">数字工坊 /<br>THE DIGITAL ATELIER</div>
                    <div id="studio-text">工作室 / STUDIO</div>
                    <div style="font-size:12px; font-weight:700; color:#1e293b; margin-top:10px;">AI 服装引擎 / AI Garment Engine</div>
                </div>
                <div class="side-nav-btn active">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                    上传 / Upload
                </div>
                <!-- ... 其他按钮 ... -->
                <div id="user-card">
                    <div style="width:32px; height:32px; background:#e2e8f0; border-radius:50%; display:flex; align-items:center; justify-content:center; color:var(--primary); font-weight:700; font-size:12px;">JS</div>
                    <div>
                        <div style="font-size:12px; font-weight:700; color:#1e293b;">创意总监 / Creative Director</div>
                        <div style="font-size:10px; color:#64748b;">专业版已激活 / Pro Plan</div>
                    </div>
                </div>
            """)

        # ─── 2. 主内容区 (Content Area) ───
        with gr.Column(elem_id="main-content-col", scale=1):
            
            # 2.1 顶部状态栏
            with gr.Row(elem_id="top-bar"):
                gr.HTML("""
                    <div style="display:flex; align-items:center; gap:12px;">
                        <span style="background:#eef2ff; color:var(--primary); padding:4px 10px; border-radius:6px; font-size:11px; font-weight:700;">项目 / PROJECT</span>
                        <span style="font-weight:700; color:#1e293b; font-size:15px;">Spring_Collection_2024.ai</span>
                    </div>
                """)

            # 2.2 核心工作区
            with gr.Row(elem_id="core-workspace"):
                
                # ── 左侧配置 ──
                with gr.Column(scale=4):
                    
                    # 步 01: 源服装上传
                    with gr.Column(elem_classes="step-card"):
                        gr.HTML('<div class="step-header">01. 源服装上传 / SOURCE GARMENT</div>')
                        garment_input = gr.Image(label="", show_label=False, elem_id="upload-area", type="pil", height=280)
                        garment_desc = gr.Textbox(
                            label="组图要求 / GARMENT DESCRIPTION", 
                            placeholder="请输入服装描述，例如：法式复古真丝连衣裙...",
                            lines=3,
                            elem_id="garment-desc"
                        )
                    
                    # 步 02: 服装属性
                    with gr.Column(elem_classes="step-card"):
                        gr.HTML('<div class="step-header">02. 服装属性设定 / GARMENT ATTRIBUTES</div>')
                        with gr.Row():
                            category = gr.Dropdown(
                                choices=[("上装 / Tops", "upper_body"), ("下装 / Bottoms", "lower_body"), ("连身衣 / Full", "dresses")],
                                value="upper_body", label="服装类别 / CATEGORY"
                            )
                            material = gr.Dropdown(
                                choices=["棉质 / Cotton", "亚麻 / Linen", "皮质 / Leather", "绸缎 / Satin", "牛仔 / Denim", "羊毛 / Wool"],
                                value="棉质 / Cotton", label="主要材质 / PRIMARY MATERIAL"
                            )
                    
                    # 步 03: 模特特征
                    with gr.Column(elem_classes="step-card"):
                        gr.HTML('<div class="step-header">03. 模特特征设定 / MODEL PERSONA</div>')
                        with gr.Row():
                            gender = gr.Dropdown(
                                choices=["女 / Female", "男 / Male", "中性 / Neutral"],
                                value="女 / Female", label="性别 / GENDER"
                            )
                            ethnicity = gr.Dropdown(
                                choices=["白人 / Caucasian", "亚洲人 / Asian", "黑人 / Black", "混血 / Mixed"],
                                value="白人 / Caucasian", label="种族 / ETHNICITY"
                            )
                            age_group = gr.Dropdown(
                                choices=["20岁组 / 20s", "30岁组 / 30s", "40岁组 / 40s", "少年组 / Teens"],
                                value="20岁组 / 20s", label="年龄段 / AGE"
                            )

                    # 步 04: 姿态选择
                    with gr.Column(elem_classes="step-card"):
                        gr.HTML('<div class="step-header">04. 姿态与动态 / POSE</div>')
                        pose_gallery = gr.Gallery(
                            label="", show_label=False, elem_id="pose-selector",
                            columns=3, rows=1, height=120, object_fit="cover",
                            allow_preview=False, selected_index=0
                        )
                        def update_poses_gallery(g_val, a_val):
                            is_child = "少年" in a_val or "Teens" in a_val
                            prefix = "child" if is_child else "adult"
                            if "女" in g_val or "Female" in g_val:
                                suffix = "female"
                            elif "男" in g_val or "Male" in g_val:
                                suffix = "male"
                            else:
                                suffix = "neutral"
                            mtype = f"{prefix}_{suffix}"
                            return get_pose_files(POSES_DIR, mtype)[:6]

                        gender.change(fn=update_poses_gallery, inputs=[gender, age_group], outputs=[pose_gallery])
                        age_group.change(fn=update_poses_gallery, inputs=[gender, age_group], outputs=[pose_gallery])

                        demo.load(
                            fn=update_poses_gallery,
                            inputs=[gender, age_group], 
                            outputs=[pose_gallery]
                        )

                    # 生成触发
                    gen_btn = gr.Button("开始 AI 换装 / START TRY-ON", variant="primary", elem_id="gen-btn")
                    
                    # 隐藏参数
                    light_direction = gr.State("top_left")
                    denoise_steps   = gr.State(30)
                    seed            = gr.State(42)
                    enable_sr       = gr.State(True)

                # ── 右侧预览 ──
                with gr.Column(scale=6):
                    with gr.Column(elem_id="preview-container"):
                        main_gallery = gr.Gallery(
                            label="", show_label=False, columns=5, height=640, 
                            object_fit="contain", preview=True, interactive=False
                        )
                        with gr.Row(elem_id="preview-controls"):
                            export = gr.Button("导出 / Export", variant="primary", elem_id="export-btn")

            # 2.3 最近生成 (底部)
            with gr.Column(elem_id="history-panel"):
                history_gallery = gr.Gallery(
                    label="最近生成 / RECENT GENERATIONS", show_label=True, columns=6, height=180, object_fit="cover"
                )

    history_state = gr.State([])

    # ── 事件绑定 ──
    gen_btn.click(
        fn=generate_v3,
        inputs=[
            garment_input, garment_desc, category, material, gender, ethnicity, age_group, pose_gallery,
            light_direction, denoise_steps, seed, enable_sr, history_state
        ],
        outputs=[main_gallery, history_gallery, history_state]
    )

    garment_desc.change(
        fn=auto_detect_ui, 
        inputs=[garment_desc], 
        outputs=[category, material, gender, ethnicity, age_group]
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(POSES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Base(primary_hue="indigo", font=gr.themes.GoogleFont("Inter")),
        css=CUSTOM_CSS,
    )
