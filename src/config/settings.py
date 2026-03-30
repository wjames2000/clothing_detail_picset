"""
全局配置管理
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Settings:
    """系统配置类，使用 dataclass 提供类型安全和默认值"""
    
    # 服务配置
    idmvton_url: str = field(
        default_factory=lambda: os.environ.get("IDMVTON_URL", "http://127.0.0.1:7861/")
    )
    
    # 目录配置
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(__file__)))
    poses_dir: Optional[str] = None
    outputs_dir: Optional[str] = None
    ckpt_dir: Optional[str] = None
    
    # 模型权重路径
    sam_ckpt: Optional[str] = None
    iclight_ckpt: Optional[str] = None
    
    # 流水线参数
    max_poses: int = 5
    max_qc_retries: int = 3
    
    # 质量控制阈值
    qc_color_thresh: float = 0.25
    qc_blur_thresh: float = 50.0
    qc_edge_thresh: float = 0.015
    qc_shadow_thresh: float = 20.0
    
    # 模特类型映射
    model_type_dirs: Dict[str, str] = field(default_factory=lambda: {
        "adult_female": "adult_female",
        "adult_male": "adult_male",
        "adult_neutral": "adult_neutral",
        "child_female": "child_female",
        "child_male": "child_male",
        "child_neutral": "child_neutral",
    })
    
    # 光源方向映射
    light_dir_prompts: Dict[str, str] = field(default_factory=lambda: {
        "top_left": "soft directional light from upper left at 45 degrees",
        "top_right": "soft directional light from upper right at 45 degrees",
        "top": "soft overhead lighting from directly above",
        "front": "soft frontal studio lighting",
    })
    
    def __post_init__(self):
        """初始化后处理，设置默认路径"""
        if self.poses_dir is None:
            self.poses_dir = os.path.join(self.base_dir, "poses")
        if self.outputs_dir is None:
            self.outputs_dir = os.path.join(self.base_dir, "outputs")
        if self.ckpt_dir is None:
            self.ckpt_dir = os.path.join(self.base_dir, "ckpt")
        if self.sam_ckpt is None:
            self.sam_ckpt = os.path.join(self.ckpt_dir, "sam_vit_h_4b8939.pth")
        if self.iclight_ckpt is None:
            self.iclight_ckpt = os.path.join(self.ckpt_dir, "iclight_sd15_fc.safetensors")
    
    def ensure_dirs(self):
        """确保所有必要目录存在"""
        for directory in [self.poses_dir, self.outputs_dir, self.ckpt_dir]:
            os.makedirs(directory, exist_ok=True)


# 全局单例
settings = Settings()
