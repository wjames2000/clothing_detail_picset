"""数据模型与检测工具模块"""
from .detectors import (
    detect_material,
    detect_gender, 
    detect_category,
    detect_ethnicity,
    detect_age_group,
    build_iclight_prompt,
)
from .quality import QualityChecker

__all__ = [
    "detect_material",
    "detect_gender",
    "detect_category", 
    "detect_ethnicity",
    "detect_age_group",
    "build_iclight_prompt",
    "QualityChecker",
]
