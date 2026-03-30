"""
服装属性检测器
基于关键词匹配的材质、类别、性别、种族、年龄段识别
"""
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# 材质关键词映射表
# ─────────────────────────────────────────────────────────────────────────────

_MATERIAL_KEYWORDS: Dict[str, List[str]] = {
    "leather": ["leather", "leatherette", "faux leather", "pu leather", "leather-like",
                "皮质", "皮革", "真皮", "pu 皮", "人造皮", "皮衣", "机车服"],
    "satin":   ["satin", "silk", "silky", "glossy", "shiny", "polyester silk",
                "绸缎", "丝缎", "缎面", "丝绸", "光泽", "亮面", "真丝", "桑蚕丝", "醋酸", "亮片"],
    "denim":   ["denim", "jeans", "jean", "washed denim",
                "牛仔", "水洗牛仔", "丹宁", "牛仔布"],
    "chiffon": ["chiffon", "georgette", "sheer", "transparent", "mesh",
                "雪纺", "乔其纱", "薄纱", "网眼", "透明", "透视"],
    "linen":   ["linen", "flax", "hemp",
                "亚麻", "麻料", "棉麻", "汉麻"],
    "cotton":  ["cotton", "jersey", "combed cotton",
                "棉质", "全棉", "纯棉", "精梳棉", "针织棉", "汗布"],
    "wool":    ["wool", "cashmere", "knit", "knitted", "sweater", "cardigan",
                "毛料", "羊毛", "羊绒", "针织", "编织", "毛衣", "开衫"],
    "velvet":  ["velvet", "velour", "suede", "flocked",
                "丝绒", "天鹅绒", "金丝绒", "磨毛", "麂皮"],
    "lace":    ["lace", "embroidery", "hollow out",
                "蕾丝", "刺绣", "镂空", "钩编"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 类别关键词映射表
# ─────────────────────────────────────────────────────────────────────────────

_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "lower_body": ["pants", "trousers", "shorts", "skirt", "jeans", "leggings", "bottom",
                   "裤", "裙", "短裤", "长裤", "牛仔裤", "半身裙", "下装", "运动裤", "休闲裤"],
    "dresses":    ["dress", "gown", "robe", "one-piece", "jumpsuit",
                   "连衣裙", "长裙", "礼服", "旗袍", "连体衣", "连身裙", "连体衫"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 性别与人群关键词映射表
# ─────────────────────────────────────────────────────────────────────────────

_GENDER_KEYWORDS: Dict[str, List[str]] = {
    "adult_female":  ["woman", "women", "female", "lady", "girl", "her", "she", 
                      "女", "女性", "女士", "女孩", "美女", "熟女"],
    "adult_male":    ["man", "men", "male", "guy", "boy", "his", "he", "him",
                      "男", "男性", "男士", "男孩", "帅哥", "型男"],
    "adult_neutral": ["neutral", "androgynous", "unisex", "中性", "男女同款", "无性别"],
    "child_female":  ["child girl", "little girl", "toddler girl", "kid girl",
                      "女童", "女孩儿", "小女孩", "小女生"],
    "child_male":    ["child boy", "little boy", "toddler boy", "kid boy",
                      "男童", "男孩儿", "小男孩", "小男生"],
    "child_neutral": ["child neutral", "unisex child", "中性童装", "男女童"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 光源方向 → IC-Light prompt 片段
# ─────────────────────────────────────────────────────────────────────────────

_LIGHT_DIR_PROMPTS: Dict[str, str] = {
    "top_left":    "soft directional light from upper left at 45 degrees",
    "top_right":   "soft directional light from upper right at 45 degrees",
    "top":         "soft overhead lighting from directly above",
    "front":       "soft frontal studio lighting",
}

DEFAULT_LIGHT_DIR = "top_left"


def detect_category(garment_desc: str) -> str:
    """
    从服装描述文本中检测类别。
    
    Args:
        garment_desc: 服装描述文本
        
    Returns:
        'upper_body' (默认) | 'lower_body' | 'dresses'
    """
    if not garment_desc:
        return "upper_body"
    desc_lower = garment_desc.lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw.lower() in desc_lower for kw in keywords):
            print(f"[Category] 识别到类别：{cat} (desc='{garment_desc}')")
            return cat
    return "upper_body"


def detect_material(garment_desc: str) -> str:
    """
    从服装描述文本中检测材质类型。
    
    Args:
        garment_desc: 服装描述文本
        
    Returns:
        材质键名或 'general'
    """
    if not garment_desc:
        return "general"
    desc_lower = garment_desc.lower()
    for material, keywords in _MATERIAL_KEYWORDS.items():
        if any(kw.lower() in desc_lower for kw in keywords):
            print(f"[Material] 识别到材质：{material} (desc='{garment_desc}')")
            return material
    return "general"


def detect_gender(garment_desc: str) -> str:
    """
    从服装描述文本中检测性别/人群。
    
    Args:
        garment_desc: 服装描述文本
        
    Returns:
        'adult_female' (默认) | 'adult_male' | 'adult_neutral' | 
        'child_female' | 'child_male' | 'child_neutral'
    """
    if not garment_desc:
        return "adult_female"
    desc_lower = garment_desc.lower()
    
    # 优先检测儿童
    if any(kw in desc_lower for kw in ["child", "kid", "little", "toddler", "童", "小"]):
        if any(kw in desc_lower for kw in ["neutral", "unisex", "中性", "男女"]):
            return "child_neutral"
        if any(kw in desc_lower for kw in ["girl", "female", "女"]):
            return "child_female"
        if any(kw in desc_lower for kw in ["boy", "male", "男"]):
            return "child_male"
            
    # 普通成人检测
    for gender, keywords in _GENDER_KEYWORDS.items():
        if any(kw.lower() in desc_lower for kw in keywords):
            return gender
            
    return "adult_female"


def detect_ethnicity(garment_desc: str) -> str:
    """
    从服装描述中检测人种。
    
    Args:
        garment_desc: 服装描述文本
        
    Returns:
        'Caucasian' (默认) | 'Asian' | 'Black' | 'Middle Eastern'
    """
    desc_lower = garment_desc.lower() if garment_desc else ""
    if any(kw in desc_lower for kw in ["亚洲", "华", "中式", "asian", "oriental", "chinese"]):
        return "Asian"
    if any(kw in desc_lower for kw in ["非", "黑", "african", "black"]):
        return "Black"
    return "Caucasian"


def detect_age_group(garment_desc: str) -> str:
    """
    从服装描述中检测年龄段。
    
    Args:
        garment_desc: 服装描述文本
        
    Returns:
        '20s' (默认) | '30s' | '40s' | 'teens'
    """
    desc_lower = garment_desc.lower() if garment_desc else ""
    if any(kw in desc_lower for kw in ["中年", "30", "40", "mature"]):
        return "40s"
    if any(kw in desc_lower for kw in ["青", "少", "18", "teens"]):
        return "Teens"
    return "20s"


def build_iclight_prompt(
    light_direction: str = DEFAULT_LIGHT_DIR,
    material: str = "general",
    specular_boost: float = 1.0,
) -> dict:
    """
    根据光源方向和材质构建 IC-Light 的 prompt / negative_prompt 和 guidance_scale。
    
    Args:
        light_direction: 光源方向
        material: 材质类型
        specular_boost: 高光增强系数
        
    Returns:
        {
          "prompt": str,
          "negative_prompt": str,
          "guidance_scale": float,
          "specular_boost": float,
        }
    """
    light_desc = _LIGHT_DIR_PROMPTS.get(light_direction, _LIGHT_DIR_PROMPTS[DEFAULT_LIGHT_DIR])

    # 基础 prompt
    base_prompt = (
        f"a person wearing clothes, {light_desc}, "
        "natural shadow, realistic lighting, commercial fashion photography, high quality, 8k"
    )

    # 材质特化 prompt 追加
    if material == "leather":
        base_prompt += (
            ", leather texture, subtle specular highlight, "
            "3D depth and thickness, rich surface detail"
        )
    elif material == "satin":
        base_prompt += (
            ", satin sheen, smooth silky surface, "
            "elegant specular reflection, 3D depth and thickness"
        )

    negative_prompt = (
        "overexposed, flat lighting, dark, blurry, noisy, "
        "low quality, cartoon, painting, illustration"
    )

    # 高光材质提升 guidance_scale 以增强对比度和高光响应
    guidance_scale = 7.5 * specular_boost if specular_boost > 1.0 else 7.5

    return {
        "prompt":         base_prompt,
        "negative_prompt": negative_prompt,
        "guidance_scale": min(guidance_scale, 15.0),
        "specular_boost": specular_boost,
    }
