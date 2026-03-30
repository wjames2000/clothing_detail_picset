"""
图像操作工具函数
"""
from typing import Tuple, Optional
from PIL import Image


def resize_and_pad(
    image: Image.Image, 
    target_size: Tuple[int, int] = (768, 1024), 
    fill_color: Tuple[int, int, int] = (255, 255, 255)
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    保持长宽比缩放图片，并在两侧补边至目标尺寸。
    
    Args:
        image: 输入 PIL Image
        target_size: 目标尺寸 (width, height)
        fill_color: 填充颜色 (R, G, B)
    
    Returns:
        (padded_image, (x_offset, y_offset, new_w, new_h))
    """
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    
    # 计算缩放比例
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    
    resized_img = image.resize((new_w, new_h), Image.LANCZOS)
    
    # 创建背景并居中贴图
    padded_img = Image.new("RGB", target_size, fill_color)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_img.paste(resized_img, (x_offset, y_offset))
    
    return padded_img, (x_offset, y_offset, new_w, new_h)


def unpad_and_resize(
    padded_image: Image.Image, 
    padding_info: Tuple[int, int, int, int], 
    final_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    根据 padding 信息裁掉边框，并可选缩放回最终尺寸。
    
    Args:
        padded_image: 带边框的 PIL Image
        padding_info: (x_offset, y_offset, new_w, new_h)
        final_size: 可选的最终尺寸 (width, height)
    
    Returns:
        裁剪后的 PIL Image
    """
    x, y, w, h = padding_info
    img_cropped = padded_image.crop((x, y, x + w, y + h))
    if final_size:
        return img_cropped.resize(final_size, Image.LANCZOS)
    return img_cropped
