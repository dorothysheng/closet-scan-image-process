import os
import uuid
import tempfile
import shutil
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def validate_image(file_path: str, max_size: int = 4096) -> Tuple[bool, Optional[str]]:
    """Validate image for Unreal Engine compatibility"""
    try:
        img = Image.open(file_path)
        
        # Check format
        if img.format not in ['PNG', 'JPEG', 'BMP']:
            return False, "Unsupported format. Use PNG, JPEG, or BMP."
        
        # Check size
        width, height = img.size
        if width > max_size or height > max_size:
            return False, f"Image too large. Max size is {max_size}x{max_size}."
        
        # Check if power of 2 (recommended for UE textures)
        if not (width & (width - 1) == 0 and height & (height - 1) == 0):
            logger.warning(f"Image dimensions ({width}x{height}) are not power of 2")
        
        return True, None
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def optimize_for_unreal(image_path: str, output_path: str) -> str:
    """Optimize image for Unreal Engine usage"""
    img = Image.open(image_path)
    
    # Convert to power of 2 dimensions
    width, height = img.size
    new_width = 2 ** int(np.ceil(np.log2(width)))
    new_height = 2 ** int(np.ceil(np.log2(height)))
    
    # Limit to max texture size
    new_width = min(new_width, 4096)
    new_height = min(new_height, 4096)
    
    # Resize if needed
    if (width, height) != (new_width, new_height):
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    # Save optimized image
    img.save(output_path, 'PNG', optimize=True)
    return output_path

def create_texture_metadata(texture_path: str) -> dict:
    """Create metadata for Unreal Engine texture import"""
    img = Image.open(texture_path)
    width, height = img.size
    
    return {
        "TextureSettings": {
            "Width": width,
            "Height": height,
            "Format": "PNG",
            "CompressionSettings": "TC_Default",
            "SRGB": True,
            "MipGenSettings": "TMGS_FromTextureGroup",
            "TextureGroup": "TEXTUREGROUP_UI"
        },
        "ImportSettings": {
            "bAllowNonPowerOfTwo": True,
            "CompressionQuality": "TCQ_Default",
            "bFlipGreenChannel": False
        }
    }