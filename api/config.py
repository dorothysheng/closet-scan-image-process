import os
from typing import Dict, Any

class APIConfig:
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
    RELOAD = True
    
    # API settings
    API_TITLE = "Closet Scan API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "API for Unreal Engine clothing digitization"
    
    # Unreal Engine specific settings
    MAX_IMAGE_SIZE = 4096  # Max texture size for UE
    SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".bmp"]
    
    # Processing settings
    BATCH_SIZE = 10
    TIMEOUT_SECONDS = 300
    
    # Directory paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(DATA_DIR, "output")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    
    # Model settings
    MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models", "cache")
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
            "host": cls.HOST,
            "port": cls.PORT,
            "reload": cls.RELOAD,
            "max_image_size": cls.MAX_IMAGE_SIZE,
            "supported_formats": cls.SUPPORTED_FORMATS,
            "directories": {
                "base": cls.BASE_DIR,
                "data": cls.DATA_DIR,
                "output": cls.OUTPUT_DIR,
                "processed": cls.PROCESSED_DIR,
                "model_cache": cls.MODEL_CACHE_DIR
            }
        }