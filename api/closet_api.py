"""
Closet Scan API

Simplified REST API for the clothing digitization pipeline.
Provides endpoints to process clothing images through the pipeline steps.
"""

import os
import sys
import base64
import uuid
import shutil
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline steps
from step1_remove_background import remove_background
from step3_texture import create_texture_with_overlay

# App initialization
app = FastAPI(
    title="Closet Scan API",
    description="API for processing clothing images through the digitization pipeline",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
PROCESSED_DIR = os.path.join(IMAGES_DIR, "processed")
BG_REMOVAL_DIR = os.path.join(PROCESSED_DIR, "step1_remove_background")
TEXTURE_DIR = os.path.join(PROCESSED_DIR, "step3_texture")
TEXTURE_EXTRACTION_DIR = os.path.join(PROCESSED_DIR, "texture_extraction")
BG_REMOVAL_ALT_DIR = os.path.join(PROCESSED_DIR, "background_removal")

for path in [IMAGES_DIR, PROCESSED_DIR, BG_REMOVAL_DIR, TEXTURE_DIR, TEXTURE_EXTRACTION_DIR, BG_REMOVAL_ALT_DIR]:
    os.makedirs(path, exist_ok=True)

model = None

@app.get("/")
async def root():
    return {
        "name": "Closet Scan API",
        "version": "1.0.0",
        "endpoints": [
            "/process_image",
            "/remove_background",
            "/extract_texture",
            "/classify"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/process_image")
async def process_image(file: UploadFile = File(...), make_seamless: bool = Form(False)):
    request_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    try:
        input_path = os.path.join(temp_dir, f"{request_id}_input.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        bg_removed_path = os.path.join(BG_REMOVAL_DIR, f"{request_id}.png")
        remove_background(input_path, bg_removed_path)

        texture_path = os.path.join(TEXTURE_DIR, f"{request_id}.png")
        create_texture_with_overlay(
            bg_removed_path,
            texture_path,
            target_size=(1024, 1024),
            patch_size=128,
            make_seamless=make_seamless
        )

        return FileResponse(texture_path, media_type="image/png", filename=f"{request_id}_texture.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir)

@app.post("/remove_background")
async def api_remove_background(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    try:
        input_path = os.path.join(temp_dir, f"{request_id}_input.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_path = os.path.join(BG_REMOVAL_ALT_DIR, f"{request_id}_no_bg.png")
        remove_background(input_path, output_path)

        return FileResponse(output_path, media_type="image/png", filename=f"{request_id}_no_bg.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir)

@app.post("/extract_texture")
async def api_extract_texture(file: UploadFile = File(...), make_seamless: bool = Form(False)):
    request_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    try:
        input_path = os.path.join(temp_dir, f"{request_id}_input.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_path = os.path.join(TEXTURE_EXTRACTION_DIR, f"{request_id}_texture.png")
        create_texture_with_overlay(input_path, output_path, target_size=(1024, 1024), patch_size=128, make_seamless=make_seamless)

        return FileResponse(output_path, media_type="image/png", filename=f"{request_id}_texture.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)

    uvicorn.run("closet_api:app", host="0.0.0.0", port=port, reload=True)
