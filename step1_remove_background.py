import os
import sys
from pathlib import Path
from PIL import Image
import warnings
from rembg import remove, new_session
warnings.filterwarnings('ignore')

DEFAULT_OUTPUT_DIR = "images\processed\step1_remove_background"

def remove_background(input_path, output_path, model_name='u2net'):
    """
    Background removal using a single AI model.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image  
        model_name: Model to use ('u2net' or 'u2net_cloth_seg')
    """
    print(f"\n🖼️  Processing: {input_path}")
    
    # Load image
    try:
        img = Image.open(input_path)
        print(f"📐 Image size: {img.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return False
    
    # Load AI model (downloads automatically on first use)
    print(f"🤖 Loading {model_name} model...")
    try:
        model = new_session(model_name)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Remove background
    print("🎨 Removing background...")
    try:
        # Simple removal without extra options
        output = remove(img, session=model)
        print("✅ Background removed successfully!")
    except Exception as e:
        print(f"❌ Error removing background: {e}")
        return False
    
    # Save result
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save as PNG to preserve transparency
        output.save(output_path, 'PNG')
        print(f"💾 Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return False

def process_batch(input_dir, output_dir, model_name='u2net'):
    """Process all images in a directory"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f"*{ext.lower()}"))
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return
    
    total_images = len(image_files)
    print(f"\n📁 Found {total_images} images to process")
    
    # Process each image
    successful = 0
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{total_images}] " + "="*40)
        
        output_file = output_path / f"{img_file.stem}.png"
        
        if remove_background(str(img_file), str(output_file), model_name):
            successful += 1
    
    print(f"\n✅ Successfully processed {successful}/{total_images} images")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI-powered background removal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python step1_remove_background.py images/raw/raw1.jpg
  
  # Use clothing-optimized model  
  python step1_remove_background.py images/raw/raw1.jpg --model u2net_cloth_seg
  
  # Process all images in folder (automatic detection)
  python step1_remove_background.py images/raw
  
Note: First run will download the model (~170MB).
        """
    )
    
    parser.add_argument("input", nargs='?', help="Input image or directory path")
    parser.add_argument("--model", default="u2net", 
                       choices=['u2net', 'u2net_cloth_seg'],
                       help="Model to use (default: u2net)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("🎯 Background Removal")
    print("================================")
    
    if args.input:
        input_path = Path(args.input)
        
        if not input_path.exists():
            print(f"❌ Input not found: {input_path}")
            return
        
        if input_path.is_dir():
            # Automatically detect directory and run batch processing
            print(f"📁 Directory detected: {input_path}")
            process_batch(str(input_path), args.output_dir, args.model)
        else:
            # Single file processing
            output_path = Path(args.output_dir) / f"{input_path.stem}.png"
            remove_background(str(input_path), str(output_path), args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()