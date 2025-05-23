"""
Enhanced Photo to Texture Converter for Closet Scan

Texture extraction pipeline for 3D clothing models:
- Extracts patterns from clothing images
- Creates seamless background textures with cross-blending
- Overlays original clothing on pattern background
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

DEFAULT_OUTPUT_DIR = "images\processed\step3_texture"

def extract_pattern_from_image(image, alpha=None, patch_size=128, min_opacity=0.5):
    """Extract pattern patches from the visible parts of the image with their positions."""
    h, w = image.shape[:2]
    patches = []
    positions = []
    
    # If no alpha channel provided, assume everything is visible
    if alpha is None:
        alpha = np.ones((h, w), dtype=np.uint8) * 255
    
    # Adjust patch size if needed
    if patch_size > min(h, w) // 2:
        patch_size = max(64, min(h, w) // 3)  # Ensure reasonable patch size
        print(f"  • Adjusted patch size to {patch_size} to fit image")
    
    # Use a sliding window to find good pattern patches
    step_size = patch_size // 2  # 50% overlap
    
    for y in range(0, h - patch_size + 1, step_size):
        for x in range(0, w - patch_size + 1, step_size):
            # Extract patch and its alpha
            patch = image[y:y+patch_size, x:x+patch_size].copy()
            patch_alpha = alpha[y:y+patch_size, x:x+patch_size]
            
            # Check if patch is mostly visible (not transparent)
            visibility = np.mean(patch_alpha) / 255.0
            
            # Check if patch has good color variation
            std_dev = np.std(patch.reshape(-1, 3))
            
            # Only use patches with good visibility and color variation
            if visibility >= min_opacity and std_dev > 10:
                patches.append(patch)
                positions.append((y, x))  # Store the original position
    
    if patches:
        print(f"  ✓ Extracted {len(patches)} pattern patches with positions")
    else:
        print("  ⚠ No suitable pattern patches found")
        # If no patches found, create a simple pattern from the image
        avg_color = np.mean(image[alpha > 128], axis=0).astype(np.uint8)
        simple_patch = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * avg_color
        patches.append(simple_patch)
        positions.append((h//2, w//2))  # Center position
    
    return patches, positions

def create_pattern_background(patches_with_positions, target_size=(1024, 1024), blend_width=64):
    """Create a tiled background using pattern patches with cross-blending.
    Preserves spatial relationships from the original image and eliminates cross-bar artifacts."""
    # Unpack patches and positions
    if isinstance(patches_with_positions, tuple) and len(patches_with_positions) == 2:
        patches, positions = patches_with_positions
    else:
        # For backward compatibility
        patches = patches_with_positions
        positions = None
    
    target_h, target_w = target_size
    
    # Create an empty canvas and weight accumulation
    result = np.zeros((target_h, target_w, 3), dtype=np.float32)
    weight_acc = np.zeros((target_h, target_w), dtype=np.float32)
    
    # If we have only one patch, just tile it
    if len(patches) <= 1:
        patch = patches[0]
        patch_h, patch_w = patch.shape[:2]
        
        # Tile the pattern across the target image
        for y in range(0, target_h, patch_h):
            for x in range(0, target_w, patch_w):
                # Calculate the region to fill
                y_end = min(y + patch_h, target_h)
                x_end = min(x + patch_w, target_w)
                
                # Copy the appropriate portion of the patch
                result[y:y_end, x:x_end] = patch[:y_end-y, :x_end-x]
        
        print(f"  ✓ Created pattern background of size {target_w}x{target_h} using single patch")
        return result.astype(np.uint8)
    
    # For multiple patches, create a pattern that preserves spatial relationships
    print(f"  • Creating pattern background with {len(patches)} patches using cross-blending")
    
    # Get average patch dimensions
    patch_sizes = [patch.shape[:2] for patch in patches]
    avg_patch_h = int(np.mean([h for h, w in patch_sizes]))
    avg_patch_w = int(np.mean([w for h, w in patch_sizes]))
    
    # Use larger patches with significant overlap for better blending
    overlap_factor = 0.8  # 80% overlap between patches for smoother transitions (increased from 70%)
    step_h = int(avg_patch_h * (1 - overlap_factor))
    step_w = int(avg_patch_w * (1 - overlap_factor))
    
    # Ensure steps are at least 8 pixels (reduced from 10 for more dense coverage)
    step_h = max(step_h, 8)
    step_w = max(step_w, 8)
    
    # Calculate grid dimensions with overlap
    grid_h = (target_h + step_h - 1) // step_h + 1  # Add extra row/column for better coverage
    grid_w = (target_w + step_w - 1) // step_w + 1
    
    # If we have positions, use them to determine which patches go where
    if positions:
        # Find the original image dimensions
        orig_positions = np.array(positions)
        orig_h_max = np.max(orig_positions[:, 0]) + avg_patch_h
        orig_w_max = np.max(orig_positions[:, 1]) + avg_patch_w
        
        # For each cell in the target grid, find the closest patch from the original image
        for i in range(grid_h):
            for j in range(grid_w):
                # Calculate the region to fill with overlap
                y_start = i * step_h
                x_start = j * step_w
                
                # Skip if outside the target area
                if y_start >= target_h or x_start >= target_w:
                    continue
                
                y_end = min(y_start + avg_patch_h, target_h)
                x_end = min(x_start + avg_patch_w, target_w)
                
                # Skip if this would be a tiny region
                if y_end - y_start < avg_patch_h // 4 or x_end - x_start < avg_patch_w // 4:
                    continue
                
                # Map grid position to original image coordinates
                orig_y = int(y_start * orig_h_max / target_h)
                orig_x = int(x_start * orig_w_max / target_w)
                
                # Find the closest patch from the original positions
                distances = []
                for idx, (py, px) in enumerate(positions):
                    # Calculate distance to this patch
                    dist = np.sqrt((py - orig_y)**2 + (px - orig_x)**2)
                    distances.append((dist, idx))
                
                # Sort by distance and get the closest patches
                distances.sort()
                closest_idx = distances[0][1]
                patch = patches[closest_idx]
                
                # Resize patch to standard size if needed
                if patch.shape[0] != avg_patch_h or patch.shape[1] != avg_patch_w:
                    patch = cv2.resize(patch, (avg_patch_w, avg_patch_h), interpolation=cv2.INTER_AREA)
                
                # Create a smooth radial blend mask (stronger in center, weaker at edges)
                h_region = y_end - y_start
                w_region = x_end - x_start
                blend_mask = np.ones((h_region, w_region), dtype=np.float32)
                
                # Create a radial gradient from center for smoother blending
                cy = h_region // 2
                cx = w_region // 2
                
                # Calculate the maximum distance from center to corner
                max_dist = np.sqrt(cy**2 + cx**2)
                
                # Create a smooth radial falloff
                y_coords, x_coords = np.ogrid[:h_region, :w_region]
                dist_from_center = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
                # Normalize distance and create smooth falloff (cosine falloff)
                normalized_dist = dist_from_center / max_dist
                # Weight is 1 at center, decreases smoothly toward edges
                blend_mask = 0.5 + 0.5 * np.cos(np.minimum(normalized_dist * np.pi, np.pi))
                
                # Get the portion of the patch that fits
                patch_portion = patch[:h_region, :w_region]
                
                # Apply weighted accumulation (for each color channel)
                for c in range(3):
                    result[y_start:y_end, x_start:x_end, c] += patch_portion[:, :, c] * blend_mask
                
                # Accumulate weights
                weight_acc[y_start:y_end, x_start:x_end] += blend_mask
    else:
        # If no positions, create a smooth gradient of patches
        patch_count = len(patches)
        for i in range(grid_h):
            for j in range(grid_w):
                # Calculate the region to fill with overlap
                y_start = i * step_h
                x_start = j * step_w
                
                # Skip if outside the target area
                if y_start >= target_h or x_start >= target_w:
                    continue
                
                y_end = min(y_start + avg_patch_h, target_h)
                x_end = min(x_start + avg_patch_w, target_w)
                
                # Skip if this would be a tiny region
                if y_end - y_start < avg_patch_h // 4 or x_end - x_start < avg_patch_w // 4:
                    continue
                
                # Use a pattern that depends on position to create a natural flow
                patch_idx = ((i + j) * 3) % patch_count
                patch = patches[patch_idx]
                
                # Resize patch to standard size if needed
                if patch.shape[0] != avg_patch_h or patch.shape[1] != avg_patch_w:
                    patch = cv2.resize(patch, (avg_patch_w, avg_patch_h), interpolation=cv2.INTER_AREA)
                
                # Create a smooth radial blend mask (stronger in center, weaker at edges)
                h_region = y_end - y_start
                w_region = x_end - x_start
                blend_mask = np.ones((h_region, w_region), dtype=np.float32)
                
                # Create a radial gradient from center for smoother blending
                cy = h_region // 2
                cx = w_region // 2
                
                # Calculate the maximum distance from center to corner
                max_dist = np.sqrt(cy**2 + cx**2)
                
                # Create a smooth radial falloff
                y_coords, x_coords = np.ogrid[:h_region, :w_region]
                dist_from_center = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
                # Normalize distance and create smooth falloff (cosine falloff)
                normalized_dist = dist_from_center / max_dist
                # Weight is 1 at center, decreases smoothly toward edges
                blend_mask = 0.5 + 0.5 * np.cos(np.minimum(normalized_dist * np.pi, np.pi))
                
                # Get the portion of the patch that fits
                patch_portion = patch[:h_region, :w_region]
                
                # Apply weighted accumulation (for each color channel)
                for c in range(3):
                    result[y_start:y_end, x_start:x_end, c] += patch_portion[:, :, c] * blend_mask
                
                # Accumulate weights
                weight_acc[y_start:y_end, x_start:x_end] += blend_mask
    
    # Normalize by accumulated weights to get the final result
    # Avoid division by zero
    weight_acc = np.maximum(weight_acc, 0.001)
    
    # Normalize each channel
    for c in range(3):
        result[:, :, c] /= weight_acc
    
    # Convert to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Apply a final Gaussian blur to smooth any remaining harsh transitions
    result = cv2.GaussianBlur(result, (5, 5), 0)
    
    print(f"  ✓ Created cross-blended pattern background of size {target_w}x{target_h} preserving spatial relationships")
    return result

def make_seamless_texture(texture, blend_width=50):
    """Create seamlessly tileable texture."""
    h, w = texture.shape[:2]
    
    # Create a mask for blending the edges
    mask = np.ones((h, w), dtype=np.float32)
    
    # Create gradual falloff at the edges
    for i in range(blend_width):
        # Calculate weight based on distance from edge (0 at edge, 1 at blend_width)
        weight = i / blend_width
        
        # Apply to all four edges
        mask[i, :] *= weight  # Top edge
        mask[h-i-1, :] *= weight  # Bottom edge
        mask[:, i] *= weight  # Left edge
        mask[:, w-i-1] *= weight  # Right edge
    
    # Create a larger canvas for the seamless texture
    seamless = np.zeros((h, w, 3), dtype=np.float32)
    weight_acc = np.zeros((h, w), dtype=np.float32)
    
    # Place the original texture in the center
    seamless += texture * mask[:, :, np.newaxis]
    weight_acc += mask
    
    # Place wrapped copies at the edges with blending
    # Top-left corner
    seamless[0:blend_width, 0:blend_width] += texture[h-blend_width:h, w-blend_width:w] * (1 - mask[0:blend_width, 0:blend_width])[:, :, np.newaxis]
    weight_acc[0:blend_width, 0:blend_width] += (1 - mask[0:blend_width, 0:blend_width])
    
    # Top edge
    seamless[0:blend_width, blend_width:w-blend_width] += texture[h-blend_width:h, blend_width:w-blend_width] * (1 - mask[0:blend_width, blend_width:w-blend_width])[:, :, np.newaxis]
    weight_acc[0:blend_width, blend_width:w-blend_width] += (1 - mask[0:blend_width, blend_width:w-blend_width])
    
    # Top-right corner
    seamless[0:blend_width, w-blend_width:w] += texture[h-blend_width:h, 0:blend_width] * (1 - mask[0:blend_width, w-blend_width:w])[:, :, np.newaxis]
    weight_acc[0:blend_width, w-blend_width:w] += (1 - mask[0:blend_width, w-blend_width:w])
    
    # Right edge
    seamless[blend_width:h-blend_width, w-blend_width:w] += texture[blend_width:h-blend_width, 0:blend_width] * (1 - mask[blend_width:h-blend_width, w-blend_width:w])[:, :, np.newaxis]
    weight_acc[blend_width:h-blend_width, w-blend_width:w] += (1 - mask[blend_width:h-blend_width, w-blend_width:w])
    
    # Bottom-right corner
    seamless[h-blend_width:h, w-blend_width:w] += texture[0:blend_width, 0:blend_width] * (1 - mask[h-blend_width:h, w-blend_width:w])[:, :, np.newaxis]
    weight_acc[h-blend_width:h, w-blend_width:w] += (1 - mask[h-blend_width:h, w-blend_width:w])
    
    # Bottom edge
    seamless[h-blend_width:h, blend_width:w-blend_width] += texture[0:blend_width, blend_width:w-blend_width] * (1 - mask[h-blend_width:h, blend_width:w-blend_width])[:, :, np.newaxis]
    weight_acc[h-blend_width:h, blend_width:w-blend_width] += (1 - mask[h-blend_width:h, blend_width:w-blend_width])
    
    # Bottom-left corner
    seamless[h-blend_width:h, 0:blend_width] += texture[0:blend_width, w-blend_width:w] * (1 - mask[h-blend_width:h, 0:blend_width])[:, :, np.newaxis]
    weight_acc[h-blend_width:h, 0:blend_width] += (1 - mask[h-blend_width:h, 0:blend_width])
    
    # Left edge
    seamless[blend_width:h-blend_width, 0:blend_width] += texture[blend_width:h-blend_width, w-blend_width:w] * (1 - mask[blend_width:h-blend_width, 0:blend_width])[:, :, np.newaxis]
    weight_acc[blend_width:h-blend_width, 0:blend_width] += (1 - mask[blend_width:h-blend_width, 0:blend_width])
    
    # Normalize by accumulated weights
    weight_acc = np.maximum(weight_acc, 0.001)[:, :, np.newaxis]
    seamless = seamless / weight_acc
    
    # Convert to uint8
    seamless = np.clip(seamless, 0, 255).astype(np.uint8)
    
    return seamless

def create_texture_with_overlay(image_path, output_path=None, target_size=(1024, 1024), 
                               patch_size=128, make_seamless=False, enlarge_ratio=1.0):
    """Create a texture by extracting patterns and overlaying the original image.
    Preserves spatial relationships from the original image.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output texture
        target_size: Size of the output texture (width, height)
        patch_size: Size of pattern patches to extract
        make_seamless: Whether to make the texture seamlessly tileable
        enlarge_ratio: Factor to enlarge the image before extracting patterns (1.0 = no change)
    """
    print(f"\n🎯 Processing: {image_path}")
    print("="*50)
    
    # Read image with PIL to properly handle transparency
    print("➤ Loading image...")
    try:
        pil_img = Image.open(image_path)
        
        # Convert to RGBA if not already
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
            
        # Get alpha channel
        r, g, b, alpha = pil_img.split()
        
        # Get original image dimensions
        orig_w, orig_h = pil_img.size
        print(f"  • Original image size: {orig_w}x{orig_h}")
        
        # Apply enlargement if ratio is not 1.0
        if enlarge_ratio != 1.0:
            print(f"  • Enlarging image by factor of {enlarge_ratio}...")
            new_w = int(orig_w * enlarge_ratio)
            new_h = int(orig_h * enlarge_ratio)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            # Get new alpha channel after resize
            r, g, b, alpha = pil_img.split()
            print(f"  • Enlarged image size: {new_w}x{new_h}")
        
        # Shrink edges by 2 pixels to remove dark borders
        current_w, current_h = pil_img.size
        if current_w > 4 and current_h > 4:  # Only if image is large enough
            print("  • Shrinking edges by 2 pixels to remove dark borders...")
            # Create a new transparent image
            shrunk_img = Image.new('RGBA', (current_w-4, current_h-4), (0,0,0,0))
            # Paste the center of the original image
            shrunk_img.paste(pil_img.crop((2, 2, current_w-2, current_h-2)), (0, 0))
            pil_img = shrunk_img
            # Get new alpha channel after shrinking
            r, g, b, alpha = pil_img.split()
            print(f"  • New image size after edge removal: {pil_img.width}x{pil_img.height}")
        
        # Convert to numpy arrays
        np_img = np.array(pil_img.convert('RGB'))
        np_alpha = np.array(alpha)
        
        # Get current image dimensions
        h, w = np_img.shape[:2]
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Extract pattern patches from the visible parts with their positions
    print("➤ Extracting pattern patches with spatial information...")
    patches_with_positions = extract_pattern_from_image(np_img, np_alpha, patch_size=patch_size)
    
    # Create pattern background that preserves spatial relationships
    print("➤ Creating pattern background with spatial relationships...")
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    background = create_pattern_background(patches_with_positions, target_size, blend_width=64)
    
    # Make seamless if requested
    if make_seamless:
        print("➤ Making background seamless...")
        background = make_seamless_texture(background)
    
    # Resize the original image to fit in the center of the target
    print("➤ Placing original image on background...")
    
    # Calculate the scale factor to fit the image within the target
    target_h, target_w = target_size
    scale_factor = min(target_w / w, target_h / h) * 0.9  # 90% of the target size
    
    # Resize the original image
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # Resize with PIL to preserve alpha
    resized_pil = pil_img.resize((new_w, new_h), Image.LANCZOS)
    
    # Create a new PIL image with the background
    result_pil = Image.fromarray(background)
    
    # Calculate position to center the original image
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    
    # Paste the original image onto the background
    result_pil.paste(resized_pil, (paste_x, paste_y), resized_pil)
    
    # Convert back to numpy for OpenCV operations
    result = np.array(result_pil)
    
    # Save result
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with PIL to preserve quality
        result_pil.save(output_path, format='PNG')
        print(f"\n✅ Saved texture to: {output_path}")
    
    print("="*50)
    return result

def visualize_process(original, mask, filled, seamless=None):
    """Display processing steps."""
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(221)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Mask
    plt.subplot(222)
    plt.title("Alpha Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    # Filled texture
    plt.subplot(223)
    plt.title("Pattern-filled Texture")
    plt.imshow(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Seamless texture (if available)
    if seamless is not None:
        plt.subplot(224)
        plt.title("Seamless Texture")
        plt.imshow(cv2.cvtColor(seamless, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_batch(input_dir, output_dir, target_size=1024, patch_size=128, make_seamless=False, enlarge_ratio=1.0):
    """Process all images in a directory"""
    
    from pathlib import Path
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f"*{ext.lower()}"))
        image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
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
        os.makedirs(output_path, exist_ok=True)
        
        result = create_texture_with_overlay(
            str(img_file),
            str(output_file),
            target_size=(target_size, target_size),
            patch_size=patch_size,
            make_seamless=make_seamless,
            enlarge_ratio=enlarge_ratio
        )
        
        if result is not None:
            successful += 1
    
    print(f"\n✅ Successfully processed {successful}/{total_images} images")

def main():
    """Process command line arguments and run the texture extraction."""
    import argparse
    from pathlib import Path
    
    # Check if output directory exists, create if not
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    
    parser = argparse.ArgumentParser(
        description="Texture extraction for clothing images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python step3_texture.py images/processed/step1_remove_background/shirt.png
  
  # Process all images in folder
  python step3_texture.py images/processed/step1_remove_background
  
  # Specify target size
  python step3_texture.py images/processed/step1_remove_background/shirt.png --size 2048
        """
    )
    
    parser.add_argument("input", help="Input image or directory path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                       help="Output directory")
    parser.add_argument("--size", type=int, default=1024,
                       help="Target texture size (default: 1024)")
    parser.add_argument("--patch-size", type=int, default=128,
                       help="Pattern patch size (default: 128)")
    parser.add_argument("--seamless", action="store_true",
                       help="Make texture seamlessly tileable")
    parser.add_argument("--enlarge", type=float, default=1.0,
                       help="Enlarge image before pattern extraction (e.g., 2.0 doubles the size)")
    
    
    args = parser.parse_args()
    
    print("\n🎨 Closet Scan Texture Extraction")
    print("="*50)
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ Input not found: {input_path}")
        return
    
    if input_path.is_dir():
        # Automatically detect directory and run batch processing
        print(f"📁 Directory detected: {input_path}")
        process_batch(
            str(input_path), 
            args.output_dir, 
            target_size=args.size,
            patch_size=args.patch_size,
            make_seamless=args.seamless,
            enlarge_ratio=args.enlarge
        )
    else:
        # Single file processing
        output_path = Path(args.output_dir) / f"{input_path.stem}.png"
        result = create_texture_with_overlay(
            str(input_path),
            str(output_path),
            target_size=(args.size, args.size),
            patch_size=args.patch_size,
            make_seamless=args.seamless,
            enlarge_ratio=args.enlarge
        )
        
        if result is not None:
            print("\n✅ Texture extraction complete!")
        else:
            print("\n❌ Texture extraction failed!")

if __name__ == "__main__":
    main()
