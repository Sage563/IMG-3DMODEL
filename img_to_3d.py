"""
FREE AI Image to 3D Model Converter (NO API KEY NEEDED)
Uses local depth estimation AI models that run on your computer
Supports: Single image OR folder of multiple views
"""

import numpy as np
from PIL import Image
import argparse
import sys
import os
import glob

def setup_depth_model():
    """Setup FREE depth estimation AI model (runs locally)"""
    try:
        import torch
    except ImportError:
        print("⚠️  Missing torch. Install with:")
        print("   pip install torch torchvision")
        sys.exit(1)
    
    try:
        import cv2
    except ImportError:
        print("⚠️  Missing opencv. Install with:")
        print("   pip install opencv-python")
        sys.exit(1)
    
    try:
        print("🤖 Loading FREE AI depth model (MiDaS)...")
        print("   This runs 100% locally - no internet/API needed after download\n")
        
        model_type = "DPT_Large"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform
        
        print(f"✓ AI model loaded on: {device}")
        return midas, transform, device
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check if packages are importable:")
        print("   python -c 'import torch; import cv2; print(\"OK\")'")
        print("2. Try reinstalling:")
        print("   pip uninstall opencv-python opencv-python-headless -y")
        print("   pip install opencv-python")
        sys.exit(1)

def ai_depth_estimation(image_path, model, transform, device):
    """Use AI to estimate depth from image"""
    import torch
    import cv2
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input_batch = transform(img).to(device)
    
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    return depth_map, img

def process_multiview_folder(folder_path, model, transform, device, output_path, depth_scale=0.3, resolution=200):
    """Process multiple images from different angles to create 360° 3D model"""
    
    # Find all JPG images in folder
    jpg_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) + 
                      glob.glob(os.path.join(folder_path, "*.JPG")) +
                      glob.glob(os.path.join(folder_path, "*.jpeg")))
    
    if len(jpg_files) == 0:
        print(f"❌ No JPG images found in {folder_path}")
        sys.exit(1)
    
    print(f"\n📸 Found {len(jpg_files)} images:")
    for i, f in enumerate(jpg_files):
        print(f"   {i+1}. {os.path.basename(f)}")
    
    print(f"\n🔄 Processing multi-view reconstruction...")
    
    all_vertices = []
    all_faces = []
    all_colors = []
    vertex_offset = 0
    
    # Calculate rotation angle for each view
    angle_step = 360.0 / len(jpg_files)
    
    for idx, img_path in enumerate(jpg_files):
        print(f"\n🧠 Processing view {idx+1}/{len(jpg_files)}: {os.path.basename(img_path)}")
        
        # Get depth map using AI
        depth_map, img = ai_depth_estimation(img_path, model, transform, device)
        
        # Resize
        h, w = depth_map.shape
        aspect_ratio = w / h
        
        if aspect_ratio > 1:
            new_w = resolution
            new_h = int(resolution / aspect_ratio)
        else:
            new_h = resolution
            new_w = int(resolution * aspect_ratio)
        
        from PIL import Image as PILImage
        depth_img = PILImage.fromarray((depth_map * 255).astype(np.uint8))
        depth_resized = np.array(depth_img.resize((new_w, new_h), PILImage.LANCZOS)) / 255.0
        
        color_img = PILImage.fromarray(img)
        color_resized = np.array(color_img.resize((new_w, new_h), PILImage.LANCZOS))
        
        # Calculate rotation angle for this view
        angle_rad = np.radians(idx * angle_step)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Generate vertices for this view
        view_vertices = []
        view_colors = []
        
        for y in range(new_h):
            for x in range(new_w):
                # Local coordinates
                local_x = (x / (new_w - 1)) - 0.5
                local_y = (y / (new_h - 1)) - 0.5
                local_z = depth_resized[y, x] * depth_scale
                
                # Rotate around Y axis to position this view
                world_x = local_z * sin_a + local_x * cos_a
                world_y = local_y
                world_z = local_z * cos_a - local_x * sin_a
                
                view_vertices.append((world_x, world_y, world_z))
                
                color = color_resized[y, x] / 255.0
                view_colors.append(color)
        
        # Generate faces for this view
        view_faces = []
        for y in range(new_h - 1):
            for x in range(new_w - 1):
                v1 = vertex_offset + y * new_w + x + 1
                v2 = vertex_offset + y * new_w + (x + 1) + 1
                v3 = vertex_offset + (y + 1) * new_w + (x + 1) + 1
                v4 = vertex_offset + (y + 1) * new_w + x + 1
                
                view_faces.append([v1, v2, v3])
                view_faces.append([v1, v3, v4])
        
        all_vertices.extend(view_vertices)
        all_colors.extend(view_colors)
        all_faces.extend(view_faces)
        vertex_offset += len(view_vertices)
        
        print(f"   ✓ Added {len(view_vertices)} vertices, {len(view_faces)} faces")
    
    # Write combined OBJ file
    print(f"\n💾 Saving complete 3D model...")
    with open(output_path, 'w') as f:
        f.write("# AI-Generated 3D Model from Multiple Views\n")
        f.write(f"# Created from {len(jpg_files)} images\n")
        f.write("# FREE local depth estimation\n\n")
        
        # Vertices with colors
        for i, v in enumerate(all_vertices):
            c = all_colors[i]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
        
        f.write("\n")
        
        # Faces
        for face in all_faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"\n✅ SUCCESS!")
    print(f"📦 3D Model: {output_path}")
    print(f"📊 Stats: {len(all_vertices)} vertices, {len(all_faces)} faces")
    print(f"🔄 Views combined: {len(jpg_files)}")
    print(f"\n🎬 Import to Blender:")
    print(f"   File → Import → Wavefront (.obj) → Select '{output_path}'")

def process_single_image(image_path, model, transform, device, output_path, depth_scale=0.3, resolution=200):
    """Process single image to create depth-based 3D model"""
    
    print(f"🧠 AI is analyzing image depth...")
    depth_map, img = ai_depth_estimation(image_path, model, transform, device)
    
    # Resize
    h, w = depth_map.shape
    aspect_ratio = w / h
    
    if aspect_ratio > 1:
        new_w = resolution
        new_h = int(resolution / aspect_ratio)
    else:
        new_h = resolution
        new_w = int(resolution * aspect_ratio)
    
    from PIL import Image as PILImage
    depth_img = PILImage.fromarray((depth_map * 255).astype(np.uint8))
    depth_resized = np.array(depth_img.resize((new_w, new_h), PILImage.LANCZOS)) / 255.0
    
    color_img = PILImage.fromarray(img)
    color_resized = np.array(color_img.resize((new_w, new_h), PILImage.LANCZOS))
    
    # Generate vertices
    vertices = []
    colors = []
    
    print(f"🔨 Building 3D mesh ({new_h}x{new_w} vertices)...")
    
    for y in range(new_h):
        for x in range(new_w):
            nx = (x / (new_w - 1)) - 0.5
            ny = (y / (new_h - 1)) - 0.5
            nz = depth_resized[y, x] * depth_scale
            
            vertices.append((nx, ny, nz))
            color = color_resized[y, x] / 255.0
            colors.append(color)
    
    # Generate faces
    faces = []
    for y in range(new_h - 1):
        for x in range(new_w - 1):
            v1 = y * new_w + x + 1
            v2 = y * new_w + (x + 1) + 1
            v3 = (y + 1) * new_w + (x + 1) + 1
            v4 = (y + 1) * new_w + x + 1
            
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
    
    # Add back face
    back_start = len(vertices) + 1
    for y in range(new_h):
        for x in range(new_w):
            nx = (x / (new_w - 1)) - 0.5
            ny = (y / (new_h - 1)) - 0.5
            vertices.append((nx, ny, 0.0))
            colors.append(colors[y * new_w + x])
    
    # Add side faces
    for y in range(new_h - 1):
        v1 = y * new_w + 1
        v2 = (y + 1) * new_w + 1
        v3 = back_start + (y + 1) * new_w
        v4 = back_start + y * new_w
        faces.append([v1, v2, v3])
        faces.append([v1, v3, v4])
    
    for y in range(new_h - 1):
        v1 = (y + 1) * new_w
        v2 = y * new_w
        v3 = back_start + y * new_w - 1
        v4 = back_start + (y + 1) * new_w - 1
        faces.append([v1, v2, v3])
        faces.append([v1, v3, v4])
    
    for x in range(new_w - 1):
        v1 = x + 1
        v2 = x + 2
        v3 = back_start + x + 1
        v4 = back_start + x
        faces.append([v1, v3, v2])
        faces.append([v2, v3, v4])
    
    for x in range(new_w - 1):
        v1 = (new_h - 1) * new_w + x + 1
        v2 = (new_h - 1) * new_w + x + 2
        v3 = back_start + (new_h - 1) * new_w + x + 1
        v4 = back_start + (new_h - 1) * new_w + x
        faces.append([v1, v2, v3])
        faces.append([v1, v3, v4])
    
    # Write OBJ file
    print(f"💾 Saving 3D model...")
    with open(output_path, 'w') as f:
        f.write("# AI-Generated 3D Model from Single Image\n")
        f.write("# FREE local depth estimation\n\n")
        
        for i, v in enumerate(vertices):
            c = colors[i] if i < len(colors) else [0.5, 0.5, 0.5]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
        
        f.write("\n")
        
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    # Save depth map
    depth_vis_path = output_path.replace('.obj', '_depth.png')
    depth_vis = PILImage.fromarray((depth_map * 255).astype(np.uint8))
    depth_vis.save(depth_vis_path)
    
    print(f"\n✅ SUCCESS!")
    print(f"📦 3D Model: {output_path}")
    print(f"🗺️  Depth Map: {depth_vis_path}")
    print(f"📊 Stats: {len(vertices)} vertices, {len(faces)} faces")
    print(f"\n🎬 Import to Blender:")
    print(f"   File → Import → Wavefront (.obj) → Select '{output_path}'")

def main():
    parser = argparse.ArgumentParser(
        description='Convert image(s) to 3D using FREE AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single image:
    python img_to_3d.py photo.jpg
  
  Multiple views (360° reconstruction):
    python img_to_3d.py --folder ./photos -o object_360.obj
    
  High resolution:
    python img_to_3d.py photo.jpg -r 300 -d 0.5
    
  Tips for multi-view:
    - Take 8-16 photos around the object
    - Keep even spacing (e.g., every 45° or 22.5°)
    - Name files in order: 001.jpg, 002.jpg, etc.
    - Keep same distance and height
        """
    )
    parser.add_argument('input', nargs='?', help='Input image path (for single image mode)')
    parser.add_argument('-f', '--folder', help='Folder with multiple JPG views of object')
    parser.add_argument('-o', '--output', default='model_3d.obj', help='Output .obj path')
    parser.add_argument('-d', '--depth', type=float, default=0.3, help='Depth scale (0.1-1.0)')
    parser.add_argument('-r', '--resolution', type=int, default=200, help='Mesh resolution (50-500)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input and not args.folder:
        parser.print_help()
        print("\n❌ Error: Provide either an image file or --folder with images")
        sys.exit(1)
    
    print("=" * 60)
    print("🎨 FREE AI Image to 3D Converter")
    print("=" * 60)
    
    # Setup AI model
    model, transform, device = setup_depth_model()
    
    # Process based on mode
    if args.folder:
        # Multi-view mode
        print(f"\n🔄 MODE: Multi-view 360° reconstruction")
        process_multiview_folder(args.folder, model, transform, device, 
                                args.output, args.depth, args.resolution)
    else:
        # Single image mode
        print(f"\n📸 MODE: Single image depth-based 3D")
        if not os.path.exists(args.input):
            print(f"❌ Error: Image not found: {args.input}")
            sys.exit(1)
        process_single_image(args.input, model, transform, device,
                           args.output, args.depth, args.resolution)
    
    print("\n🎉 Done! Your 3D model is ready!")

if __name__ == '__main__':
    main()