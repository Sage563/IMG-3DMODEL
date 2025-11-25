"""
FREE AI Multi-View Generator (NO API KEY NEEDED)
Takes 1 image and generates multiple viewing angles using AI
Works with Zero123 or stable diffusion models locally
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np
import shutil

def get_cache_dir():
    """Get huggingface cache directory"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    return cache_dir

def list_cached_models():
    """List all cached models"""
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        return []
    
    models = []
    for item in os.listdir(cache_dir):
        if item.startswith("models--"):
            path = os.path.join(cache_dir, item)
            size = get_folder_size(path)
            models.append({
                'name': item.replace("models--", "").replace("--", "/"),
                'path': path,
                'size_gb': size / (1024**3)
            })
    return models

def get_folder_size(path):
    """Calculate folder size in bytes"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except Exception as e:
        pass
    return total

def cleanup_models(keep_model=None):
    """Delete all cached models except the one being used"""
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        print("✓ No cached models found")
        return 0
    
    deleted_size = 0
    deleted_count = 0
    
    print("\n🧹 Cleaning up old AI models...")
    
    for item in os.listdir(cache_dir):
        if item.startswith("models--"):
            model_name = item.replace("models--", "").replace("--", "/")
            
            # Skip if this is the model we want to keep
            if keep_model and keep_model in model_name:
                print(f"   ⏭️  Keeping: {model_name}")
                continue
            
            path = os.path.join(cache_dir, item)
            size = get_folder_size(path)
            
            try:
                shutil.rmtree(path)
                deleted_size += size
                deleted_count += 1
                print(f"   🗑️  Deleted: {model_name} ({size/(1024**3):.2f} GB)")
            except Exception as e:
                print(f"   ⚠️  Could not delete {model_name}: {e}")
    
    if deleted_count > 0:
        print(f"\n✅ Freed {deleted_size/(1024**3):.2f} GB of space!")
    else:
        print("   No models to delete")
    
    return deleted_size

def check_disk_space():
    """Check available disk space"""
    import shutil
    stat = shutil.disk_usage(os.path.expanduser("~"))
    free_gb = stat.free / (1024**3)
    return free_gb

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import torch
        import diffusers
        from diffusers import DiffusionPipeline
        return True
    except ImportError:
        print("⚠️  Missing dependencies. Install with:")
        print("   pip install torch torchvision diffusers transformers accelerate pillow")
        return False

def setup_zero123_model():
    """Setup Zero123 model for multi-view generation"""
    try:
        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        
        print("🤖 Loading Zero123 AI model (generates 3D views)...")
        print("   First run: ~4GB download, then runs offline\n")
        
        # Use Zero123 for novel view synthesis
        model_id = "ashawkey/zero123-xl-diffusers"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None
        )
        pipe.to(device)
        
        print(f"✓ AI model loaded on: {device}")
        return pipe, device
        
    except Exception as e:
        print(f"❌ Error loading Zero123: {e}")
        print("\n💡 Falling back to simple rotation method...")
        return None, None

def generate_views_with_ai(image_path, output_folder, num_views=8, pipe=None):
    """Generate multiple views using AI"""
    
    if pipe is None:
        print("⚠️  AI model not available, using geometric rotation instead")
        return generate_views_geometric(image_path, output_folder, num_views)
    
    import torch
    from PIL import Image
    
    # Load input image
    input_image = Image.open(image_path).convert("RGB")
    input_image = input_image.resize((256, 256))
    
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\n🎨 Generating {num_views} AI views...")
    
    # Generate views at different angles
    angles = np.linspace(0, 360, num_views, endpoint=False)
    
    for idx, angle in enumerate(angles):
        print(f"\n📸 Generating view {idx+1}/{num_views} (angle: {angle:.1f}°)")
        
        # Calculate elevation and azimuth for Zero123
        azimuth = angle
        elevation = 0  # Keep at eye level
        
        try:
            # Generate novel view with Zero123
            with torch.no_grad():
                output = pipe(
                    input_image,
                    polar_angle=elevation,
                    azimuth_angle=azimuth,
                    guidance_scale=3.0,
                    num_inference_steps=50,
                ).images[0]
            
            # Save generated view
            output_path = os.path.join(output_folder, f"{idx+1:03d}.jpg")
            output.save(output_path, quality=95)
            print(f"   ✓ Saved: {output_path}")
            
        except Exception as e:
            print(f"   ⚠️  Error generating view: {e}")
            continue
    
    print(f"\n✅ Generated {num_views} views in: {output_folder}")
    return output_folder

def generate_views_geometric(image_path, output_folder, num_views=8):
    """Fallback: Generate views using geometric transformations"""
    
    print("\n🔄 Using geometric rotation (fallback method)")
    print("   Note: This creates simple rotations, not true 3D views")
    
    from PIL import Image, ImageDraw, ImageFilter
    
    img = Image.open(image_path).convert("RGBA")
    os.makedirs(output_folder, exist_ok=True)
    
    # Get object bounds (remove background if possible)
    angles = np.linspace(0, 360, num_views, endpoint=False)
    
    for idx, angle in enumerate(angles):
        print(f"📸 Creating view {idx+1}/{num_views} (angle: {angle:.1f}°)")
        
        # Create perspective transformation
        rotated = img.rotate(angle * 0.3, expand=False, fillcolor=(255, 255, 255, 0))
        
        # Add slight perspective scaling
        scale_factor = 1.0 - abs(np.sin(np.radians(angle))) * 0.15
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        scaled = rotated.resize(new_size, Image.LANCZOS)
        
        # Paste onto white background
        final = Image.new("RGB", img.size, (255, 255, 255))
        offset = ((img.width - scaled.width) // 2, (img.height - scaled.height) // 2)
        final.paste(scaled, offset, scaled if scaled.mode == 'RGBA' else None)
        
        # Add slight blur for depth effect
        if abs(angle - 180) < 45:
            final = final.filter(ImageFilter.GaussianBlur(radius=1))
        
        output_path = os.path.join(output_folder, f"{idx+1:03d}.jpg")
        final.save(output_path, quality=95)
    
    print(f"\n✅ Created {num_views} geometric views in: {output_folder}")
    print("⚠️  Note: For better results, install Zero123 AI model")
    return output_folder

def generate_views_stable_diffusion(image_path, output_folder, num_views=8):
    """Alternative: Use Stable Diffusion with ControlNet for view generation"""
    
    try:
        import torch
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        from diffusers import UniPCMultistepScheduler
        
        print("🤖 Loading Stable Diffusion + ControlNet...")
        
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            torch_dtype=torch.float16
        )
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Load and process input
        img = Image.open(image_path).convert("RGB").resize((512, 512))
        
        angles = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]
        
        for idx in range(num_views):
            angle_name = angles[idx] if idx < len(angles) else f"view_{idx}"
            print(f"\n📸 Generating {angle_name} view ({idx+1}/{num_views})")
            
            prompt = f"product photo, {angle_name} view, professional photography, white background, centered"
            
            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    image=img,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                ).images[0]
            
            output_path = os.path.join(output_folder, f"{idx+1:03d}.jpg")
            output.save(output_path, quality=95)
            print(f"   ✓ Saved: {output_path}")
        
        return output_folder
        
    except Exception as e:
        print(f"⚠️  Stable Diffusion failed: {e}")
        return generate_views_geometric(image_path, output_folder, num_views)

def generate_from_text_prompt(prompt, output_folder, num_views=8, use_light=False, auto_cleanup=True):
    """Generate multi-view images from text description only"""
    
    try:
        import torch
        from diffusers import DiffusionPipeline, StableDiffusionPipeline
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Check disk space
        free_space = check_disk_space()
        print(f"💾 Available disk space: {free_space:.2f} GB\n")
        
        if use_light:
            model_name = "runwayml/stable-diffusion-v1-5"
            required_space = 4.5
            print("🤖 Loading Stable Diffusion 1.5 (Light mode - ~4GB)...")
        else:
            model_name = "stabilityai/stable-diffusion-xl-base-1.0"
            required_space = 10.5
            print("🤖 Loading Stable Diffusion XL for text-to-image...")
        
        # Auto cleanup if not enough space
        if free_space < required_space and auto_cleanup:
            print(f"⚠️  Low disk space ({free_space:.1f}GB free, need {required_space:.1f}GB)")
            print("🧹 Auto-cleaning old models to make space...\n")
            
            # Keep only the model we're about to use
            cleanup_models(keep_model=model_name)
            
            # Check space again
            free_space = check_disk_space()
            print(f"\n💾 After cleanup: {free_space:.2f} GB free\n")
            
            if free_space < required_space:
                print(f"❌ Still not enough space. Try:")
                print(f"   python {sys.argv[0]} --prompt '{prompt}' --light")
                print(f"   (Uses smaller 4GB model instead of 10GB)")
                sys.exit(1)
        
        print(f"   First run: ~{int(required_space)}GB download, then runs offline\n")
        
        if use_light:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                safety_checker=None
            )
            image_size = 512
        else:
            pipe = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None
            )
            image_size = 1024
        
        pipe.to(device)
        
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"✓ AI model loaded on: {device}\n")
        print(f"🎨 Generating {num_views} views from prompt:")
        print(f"   '{prompt}'\n")
        
        # Define angle descriptions
        angle_descriptions = [
            "front view",
            "front-right view, 45 degrees",
            "right side view, 90 degrees",
            "back-right view, 135 degrees",
            "back view, 180 degrees",
            "back-left view, 225 degrees",
            "left side view, 270 degrees",
            "front-left view, 315 degrees"
        ]
        
        # Generate views
        angles = np.linspace(0, 360, num_views, endpoint=False)
        
        for idx in range(num_views):
            angle_idx = int((idx / num_views) * 8)
            angle_desc = angle_descriptions[min(angle_idx, len(angle_descriptions)-1)]
            
            print(f"📸 Generating view {idx+1}/{num_views}: {angle_desc}")
            
            # Craft prompt for this specific view
            full_prompt = f"{prompt}, {angle_desc}, professional product photography, studio lighting, white background, highly detailed, 8k, centered composition"
            
            negative_prompt = "blurry, distorted, low quality, watermark, text, multiple objects, cluttered"
            
            try:
                with torch.no_grad():
                    image = pipe(
                        prompt=full_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        height=image_size,
                        width=image_size,
                    ).images[0]
                
                # Resize for processing
                image = image.resize((512, 512), Image.LANCZOS)
                
                # Save
                output_path = os.path.join(output_folder, f"{idx+1:03d}.jpg")
                image.save(output_path, quality=95)
                print(f"   ✓ Saved: {output_path}\n")
                
            except Exception as e:
                print(f"   ⚠️  Error: {e}\n")
                continue
        
        print(f"✅ Generated {num_views} views from text prompt!")
        print(f"📁 Saved to: {output_folder}")
        return output_folder
        
    except Exception as e:
        print(f"❌ Error with text-to-image generation: {e}")
        print("\n💡 Make sure you have installed:")
        print("   pip install diffusers transformers accelerate")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-view images using FREE AI (from photo OR text prompt)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  From existing image (8 views):
    python script.py object.jpg
  
  From text prompt (create from imagination!):
    python script.py --prompt "red sports car" -n 8
    python script.py --prompt "wooden chair, modern design" -n 16
    python script.py --prompt "ceramic coffee mug with handle" -o mug_views
  
  More views from image:
    python script.py product.jpg -n 16 -o product_views
  
  Use geometric fallback (faster, requires image):
    python script.py image.jpg --geometric

Note: First run downloads AI models (~4-7GB), then works offline forever!
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input image of object (optional if using --prompt)')
    parser.add_argument('-p', '--prompt', type=str,
                       help='Text description to generate object from scratch (no image needed!)')
    parser.add_argument('-n', '--num-views', type=int, default=8, 
                       help='Number of views to generate (default: 8)')
    parser.add_argument('-o', '--output', default='generated_views',
                       help='Output folder name (default: generated_views)')
    parser.add_argument('--geometric', action='store_true',
                       help='Use geometric rotation instead of AI (faster, requires image)')
    parser.add_argument('--method', choices=['zero123', 'stable-diffusion', 'geometric'],
                       default='zero123', help='AI method to use (for image input)')
    parser.add_argument('--light', action='store_true',
                       help='Use lighter model (4GB instead of 10GB) for text-to-image')
    parser.add_argument('--cleanup', action='store_true',
                       help='Delete old AI models before running (frees space)')
    parser.add_argument('--list-models', action='store_true',
                       help='List all cached AI models and their sizes')
    
    args = parser.parse_args()
    
    # List models mode
    if args.list_models:
        print("=" * 60)
        print("📦 Cached AI Models")
        print("=" * 60)
        models = list_cached_models()
        if not models:
            print("\nNo cached models found")
        else:
            total_size = 0
            for model in models:
                print(f"\n📁 {model['name']}")
                print(f"   Size: {model['size_gb']:.2f} GB")
                print(f"   Path: {model['path']}")
                total_size += model['size_gb']
            print(f"\n💾 Total cached: {total_size:.2f} GB")
            print(f"💾 Free space: {check_disk_space():.2f} GB")
            print(f"\nTo delete all: python {sys.argv[0]} --cleanup")
        sys.exit(0)
    
    # Cleanup mode
    if args.cleanup:
        print("=" * 60)
        print("🧹 Model Cleanup")
        print("=" * 60)
        cleanup_models()
        sys.exit(0)
    
    if not args.input and not args.prompt:
        parser.print_help()
        print("\n❌ Error: Provide either an image file OR --prompt for text-to-3D")
        sys.exit(1)
    
    if args.input and not os.path.exists(args.input):
        print(f"❌ Error: Image not found: {args.input}")
        sys.exit(1)
    
    print("=" * 60)
    print("🎨 FREE AI Multi-View Generator")
    print("=" * 60)
    
    # TEXT-TO-3D MODE: Generate from prompt only!
    if args.prompt:
        print(f"\n🌟 MODE: Text-to-3D (Create from imagination!)")
        print(f"💭 Prompt: {args.prompt}\n")
        output_folder = generate_from_text_prompt(
            args.prompt, 
            args.output, 
            args.num_views, 
            use_light=args.light,
            auto_cleanup=True
        )
    
    # IMAGE-TO-3D MODE: Generate views from existing photo
    else:
        print(f"\n📸 MODE: Image-to-3D (Generate views from photo)")
        
        # Check dependencies
        if not args.geometric and not check_dependencies():
            print("\n💡 Falling back to geometric method...")
            args.geometric = True
        
        # Generate views
        if args.geometric or args.method == 'geometric':
            output_folder = generate_views_geometric(args.input, args.output, args.num_views)
        elif args.method == 'stable-diffusion':
            output_folder = generate_views_stable_diffusion(args.input, args.output, args.num_views)
        else:  # zero123
            pipe, device = setup_zero123_model()
            output_folder = generate_views_with_ai(args.input, args.output, args.num_views, pipe)
    
    print("\n" + "=" * 60)
    print("🎉 DONE! Multi-view images generated!")
    print("=" * 60)
    print(f"\n📁 Output folder: {output_folder}")
    print(f"📸 Total views: {args.num_views}")
    print("\n🔄 Next step: Generate 3D model with:")
    print(f"   python img_to_3d.py --folder {output_folder} -o model.obj")

if __name__ == '__main__':
    main()