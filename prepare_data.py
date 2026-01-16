import os
import shutil
import subprocess
import argparse
from pathlib import Path
import random
import sys


def extract_frames_from_video(video_path, output_dir, fps=2):
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps}',
        '-qscale:v', '2',
        os.path.join(output_dir, 'frame_%04d.jpg')
    ]
    
    print(f"Extracting frames at {fps} fps...")
    subprocess.run(cmd, check=True)


def split_images(image_dir, train_ratio=0.8, val_ratio=0.1):
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    return {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }


def create_colmap_structure(output_dir, image_source, is_video=False, fps=2, run_colmap=True):
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = base_dir / "input"
    input_dir.mkdir(exist_ok=True)
    
    distorted_dir = base_dir / "distorted"
    sparse_dir = distorted_dir / "sparse"
    database_path = distorted_dir / "database.db"
    
    distorted_dir.mkdir(exist_ok=True)
    sparse_dir.mkdir(exist_ok=True)
    
    if is_video:
        frames_dir = base_dir / "extracted_frames"
        extract_frames_from_video(image_source, str(frames_dir), fps)
        source_images = frames_dir
    else:
        source_images = Path(image_source)
    
    print("Copying images...")
    for img in source_images.glob('*'):
        if img.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            shutil.copy2(img, input_dir / img.name)
    
    splits = split_images(str(input_dir))
    
    for split_name, img_list in splits.items():
        with open(base_dir / f"{split_name}_list.txt", 'w') as f:
            f.write('\n'.join(img_list))
    
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    
    if run_colmap:
        run_colmap_pipeline(input_dir, distorted_dir, database_path, sparse_dir)
    
    print(f"Dataset ready: {base_dir}")
    return base_dir


def run_colmap_pipeline(image_dir, distorted_dir, database_path, sparse_dir):
    
    print("Running feature extraction...")
    cmd_extract = [
        'colmap', 'feature_extractor',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', 'OPENCV',
        '--SiftExtraction.use_gpu', '1'
    ]
    subprocess.run(cmd_extract, check=True)
    
    print("Running feature matching...")
    cmd_match = [
        'colmap', 'exhaustive_matcher',
        '--database_path', str(database_path),
        '--SiftMatching.use_gpu', '1'
    ]
    subprocess.run(cmd_match, check=True)
    
    print("Running sparse reconstruction...")
    cmd_mapper = [
        'colmap', 'mapper',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--output_path', str(sparse_dir)
    ]
    subprocess.run(cmd_mapper, check=True)
    
    model_dir = sparse_dir / "0"
    if model_dir.exists():
        print("Converting to text format...")
        cmd_convert = [
            'colmap', 'model_converter',
            '--input_path', str(model_dir),
            '--output_path', str(model_dir),
            '--output_type', 'TXT'
        ]
        subprocess.run(cmd_convert, check=True)
    
    print("COLMAP processing complete")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for 3D Gaussian Splatting with COLMAP')
    parser.add_argument('--input', required=True, help='Path to video file or directory containing images')
    parser.add_argument('--output', required=True, help='Output directory for processed dataset')
    parser.add_argument('--type', choices=['video', 'photos'], required=True, help='Input type: video or photos')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second for video extraction (default: 2)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio (default: 0.1)')
    parser.add_argument('--skip_colmap', action='store_true', help='Skip COLMAP processing')
    parser.add_argument('--skip_dependency_check', action='store_true', help='Skip dependency checking')
    
    args = parser.parse_args()
    
    if not args.skip_dependency_check:
        if args.type == 'video':
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("ERROR: ffmpeg not found")
                print("Install: sudo apt-get install ffmpeg")
                sys.exit(1)
        
        if not args.skip_colmap:
            try:
                subprocess.run(['colmap', '-h'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("ERROR: COLMAP not found")
                print("Install: sudo apt-get install colmap")
                sys.exit(1)
    
    random.seed(42)
    
    if args.type == 'video':
        if not os.path.isfile(args.input):
            raise ValueError(f"Video file not found: {args.input}")
    else:
        if not os.path.isdir(args.input):
            raise ValueError(f"Image directory not found: {args.input}")
    
    create_colmap_structure(
        output_dir=args.output,
        image_source=args.input,
        is_video=(args.type == 'video'),
        fps=args.fps,
        run_colmap=not args.skip_colmap
    )


if __name__ == "__main__":
    main()