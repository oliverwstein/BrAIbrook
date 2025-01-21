import os
import argparse
import subprocess
from pathlib import Path

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def run_kraken_command(command):
    """Execute a kraken command and handle errors."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Error output: {e.stderr}")
        return False

def process_image(input_path, output_dir, model_path):
    """Process an image through Kraken workflow."""
    # Create output directory structure
    ensure_directory(output_dir)
    
    # Construct file paths
    input_filename = Path(input_path).stem
    bw_image = os.path.join(output_dir, f"{input_filename}_bw.png")
    hocr_output = os.path.join(output_dir, f"{input_filename}.hocr")
    json_output = os.path.join(output_dir, f"{input_filename}.json")
    
    # Binarize image
    binarize_cmd = [
        "kraken", 
        "-i", input_path, bw_image,
        "binarize"
    ]
    
    if not run_kraken_command(binarize_cmd):
        return False
    
    # Generate hOCR output
    hocr_cmd = [
        "kraken",
        "-h",
        "-i", bw_image, hocr_output,
        "segment", "ocr",
        "-m", model_path
    ]
    
    if not run_kraken_command(hocr_cmd):
        return False
    
    # Generate JSON output
    json_cmd = [
        "kraken",
        "-n",
        "-i", bw_image, json_output,
        "segment", "ocr",
        "-m", model_path
    ]
    
    if not run_kraken_command(json_cmd):
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Process images through Kraken OCR workflow.')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('--output-dir', default='output', help='Output directory (default: output)')
    parser.add_argument('--model', default='cremma-generic-1.0.1.mlmodel', 
                      help='Path to Kraken model (default: cremma-generic-1.0.1.mlmodel)')
    
    args = parser.parse_args()
    
    # Handle single file or directory
    input_path = Path(args.input)
    if input_path.is_file():
        # Process single file
        success = process_image(str(input_path), args.output_dir, args.model)
        if success:
            print(f"Successfully processed {input_path}")
        else:
            print(f"Failed to process {input_path}")
    elif input_path.is_dir():
        # Process all images in directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        for file in input_path.iterdir():
            if file.suffix.lower() in image_extensions:
                success = process_image(str(file), args.output_dir, args.model)
                if success:
                    print(f"Successfully processed {file}")
                else:
                    print(f"Failed to process {file}")
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()