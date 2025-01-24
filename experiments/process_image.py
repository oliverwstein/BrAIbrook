import os
import argparse
import subprocess
from pathlib import Path
import re
import json

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def run_kraken_command(command):
    """Execute a kraken command and handle errors."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Error output: {e.stderr}")
        return False, ""

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

    if not run_kraken_command(binarize_cmd)[0]:
        return False

    # Generate hOCR output
    hocr_cmd = [
        "kraken",
        "-i", bw_image, hocr_output,
        "segment", "ocr",
        "-m", model_path
    ]

    if not run_kraken_command(hocr_cmd)[0]:
        return False

    # Generate JSON output
    json_cmd = [
        "kraken",
        "-i", bw_image, json_output,
        "segment", "ocr",
        "-m", model_path
    ]

    if not run_kraken_command(json_cmd)[0]:
        return False

    return True

def get_kraken_models():
    """
    Gets a list of available Kraken models and their descriptions using 'kraken list'
    and 'kraken show'.
    """
    success, output = run_kraken_command(['kraken', 'list'])
    if not success:
        return []

    model_identifiers = []
    for line in output.splitlines():
        match = re.match(r"(\S+)\s+\(pytorch\)", line)
        if match:
            model_identifiers.append(match.group(1))

    models = []
    for model_id in model_identifiers:
        success, output = run_kraken_command(['kraken', 'show', model_id])
        if success:
            try:
                # Use regex to find JSON-like structure
                match = re.search(r"(\{[\s\S]*?\})", output, re.DOTALL)
                if match:
                    metadata_str = match.group(1)
                    metadata = json.loads(metadata_str)
                    models.append({"id": model_id, "metadata": metadata})
                else:
                    print(f"Could not find JSON metadata for model: {model_id}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON metadata for model: {model_id}")
                print(f"Raw output: {output}")  # Print raw output to help debug
        else:
            print(f"Failed to retrieve metadata for model: {model_id}")

    return models

def download_model(model_id, output_dir):
    """Downloads a Kraken model."""
    print(f"Downloading model: {model_id}")
    success, _ = run_kraken_command(['kraken', 'get', model_id, "-o", output_dir])
    if success:
        print(f"Successfully downloaded model {model_id} to {output_dir}")
        return True
    else:
        print(f"Failed to download model {model_id}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Process images through Kraken OCR workflow.')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('--output-dir', default='output', help='Output directory (default: output)')
    parser.add_argument('--model-dir', default='models/kraken', help='Directory for downloaded models (default: models/kraken)')

    args = parser.parse_args()

    # Ensure the model directory exists
    ensure_directory(args.model_dir)

    # Get available models
    models = get_kraken_models()

    if not models:
        print("No Kraken models found.")
        return

    # Display available models and let the user choose one
    print("Available Kraken models:")
    for i, model in enumerate(models):
        print(f"{i + 1}. {model['id']} - {model['metadata'].get('name', 'No Name')}")

    while True:
        try:
            choice = int(input("Choose a model by its number: ")) - 1
            if 0 <= choice < len(models):
                chosen_model = models[choice]
                break
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Download the chosen model if it doesn't exist
    model_path = os.path.join(args.model_dir, chosen_model['id'] + ".mlmodel")
    if not os.path.exists(model_path):
        if not download_model(chosen_model['id'], args.model_dir):
            print(f"Failed to download model {chosen_model['id']}. Exiting.")
            return

    # Handle single file or directory
    input_path = Path(args.input)
    if input_path.is_file():
        # Process single file
        success = process_image(str(input_path), args.output_dir, model_path)
        if success:
            print(f"Successfully processed {input_path}")
        else:
            print(f"Failed to process {input_path}")
    elif input_path.is_dir():
        # Process all images in directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        for file in input_path.iterdir():
            if file.suffix.lower() in image_extensions:
                success = process_image(str(file), args.output_dir, model_path)
                if success:
                    print(f"Successfully processed {file}")
                else:
                    print(f"Failed to process {file}")
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()