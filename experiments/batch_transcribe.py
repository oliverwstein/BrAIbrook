import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sanitize_filename(filename):
    """
    Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename (str): Original filename to sanitize
    
    Returns:
        str: Sanitized filename
    """
    return "".join(c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in filename)

def run_kraken_command(command):
    """
    Execute a Kraken command with error handling.
    
    Args:
        command (list): Command to execute
    
    Returns:
        bool: True if command was successful, False otherwise
    """
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(command)}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running Kraken command: {e}")
        return False

def process_manuscript_image(input_path, output_dir, model_path):
    """
    Process a single image from a manuscript using Kraken OCR workflow.
    
    Args:
        input_path (str): Path to the input image
        output_dir (str): Directory to save output files
        model_path (str): Path to the Kraken OCR model
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct file paths
    input_filename = Path(input_path).stem
    # hocr_output = os.path.join(output_dir, f"{input_filename}.hocr")
    json_output = os.path.join(output_dir, f"{input_filename}.json")
    
    # Use a temporary directory for binarized image
    with tempfile.TemporaryDirectory() as temp_dir:
        bw_image = os.path.join(temp_dir, f"{input_filename}_bw.png")
        
        # Binarize image
        binarize_cmd = [
            "kraken", 
            "-i", input_path, bw_image,
            "binarize"
        ]
        
        if not run_kraken_command(binarize_cmd):
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

def transcribe_manuscript(manuscript_path, output_base_dir='data/raw_transcripts', 
                          model_path='cremma-generic-1.0.1.mlmodel'):
    """
    Transcribe all images in a manuscript folder using Kraken OCR.
    
    Args:
        manuscript_path (str): Path to the manuscript folder in data/raw/
        output_base_dir (str): Base directory for transcription outputs
        model_path (str): Path to the Kraken OCR model
    
    Returns:
        dict: Transcription metadata and results
    """
    # Read manuscript metadata
    metadata_path = os.path.join(manuscript_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        logger.error(f"No metadata found for {manuscript_path}")
        return None

    # Load manuscript metadata
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in metadata for {manuscript_path}")
        return None

    # Extract or generate a safe title for the output directory
    manuscript_title = metadata.get('Title', os.path.basename(manuscript_path))
    safe_title = sanitize_filename(manuscript_title)
    
    # Create output directory for this manuscript
    output_dir = os.path.join(output_base_dir, safe_title)
    os.makedirs(output_dir, exist_ok=True)

    # Find image files to transcribe
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = [
        f for f in sorted(os.listdir(manuscript_path)) 
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    # Transcription results tracking
    transcription_results = {
        'manuscript_title': safe_title,
        'total_pages': len(image_files),
        'processed_pages': 0,
        'failed_pages': []
    }

    # Transcribe each image
    for image_file in image_files:
        input_path = os.path.join(manuscript_path, image_file)
        
        try:
            # Process the image
            if process_manuscript_image(input_path, output_dir, model_path):
                transcription_results['processed_pages'] += 1
            else:
                transcription_results['failed_pages'].append(image_file)

        except Exception as e:
            logger.error(f"Unexpected error processing {image_file}: {e}")
            transcription_results['failed_pages'].append(image_file)

    # Save transcription results metadata
    results_metadata_path = os.path.join(output_dir, 'transcription_metadata.json')
    with open(results_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(transcription_results, f, indent=4)

    return transcription_results

def batch_transcribe_manuscripts(input_dir='data/raw', output_dir='data/raw_transcripts'):
    """
    Batch transcribe all manuscripts in the input directory.
    
    Args:
        input_dir (str): Directory containing manuscript folders
        output_dir (str): Base directory for transcription outputs
    
    Returns:
        list: List of transcription results for each manuscript
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Track overall batch results
    batch_results = []

    # Iterate through potential manuscript folders
    for item in sorted(os.listdir(input_dir)):
        manuscript_path = os.path.join(input_dir, item)
        
        # Ensure it's a directory and contains a metadata.json
        if (os.path.isdir(manuscript_path) and 
            os.path.exists(os.path.join(manuscript_path, 'metadata.json'))):
            
            logger.info(f"Transcribing manuscript: {item}")
            
            # Transcribe the manuscript
            try:
                result = transcribe_manuscript(
                    manuscript_path, 
                    output_dir
                )
                
                if result:
                    batch_results.append(result)
                else:
                    logger.warning(f"Failed to transcribe manuscript: {item}")
            
            except Exception as e:
                logger.error(f"Unexpected error transcribing {item}: {e}")

    # Save overall batch results
    batch_results_path = os.path.join(output_dir, 'batch_transcription_results.json')
    with open(batch_results_path, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=4)

    return batch_results

def main():
    # Configure logging to show all levels
    logging.getLogger().setLevel(logging.INFO)
    
    # Run batch transcription
    batch_transcribe_manuscripts()

if __name__ == "__main__":
    main()