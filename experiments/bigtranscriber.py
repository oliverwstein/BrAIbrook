import os
import json
import subprocess
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
import google.generativeai as genai
import subprocess

def configure_gemini_api():
    """
    Configure Gemini API using environment variable for API key.

    Returns:
        (genai.GenerativeModel, genai.GenerativeModel): Configured Gemini models (text and vision)
    """
    # Retrieve API key from environment variable
    api_key = os.environ.get('GEMINI_API_KEY')

    if not api_key:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash"), genai.GenerativeModel("gemini-1.5-pro")

def create_directory(directory):
    """Creates a directory if it does not already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_kraken_command(command):
    """
    Execute a kraken command and handle errors.

    Args:
        command: A list representing the kraken command and its arguments.

    Returns:
        A tuple: (success_status, output_or_error).
        success_status: True if the command executed successfully, False otherwise.
        output_or_error: The standard output if successful, or the standard error if the command failed.
    """
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Error output: {e.stderr}")
        return False, e.stderr

def download_kraken_model(model_name):
    """
    Download a Kraken model and return its path.
    
    Args:
        model_name (str): Name or DOI of the Kraken model
    
    Returns:
        str: Path to the downloaded model file
    """
    try:
        # Use subprocess to run kraken get and capture output
        result = subprocess.run(
            ['kraken', 'get', model_name], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Parse the output to find the downloaded model path
        # This might need adjustment based on exact kraken get behavior
        model_path = result.stdout.strip().split('\n')[-1]
        
        if os.path.exists(model_path):
            return model_path
        else:
            print(f"Model download detected, but path not confirmed: {model_path}")
            return None
    
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model {model_name}: {e.stderr}")
        return None
    
def extract_text_from_hocr(hocr_file):
    """
    Extracts text content from an hOCR file.

    Args:
        hocr_file: Path to the hOCR file.

    Returns:
        A string containing the extracted text, with words separated by spaces and lines by newlines.
    """
    try:
        tree = ET.parse(hocr_file)
        root = tree.getroot()
        lines = []
        for line in root.findall(".//span[@class='ocr_line']"):
            line_text = ' '.join(word.text for word in line.findall(".//span[@class='ocrx_word']") if word.text is not None)
            lines.append(line_text)
        return '\n'.join(lines)
    except ET.ParseError as e:
        print(f"Error parsing hOCR file {hocr_file}: {e}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred while processing {hocr_file}: {e}")
        return ""
    
def process_image_with_kraken(image_path, output_dir, model_name):
    """
    Process an image through Kraken OCR workflow: binarizes the image, runs Kraken OCR,
    and generates hOCR and JSON output files.

    Args:
        image_path: Path to the input image.
        output_dir: The directory to store the output files.
        model_name: The name of the Kraken model to use.

    Returns:
        A tuple: (success, hocr_output, json_output).
        success: True if all steps were successful, False otherwise.
        hocr_output: Path to the generated hOCR file (or None if failed).
        json_output: Path to the generated JSON file (or None if failed).
    """
    # Construct file paths
    input_filename = os.path.basename(image_path)
    print(f"Processing image: {input_filename}")
    bw_image = os.path.join(output_dir, f"{os.path.splitext(input_filename)[0]}_bw.png")
    hocr_output = os.path.join(output_dir, f"{os.path.splitext(input_filename)[0]}.hocr")
    json_output = os.path.join(output_dir, f"{os.path.splitext(input_filename)[0]}.json")

    # Binarize image
    binarize_cmd = [
        "kraken",
        "-i", image_path, bw_image,
        "binarize"
    ]

    success, _ = run_kraken_command(binarize_cmd)
    if not success:
        print(f"Skipping further processing for {input_filename} due to binarization error.")
        return False, None, None

    # Download the model if it doesn't exist
    model_path = download_kraken_model(model_name)
    if not model_path:
        print(f"Failed to download model {model_name}")
        return False, None, None

    # Use the confirmed model path
    hocr_cmd = [
        "kraken",
        "-i", bw_image, hocr_output,
        "segment", "ocr",
        "-m", model_path  # Use the confirmed path
    ]
    success, _ = run_kraken_command(hocr_cmd)
    if not success:
        print(f"Skipping JSON generation for {input_filename} due to hOCR generation error.")
        return False, None, None

    # Generate JSON output
    json_cmd = [
        "kraken",
        "-i", bw_image, json_output,
        "segment", "ocr",
        "-m", model_path
    ]

    success, _ = run_kraken_command(json_cmd)
    if not success:
        print(f"Error generating JSON for {input_filename}.")
        return False, None, None

    return True, hocr_output, json_output

def get_kraken_models():
    """
    Gets a list of available Kraken models and their descriptions.

    Returns:
        A list of dictionaries, where each dictionary represents a model
        and contains "name" and "description" keys.
        Returns an empty list if an error occurs.
    """
    try:
        result = subprocess.run(['kraken', 'list'], capture_output=True, text=True, check=True)
        output = result.stdout
        model_lines = re.findall(r"(\S+)\s+\(pytorch\) - (.+)", output)

        models = []
        for name, desc in model_lines:
            models.append({
                "name": name,
                "description": desc,
            })

        return models
    except subprocess.CalledProcessError as e:
        print(f"Error getting Kraken model list: {e}")
        return []
  
def choose_kraken_model(model_text, metadata, models):
    """
    Select the most appropriate Kraken OCR model using Gemini analysis.

    Args:
        model_text (genai.GenerativeModel): Gemini text model for analysis
        metadata (dict): Manuscript metadata containing contextual information
        models (list): Available Kraken OCR models

    Returns:
        str: Name of the most suitable Kraken OCR model
    """
    # Preliminary validation
    if not models:
        return "cremma-generic-1.0.1.mlmodel"

    # Prepare metadata for analysis
    manuscript_details = {
        "Language": metadata.get('Language', 'Unknown'),
        "Date": metadata.get('Date', 'Unknown'),
        "Type": metadata.get('Type', 'Unknown')
    }

    model_descriptions = "\n".join([
        f"- {model['name']}: {model['description']}" 
        for model in models
    ])

    prompt = f"""You are an expert in medieval manuscript transcription.

    Manuscript Details:
    {json.dumps(manuscript_details, indent=2)}

    Available Kraken OCR Models:
    {model_descriptions}

    Select the most appropriate Kraken OCR model by carefully analyzing:
    1. Script and character recognition capabilities
    2. Historical time period coverage
    3. Language specificity
    4. Manuscript type compatibility

    Provide only the model name that best matches these criteria.
    """

    try:
        # Use Gemini to recommend a model
        response = model_text.generate_content(prompt)
        recommended_model = response.text.strip()

        # Validate the recommended model
        valid_models = [model['name'] for model in models]
        
        if recommended_model in valid_models:
            return recommended_model

    except Exception as e:
        print(f"Error in model selection: {e}")

    # Fallback to default model if recommendation fails
    return "cremma-generic-1.0.1.mlmodel"
    
def transcribe_manuscript(manuscript_dir, model_text):
    """
    Transcribes all pages of a manuscript using Kraken OCR.

    Generates JSON and text transcripts for each page and saves them in a
    raw_transcript.json file within the manuscript directory.

    Args:
        manuscript_dir: Path to the directory containing manuscript images and metadata.
        model_text: The Gemini text model.
    """
    manuscript_name = os.path.basename(manuscript_dir)

    # Load manuscript metadata
    metadata_path = os.path.join(manuscript_dir, "metadata.json")
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: metadata.json not found in {manuscript_dir}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {metadata_path}")
        return
    print(metadata.get("Title"))
    # Get Kraken models
    models = get_kraken_models()

    # Choose Kraken model based on metadata
    kraken_model = choose_kraken_model(model_text, metadata, models)
    print(f"Using Kraken model: {kraken_model} for manuscript: {manuscript_name}")

    # Get and sort image files
    image_files = sorted([
        f for f in os.listdir(manuscript_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if not image_files:
        print(f"No image files found in {manuscript_dir}")
        return

    transcription_data = {}
    for image_file in tqdm(image_files, desc=f"Transcribing {manuscript_name}"):
        image_path = os.path.join(manuscript_dir, image_file)
        image_base, _ = os.path.splitext(image_file)
        page_num = image_base

        # Process the image with Kraken
        success, hocr_output, json_output = process_image_with_kraken(image_path, manuscript_dir, kraken_model)
        if not success:
            print(f"Skipping transcription for {image_file} due to Kraken processing error.")
            continue

        # Extract text from hOCR
        hocr_text = extract_text_from_hocr(hocr_output)

        # Load JSON transcript
        try:
            with open(json_output, 'r') as f:
                json_transcript = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON transcript not found for {image_file}")
            json_transcript = None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_output}")
            json_transcript = None

        # Store transcription data
        transcription_data[page_num] = {
            "json_transcript": json_transcript,
            "hocr_text": hocr_text
        }

    # Save the combined transcription data ALONG WITH MODEL NAME
    raw_transcript_path = os.path.join(manuscript_dir, "raw_transcript.json")
    try:
        with open(raw_transcript_path, 'w', encoding='utf-8') as f:
            json.dump({"model": kraken_model, "transcription_data": transcription_data}, f, indent=4, ensure_ascii=False)  # Add model to output
        print(f"Raw transcripts saved to {raw_transcript_path}")
    except (IOError, OSError) as e:
        print(f"Error saving raw transcripts to {raw_transcript_path}: {e}")

def main():
    """
    Main function to transcribe all manuscripts in the data/raw directory.
    """
    raw_dir = "data/raw"

    # Get a list of manuscript directories
    manuscript_dirs = [
        os.path.join(raw_dir, d)
        for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ]

    if not manuscript_dirs:
        print("No manuscript directories found in data/raw.")
        return
    
    model_text, model_vision = configure_gemini_api()

    for manuscript_dir in manuscript_dirs:
        transcribe_manuscript(manuscript_dir, model_text)

if __name__ == "__main__":
    main()