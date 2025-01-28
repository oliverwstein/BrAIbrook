import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
import google.generativeai as genai
from gemini_analysis import configure_gemini_api, load_ocr_files
import re
import io
import xml.etree.ElementTree as ET

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_kraken_command(command):
    """Execute a kraken command and handle errors."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True, result.stdout  # Return success status and output
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Error output: {e.stderr}")
        return False, e.stderr  # Return failure status and error output
    
def run_kraken(image_path, output_path, model):
    """Runs Kraken OCR on a single image."""
    kraken_cmd = [
        "kraken",
        "-x",
        "-i", image_path, output_path,
        "segment", "ocr",
        "-m", model,
    ]
    subprocess.run(kraken_cmd, check=True)

def get_kraken_models():
    """Gets a list of available Kraken models and their descriptions."""
    try:
        # Run kraken list and capture the output
        result = subprocess.run(['kraken', 'list'], capture_output=True, text=True, check=True)
        output = result.stdout
        # Use regex to find lines with model information
        model_lines = re.findall(r"(\S+)\s+\(pytorch\) - (.+)", output)

        # Format the models for the prompt
        models = [{"name": name, "description": desc} for name, desc in model_lines]
        return models
    except subprocess.CalledProcessError as e:
        print(f"Error getting Kraken model list: {e}")
        return []

def choose_kraken_model(model_text, metadata):
    """Uses the text model to suggest a Kraken model based on metadata."""
    models = get_kraken_models()
    model_descriptions = "\n".join([f"- {model['name']}: {model['description']}" for model in models])

    prompt = f"""You are an expert in medieval manuscripts.
    This is the metadata associated with a manuscript: {json.dumps(metadata)}
    
    Here is a list of available Kraken OCR models:
    {model_descriptions}

    Based on the metadata and the available models, which Kraken OCR model would be most suitable for transcribing this manuscript?
    Consider models specializing in different scripts, time periods, or languages.

    Return only the name of the Kraken model (e.g., 'cremma-generic-1.0.1.mlmodel').
    If unsure, or if no specific model is deemed significantly better, return 'cremma-generic-1.0.1.mlmodel' (the default).
    """
    try:
        response = model_text.generate_content(prompt)
        model_name = response.text.strip()
        print(f"Chosen model from prompt: {model_name}")
        # Basic validation/sanitization using the output of 'kraken list'
        if not any(model_name == model["name"] for model in models):
            model_name = "cremma-generic-1.0.1.mlmodel"
        return model_name
    except Exception as e:
        print(f"Error in choosing Kraken model: {e}")
        return "cremma-generic-1.0.1.mlmodel"

def should_transcribe(model_vision, image_path, metadata):
    """Uses the vision model to determine if a page should be transcribed."""
    prompt = f"""You are an expert in medieval manuscripts.
    This is the metadata associated with a manuscript: {json.dumps(metadata)}
    Analyze this image: {image_path}
    Does this page contain significant text that should be transcribed using OCR?
    Consider whether it's a title page, a blank page, an illustration, or a page with substantial text.

    Return only 'yes' or 'no'.
    """
    try:
        image = Image.open(image_path)
        response = model_vision.generate_content([prompt, image])
        decision = response.text.strip().lower()
        return decision == "yes"
    except Exception as e:
        print(f"Error in deciding whether to transcribe: {e}")
        return False  # Default to not transcribing on error

def analyze_page(model_vision, image_path, transcript_text, metadata):
    """
    Uses the vision model to analyze a single page and provide feedback on the transcript.
    """
    prompt = f"""You are an expert in medieval manuscripts.
    This is the metadata associated with a manuscript: {json.dumps(metadata)}
    Analyze this image: {image_path}
    
    Here is the OCR transcript for this page:
    {transcript_text}

    Provide the following:
    1. **Page Description:** Briefly describe the content and layout of the page.
    2. **Transcript Assessment:**
        - Identify any potential errors or areas of uncertainty in the transcript.
        - Suggest corrections or improvements to the transcript based on your visual analysis.
        - Indicate any lines or sections that should be ignored (e.g., annotations, marginalia, non-textual elements).
    3. **Line-by-line suggestions:** Taking each line as a unit, suggest what you think it should be.
    """

    try:
        image = Image.open(image_path)
        response = model_vision.generate_content([prompt, image])
        return response.text
    except Exception as e:
        print(f"Error in analyzing page: {e}")
        return "Error during page analysis."

def refine_page_content(model_text, transcript, analysis, metadata):
    """
    Uses the language model to refine the transcript and generate page-level metadata.
    """
    prompt = f"""You are an expert in medieval manuscripts.

    Here is the metadata for the entire manuscript:
    {json.dumps(metadata)}

    Here is the initial OCR transcript for a single page:
    {transcript}

    Here is an analysis of the page image and transcript, with suggestions for improvement:
    {analysis}

    Please provide the following:

    1. **Page Metadata:** A JSON object with metadata specific to this page, including:
        -   `page_number`:  (If available in the transcript or inferred)
        -   `title`: (If this page has a distinct title or heading)
        -   `script_type`: (e.g., "Gothic", "Carolingian minuscule")
        -   `language`: (e.g., "Latin", "Middle English")
        -   `date`: (Estimated date or range, if inferable)
        -   `summary`: (A brief summary of the page's content)

    2. **Revised Transcript:** The improved transcript, incorporating the feedback from the analysis and your own assessment.

    3. **Page Summary:** A more detailed summary of the page's content based on your analysis of the page.
    """

    try:
        response = model_text.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error in refining page content: {e}")
        return "Error during page content refinement."

def process_image(image_path, output_dir, model_name):
    """Process an image through Kraken OCR workflow."""
    # Create output directory structure
    create_directory(output_dir)
    
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
        return False, None, None  # Return False and None for hocr and json outputs

    # Download the model if it doesn't exist
    model_path = os.path.join(output_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found, attempting to download...")
        success, _ = run_kraken_command(["kraken", "get", model_name])
        if not success:
            print(f"Skipping OCR for {input_filename} due to model download error.")
            return False, None, None  # Return False and None for hocr and json outputs

    # Generate hOCR output
    hocr_cmd = [
        "kraken",
        "-i", bw_image, hocr_output,
        "segment", "ocr",
        "-m", model_name
    ]
    
    success, _ = run_kraken_command(hocr_cmd)
    if not success:
        print(f"Skipping JSON generation for {input_filename} due to hOCR generation error.")
        return False, None, None  # Return False and None for hocr and json outputs
    
    # Generate JSON output
    json_cmd = [
        "kraken",
        "-i", bw_image, json_output,
        "segment", "ocr",
        "-m", model_name
    ]
    
    success, _ = run_kraken_command(json_cmd)
    if not success:
        print(f"Error generating JSON for {input_filename}.")
        return False, None, None  # Return False and None for hocr and json outputs
    
    return True, hocr_output, json_output  # Return True and paths to hocr and json

def process_manuscript(manuscript_dir, processed_dir, analyzed_dir, model_text, model_vision):
    """Processes a single manuscript (all pages)."""
    manuscript_name = os.path.basename(manuscript_dir)
    manuscript_processed_dir = os.path.join(processed_dir, manuscript_name)
    manuscript_analyzed_dir = os.path.join(analyzed_dir, manuscript_name)
    create_directory(manuscript_processed_dir)
    create_directory(manuscript_analyzed_dir)

    metadata_path = os.path.join(manuscript_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Choose a Kraken model for the entire manuscript based on metadata
    kraken_model = choose_kraken_model(model_text, metadata)
    print(f"Using Kraken model: {kraken_model} for manuscript: {manuscript_name}")

    image_files = sorted([
        f for f in os.listdir(manuscript_dir)
        if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")
    ])

    manuscript_transcript_data = {}  # To store transcripts and analyses for all pages

    for image_file in tqdm(image_files, desc=f"Processing {manuscript_name}"):
        image_path = os.path.join(manuscript_dir, image_file)
        image_base, _ = os.path.splitext(image_file)
        page_num = image_base

        # Process the image with Kraken
        success, hocr_output, json_output = process_image(image_path, manuscript_processed_dir, kraken_model)
        if not success:
            print(f"Skipping analysis for {image_file} due to Kraken processing error.")
            continue

        # Proceed with analysis only if the hOCR file exists
        if os.path.exists(hocr_output):
            try:
                # Use ET to extract text from hOCR file
                tree = ET.parse(hocr_output)
                root = tree.getroot()
                page_text = ' '.join([word.text for word in root.findall(".//span[@class='ocrx_word']") if word.text])

                # Analyze the page with the vision model
                analysis_result = analyze_page(model_vision, image_path, page_text, metadata)

                # Store transcript and analysis in the dictionary
                manuscript_transcript_data[page_num] = {
                    "transcript": page_text,
                    "analysis": analysis_result
                }

            except Exception as e:
                print(f"Error during analysis: {e}")
                continue

        else:
            print(f"Skipping transcription for {image_file} based on vision model's decision.")

    # Save the transcript and analysis data to transcript.json
    transcript_json_path = os.path.join(manuscript_analyzed_dir, "transcript.json")
    with open(transcript_json_path, 'w', encoding='utf-8') as f:
        json.dump(manuscript_transcript_data, f, indent=4)

def combine_page_analyses(manuscript_analyzed_dir):
    """Combines the analysis of each page into a single manuscript analysis."""
    combined_analysis = {
        "metadata": {},
        "pages": []
    }

    # Load original metadata
    metadata_path = os.path.join(manuscript_analyzed_dir, "..", "..", "raw", os.path.basename(manuscript_analyzed_dir), "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            combined_analysis["metadata"]["original"] = json.load(f)
    else:
        print(f"Warning: Original metadata not found at {metadata_path}")

    # Load generated metadata from each page's analysis
    combined_generated_metadata = {}
    for analysis_file in os.listdir(manuscript_analyzed_dir):
        if analysis_file.endswith("_analysis.txt"):
            page_num = analysis_file.split("_")[0]  # Extract page number from filename
            with open(os.path.join(manuscript_analyzed_dir, analysis_file), 'r') as f:
                analysis_text = f.read()
                metadata_match = re.search(r"Generated Metadata:\n(.*?)(?=\n---|$)", analysis_text, re.DOTALL)
                if metadata_match:
                    metadata_str = metadata_match.group(1)
                    metadata = {}
                    for line in metadata_str.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip().lower().replace("/", "_").replace(" ", "_")] = value.strip()
                    combined_generated_metadata[page_num] = metadata
                else:
                    print(f"Warning: Metadata section not found in {analysis_file}")

    # Combine generated metadata (taking the most frequent value for each field)
    combined_analysis["metadata"]["generated"] = {}
    for key in ["title", "author", "date", "language", "script", "type", "origin", "subject_keyword_tags", "summary"]:
         values = [combined_generated_metadata[page_num][key] for page_num in combined_generated_metadata if key in combined_generated_metadata[page_num]]
         if values:
             combined_analysis["metadata"]["generated"][key] = max(set(values), key=values.count)

    # Aggregate transcripts, translations, and summaries from each page
    transcripts = []
    translations = []
    summaries = []
    for analysis_file in os.listdir(manuscript_analyzed_dir):
        if analysis_file.endswith("_analysis.txt"):
            with open(os.path.join(manuscript_analyzed_dir, analysis_file), 'r') as f:
                analysis_text = f.read()

                # Extract transcript (revised, from the final analysis section)
                transcript_match = re.search(r"1\. Revised Transcript:\n(.*?)(?=\n2\.|\n---|$)", analysis_text, re.DOTALL)
                if transcript_match:
                    transcripts.append(transcript_match.group(1).strip())

                # Extract translation (final, from the final analysis section)
                translation_match = re.search(r"4\. Final Translation:\n(.*?)(?=\n---|$)", analysis_text, re.DOTALL)
                if translation_match:
                    translations.append(translation_match.group(1).strip())

                # Extract summary (from final analysis)
                summary_match = re.search(r"Synthesis of Analyses:\n(.*?)(?=\n---|$)", analysis_text, re.DOTALL)
                if summary_match:
                    summaries.append(summary_match.group(1).strip())
    
    combined_analysis["transcript"] = "\n".join(transcripts)
    combined_analysis["translation"] = "\n".join(translations)
    combined_analysis["summary"] = "\n".join(summaries)  # You might want to combine these differently

    # Save combined analysis
    with open(os.path.join(manuscript_analyzed_dir, "combined_analysis.json"), 'w') as f:
        json.dump(combined_analysis, f, indent=4)

def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    analyzed_dir = "data/analyzed"
    create_directory(processed_dir)
    create_directory(analyzed_dir)

    model_text, model_vision = configure_gemini_api()

    # Get a list of manuscript directories
    manuscript_dirs = [
        os.path.join(raw_dir, d)
        for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ]

    # Prompt the user to choose a manuscript
    print("Available manuscripts:")
    for i, manuscript_dir in enumerate(manuscript_dirs):
        print(f"{i + 1}. {os.path.basename(manuscript_dir)}")

    while True:
        try:
            choice = int(input("Enter the number of the manuscript to process: "))
            if 1 <= choice <= len(manuscript_dirs):
                manuscript_dir = manuscript_dirs[choice - 1]
                break
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Process the selected manuscript
    print(f"Processing manuscript: {os.path.basename(manuscript_dir)}")
    process_manuscript(manuscript_dir, processed_dir, analyzed_dir, model_text, model_vision)

    # Combine the analyses for the processed manuscript
    manuscript_analyzed_dir = os.path.join(analyzed_dir, os.path.basename(manuscript_dir))
    combine_page_analyses(manuscript_analyzed_dir)

    print(f"Manuscript analysis complete for {os.path.basename(manuscript_dir)}!")

if __name__ == "__main__":
    main()