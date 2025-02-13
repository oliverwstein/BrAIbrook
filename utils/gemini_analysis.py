import os
import sys
import json
import xml.etree.ElementTree as ET
import google.generativeai as genai
import logging
import re
from PIL import Image

# Suppress gRPC and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
logging.getLogger('google.auth.transport.requests').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

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

def extract_word_details_from_hocr(hocr_path):
    """
    Extract detailed word information from hOCR, including confidence.

    Args:
        hocr_path (str): Path to the hOCR XML file

    Returns:
        list: List of dictionaries with word details
    """
    tree = ET.parse(hocr_path)
    root = tree.getroot()

    word_details = []

    # Find all word elements
    words = root.findall(".//span[@class='ocrx_word']")

    for word in words:
        # Extract text and attributes
        text = word.text if word.text else ''
        title = word.get('title', '')

        # Parse confidence from title attribute
        confidence_match = re.search(r'x_conf (\d+\.\d+)', title)
        if confidence_match:
            confidence = float(confidence_match.group(1))
        else:
            confidence_match = re.search(r'x_confs ([\d\.]+)', title)
            if confidence_match:
                confidence = float(confidence_match.group(1))
            else:
                confidence = 0.0

        word_details.append({
            'text': text,
            'confidence': confidence
        })

    return word_details

def load_ocr_files(hocr_path, json_path):
    """
    Load and parse hOCR and JSON files.

    Args:
        hocr_path (str): Path to the hOCR XML file
        json_path (str): Path to the JSON file

    Returns:
        tuple: Containing parsed text, word details, and metadata
    """
    # Extract full text and word details from hOCR
    word_details = extract_word_details_from_hocr(hocr_path)
    full_text = ' '.join([word['text'] for word in word_details])

    # Load JSON file
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
    except json.JSONDecodeError:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_content = f.read()

    return full_text, word_details, json_content

def analyze_manuscript(model_text, model_vision, image_path, full_text, word_details, json_content):
    """
    Perform a comprehensive analysis of the manuscript using both text and vision models.
    """
    # --- Part 1: Text Analysis (using gemini-1.5-flash) ---
    text_analysis_prompt = f"""
    Perform a comprehensive analysis of the manuscript text and available metadata:

    Raw Text Content (by line):
    {[line for line in full_text.split('. ') if line]}

    Manuscript Metadata:
    {json.dumps(json_content, indent=2) if isinstance(json_content, dict) else json_content}

    Provide a structured analysis with the following sections:

    1. Synopsis:
    - Identify the likely type, origin, and potential historical context of the document
    - Infer the language, script, and approximate time period
    - Highlight any unique characteristics or potential significance

    2. Line-by-Line Analysis and Transcription Refinement:
    - Analyze each line of text as a coherent unit, considering the context of surrounding lines.
    - Identify potential OCR errors or ambiguities within each line.
    - Based on contextual analysis and linguistic patterns, propose corrections or alternative interpretations for each line.
    - Judiciously remove lines that are likely not part of the main text (e.g., annotations, marginalia, or spurious outputs).
    - Provide a refined transcription of each line, incorporating corrections and interpretations.
    - Clearly indicate any lines that have been removed or significantly altered.

    3. Context-Informed Transcript:
    - Create a cleaned transcript by integrating the refined line-by-line transcriptions.
    - Ensure the transcript maintains a logical flow and reflects the original document's structure where possible.
    - Keep line breaks where reasonable.
    - Explain any significant modifications or interpretations made during the refinement process.

    4. Modern English Translation:
    - Provide a clear, scholarly translation of the cleaned transcript.
    - Include brief explanatory notes for challenging or ambiguous passages.
    - Maintain the scholarly and historical context of the original text.
    - Keep line breaks where reasonable.

    5. Generated Metadata:
    - Based on your analysis, generate a set of metadata fields that describe the manuscript.
    - Use standard metadata terms where possible.
    - Include the following fields at minimum:
    - Title (if discernible from the text)
    - Author (if identifiable or inferable)
    - Date (estimated range)
    - Language
    - Script
    - Type (e.g., letter, poem, legal document)
    - Origin (place of creation, if inferable)
    - Subject/Keyword tags
    - Summary (a brief description of the content)

    Additional Instructions:
    - Be precise and scholarly in your approach.
    - Provide a rationale for any significant interpretative decisions, especially for line removal or alteration.
    - Consider potential paleographic nuances.
    - Use academic language appropriate for manuscript analysis.
    - Focus on producing a reliable and accurate representation of the original text's content and meaning.
    """
    text_response = model_text.generate_content(text_analysis_prompt)

    # --- Part 2: Image Analysis (using gemini-1.5-pro) ---
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return {
            "full_text": full_text,
            "word_details": word_details,
            "analysis": text_response.text,
            "image_analysis": "Image analysis failed: Image not found.",
            "final_analysis": "Final analysis failed: Image not found."
        }
    except Exception as e:
        print(f"Error opening image: {e}")
        return {
            "full_text": full_text,
            "word_details": word_details,
            "analysis": text_response.text,
            "image_analysis": f"Image analysis failed: {e}",
            "final_analysis": f"Final analysis failed: {e}"
        }

    image_analysis_prompt = """You are an expert paleographer. Please analyze this image of a manuscript and answer the following questions:

1. **Content and Layout:**
    *   Describe the overall layout and organization of the manuscript (e.g., number of columns, presence of margins, any special formatting).
    *   What can you infer about the general content or purpose of the manuscript based on its visual appearance?

2. **Script and Style:**
    *   Identify the type of script used (e.g., Carolingian minuscule, Gothic textura, etc.).
    *   Describe any notable features of the script, such as letterforms, ligatures, or abbreviations.
    *   Are there any indications of the manuscript's potential date or origin based on the script and style?

3. **Condition and Features:**
    *   Describe the overall condition of the manuscript page. Are there any visible signs of damage, wear, or aging?
    *   Are there any annotations, corrections, or other markings visible in the margins or between lines? If so, describe them.
    *   Are there any other notable features, such as illuminations, decorated initials, or rubrications?

4. **Comparison with Transcript:**
    *   Compare your observations with the following transcript of the text extracted using OCR:
    """

    image_analysis_prompt += text_response.text + """
    *   Based on your visual analysis, do you see any potential errors or areas of uncertainty in the provided transcript?
    *   Are there any discrepancies between the visual features of the manuscript and the information presented in the transcript?
    *   Do you have any suggestions for improving the accuracy or completeness of the transcript based on your visual assessment?

5. **Additional Insights:**
    *   Provide any other insights or observations about the manuscript that you deem relevant from a paleographic perspective.
    *   If possible, suggest any further steps that could be taken to analyze or understand the manuscript better.

Please provide a detailed and scholarly analysis, using appropriate paleographic terminology."""

    # Generate image analysis
    image_response = model_vision.generate_content(
        [image_analysis_prompt, image],
        safety_settings={
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        },
    )
    image_analysis = image_response.text

    # --- Part 3: Final Text Analysis (using gemini-1.5-flash) ---
    final_analysis_prompt = f"""
    You have performed two analyses of a manuscript:

    **1. Initial Text Analysis:**
    {text_response.text}

    **2. Image Analysis:**
    {image_analysis}

    Now, integrate these two analyses to produce a final, refined analysis of the manuscript. Consider the following:

    - **Reconcile Discrepancies:** Address any discrepancies or uncertainties identified between the initial text analysis (based on the OCR transcript) and the visual image analysis.
    - **Refine the Transcript:** Based on the insights from the image analysis, make any necessary corrections or improvements to the transcript provided in the initial text analysis.
    - **Update Metadata:** If the image analysis provided any new information about the manuscript's date, origin, script, or other characteristics, update the generated metadata accordingly.
    - **Enhance Overall Analysis:** Use the combined insights from both analyses to provide a richer, more nuanced understanding of the manuscript's content, context, and significance.

    Provide the following in your final analysis:

    1. **Revised Transcript:** The corrected and improved transcript, incorporating insights from the image analysis.
    2. **Updated Metadata:** Any revisions to the metadata based on the image analysis.
    3. **Synthesis of Analyses:** A comprehensive discussion that integrates the findings of both the text and image analyses, highlighting any significant revisions or new insights.
    4. **Final Translation:** An updated translation based on the revised transcript.
    """

    final_response = model_text.generate_content(final_analysis_prompt)

    return {
        "full_text": full_text,
        "word_details": word_details,
        "analysis": text_response.text,
        "image_analysis": image_analysis,
        "final_analysis": final_response.text
    }

def save_analysis_results(analysis, output_path='gemini_analysis.txt', print_output=False):
    """
    Save analysis results, including image analysis and final analysis.
    """
    # Prepare content, including image analysis and final analysis
    output_content = f"""Manuscript Analysis Report

Full Original Text:
{analysis['full_text']}

--- Comprehensive Text Analysis ---
{analysis['analysis']}

--- Image Analysis ---
{analysis['image_analysis']}

--- Final Analysis ---
{analysis['final_analysis']}
"""

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_content)

    print(f"Analysis saved to {output_path}")

    # Print to terminal if requested
    if print_output:
        print("\n--- Full Analysis ---")
        print(output_content)

def main(hocr_path, json_path, image_path, print_output=False):
    """
    Main function to orchestrate manuscript analysis.
    """
    try:
        # Suppress annoying warnings
        import warnings
        warnings.filterwarnings('ignore')

        # Load files
        full_text, word_details, json_content = load_ocr_files(hocr_path, json_path)

        # Configure Gemini models
        model_text, model_vision = configure_gemini_api()

        # Analyze manuscript
        analysis = analyze_manuscript(model_text, model_vision, image_path, full_text, word_details, json_content)

        # Save results
        save_analysis_results(analysis, print_output=print_output)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze manuscript using Gemini.')
    parser.add_argument('hocr_file', help='Path to hOCR file')
    parser.add_argument('json_file', help='Path to JSON file')
    parser.add_argument('image_file', help='Path to the image file of the manuscript')
    parser.add_argument('-p', '--print', action='store_true',
                        help='Print analysis to terminal')

    # Parse arguments
    args = parser.parse_args()

    # Run main analysis
    main(args.hocr_file, args.json_file, args.image_file, args.print)