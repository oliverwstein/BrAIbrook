import argparse
import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_fibonacci_sequence(max_num: int) -> List[int]:
    sequence = [1, 2]
    while sequence[-1] < max_num:
        sequence.append(sequence[-1] + sequence[-2])
    return [n for n in sequence if n <= max_num]

def select_pages(total_pages: int) -> List[int]:
    if total_pages <= 10:
        return list(range(1, total_pages + 1))
    fib_sequence = get_fibonacci_sequence(total_pages)
    return [n for n in fib_sequence if n <= total_pages]

def extract_json_fields(text: str) -> Optional[Dict]:
    # Remove code block markers if present
    text = re.sub(r'^```(json)?\s*|\s*```\s*$', '', text, flags=re.MULTILINE)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.info("Direct JSON parsing failed, attempting field extraction")
    
    # Define robust patterns for each field
    patterns = {
        "title": r'"title"\s*:\s*"([^"]+)"',
        "shelfmark": r'"shelfmark"\s*:\s*"([^"]+)"',
        "repository": r'"repository"\s*:\s*"([^"]+)"',
        "contents_summary": r'"contents_summary"\s*:\s*"([^"]+)"',
        "historical_context": r'"historical_context"\s*:\s*"([^"]+)"',
        "origin_location": r'"origin_location"\s*:\s*"([^"]+)"'
    }
    
    # Extract coordinates
    coordinates_match = re.search(r'"coordinates"\s*:\s*\[(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\]', text)
    if coordinates_match:
        result["coordinates"] = [float(coordinates_match.group(1)), float(coordinates_match.group(2))]
    
    # Extract array fields
    array_patterns = {
        "authors": r'"authors"\s*:\s*\[(.*?)\]',
        "alternative_titles": r'"alternative_titles"\s*:\s*\[(.*?)\]',
        "languages": r'"languages"\s*:\s*\[(.*?)\]',
        "scribes": r'"scribes"\s*:\s*\[(.*?)\]',
        "themes": r'"themes"\s*:\s*\[(.*?)\]',
        "provenance": r'"provenance"\s*:\s*\[(.*?)\]',
        "reference_materials": r'"reference_materials"\s*:\s*\[(.*?)\]'
    }
    
    result = {}
    
    # Extract string fields
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result[field] = match.group(1)
    
    # Extract array fields
    for field, pattern in array_patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            items = re.findall(r'"([^"]+)"', match.group(1))
            result[field] = items
    
    # Extract date range
    date_match = re.search(r'"date_range"\s*:\s*\[(\d+)\s*,\s*(\d+)\]', text)
    if date_match:
        result["date_range"] = [int(date_match.group(1)), int(date_match.group(2))]
    
    # Extract physical description
    phys_desc_match = re.search(r'"physical_description"\s*:\s*{([^}]+)}', text)
    if phys_desc_match:
        phys_desc = {}
        phys_fields = {
            "material": r'"material"\s*:\s*"([^"]+)"',
            "dimensions": r'"dimensions"\s*:\s*"([^"]+)"',
            "condition": r'"condition"\s*:\s*"([^"]+)"',
            "script_type": r'"script_type"\s*:\s*"([^"]+)"'
        }
        
        for field, pattern in phys_fields.items():
            match = re.search(pattern, phys_desc_match.group(1))
            if match:
                phys_desc[field] = match.group(1)
        
        # Extract layout
        layout_match = re.search(r'"layout"\s*:\s*{([^}]+)}', phys_desc_match.group(1))
        if layout_match:
            layout = {}
            layout_fields = {
                "columns_per_page": r'"columns_per_page"\s*:\s*(\d+)',
                "lines_per_page": r'"lines_per_page"\s*:\s*"([^"]+)"',
                "ruling_pattern": r'"ruling_pattern"\s*:\s*"([^"]+)"'
            }
            
            for field, pattern in layout_fields.items():
                match = re.search(pattern, layout_match.group(1))
                if match:
                    value = match.group(1)
                    layout[field] = int(value) if field == "columns_per_page" else value
            
            if layout:
                phys_desc["layout"] = layout
        
        # Extract decoration
        decoration_match = re.search(r'"decoration"\s*:\s*{([^}]+)}', phys_desc_match.group(1))
        if decoration_match:
            decoration = {}
            decoration_fields = {
                "illuminations": r'"illuminations"\s*:\s*"([^"]+)"',
                "artistic_style": r'"artistic_style"\s*:\s*"([^"]+)"'
            }
            
            for field, pattern in decoration_fields.items():
                match = re.search(pattern, decoration_match.group(1))
                if match:
                    decoration[field] = match.group(1)
            
            if decoration:
                phys_desc["decoration"] = decoration
        
        if phys_desc:
            result["physical_description"] = phys_desc
    
    # Extract technical metadata
    tech_meta_match = re.search(r'"technical_metadata"\s*:\s*{([^}]+)}', text)
    if tech_meta_match:
        tech_meta = {}
        tech_fields = {
            "image_quality": r'"image_quality"\s*:\s*"([^"]+)"',
            "special_features": r'"special_features"\s*:\s*"([^"]+)"',
            "restoration_history": r'"restoration_history"\s*:\s*"([^"]+)"'
        }
        
        for field, pattern in tech_fields.items():
            match = re.search(pattern, tech_meta_match.group(1))
            if match:
                tech_meta[field] = match.group(1)
        
        if tech_meta:
            result["technical_metadata"] = tech_meta
    
    return result if result else None

class ManuscriptCataloguer:
    def __init__(self, prompt_path: str = "prompts/metadata_prompt.txt"):
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-pro-exp")

    def process_manuscript(self, manuscript_dir: str, notes: str = "") -> Dict:
        manuscript_path = Path(manuscript_dir)
        
        with open(manuscript_path / 'metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        image_dir = manuscript_path / 'images'
        image_files = sorted([f for f in image_dir.iterdir() 
                            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}])
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        total_pages = len(image_files)
        selected_pages = select_pages(total_pages)
        selected_images = [image_files[i-1] for i in selected_pages]
        
        logger.info(f"Analyzing pages {selected_pages} from {total_pages} total")
        
        prompt = self.prompt_template.format(
            metadata=json.dumps(metadata, indent=2),
            total_pages=total_pages,
            pages_analyzed=selected_pages,
            notes=notes if notes else "None provided"
        )
        images = [Image.open(img_path) for img_path in selected_images]
        
        try:
            logger.info(self.model.count_tokens([prompt, *images]))
            response = self.model.generate_content(
                [prompt, *images],
                generation_config={"temperature": 0.9},
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Save raw response
            response_path = manuscript_path / 'standard_metadata.txt'
            with open(response_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Try standard JSON parsing first
            try:
                result = json.loads(response.text)
            except json.JSONDecodeError:
                # Fall back to regex-based extraction
                result = extract_json_fields(response.text)
                if not result:
                    logger.error("Failed to extract metadata from response")
                    return None
            
            # Save processed JSON
            json_path = manuscript_path / 'standard_metadata.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            return result
                
        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            return None
        
def main():
    parser = argparse.ArgumentParser(description='Generate standardized manuscript metadata')
    parser.add_argument('dir', help='Path to directory containing manuscript folders')
    parser.add_argument('--prompt', default='prompts/catalogue_prompt.txt', 
                       help='Path to prompt template file')
    parser.add_argument('--notes', default='', 
                       help='Additional notes or context for analysis')
    parser.add_argument('--single', default=None,
                       help='Process only this manuscript folder name')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip directories that already have standard_metadata.json')
    args = parser.parse_args()
    
    try:
        base_path = Path(args.dir)
        if not base_path.is_dir():
            raise ValueError(f"Not a directory: {base_path}")
            
        cataloguer = ManuscriptCataloguer(args.prompt)
        
        # Process single manuscript if specified
        if args.single:
            manuscript_dir = base_path / args.single
            if not manuscript_dir.is_dir():
                raise ValueError(f"Not a directory: {manuscript_dir}")
            
            metadata_file = manuscript_dir / 'standard_metadata.json'
            if args.skip_existing and metadata_file.exists():
                logger.info(f"Skipping {args.single} (standard_metadata.json exists)")
                return
                
            logger.info(f"Processing single manuscript: {args.single}")
            result = cataloguer.process_manuscript(manuscript_dir, args.notes)
            if result:
                logger.info(f"Successfully processed {args.single}")
            else:
                logger.error(f"Failed to process {args.single}")
            return
            
        # Process all manuscripts in directory
        success_count = 0
        fail_count = 0
        skip_count = 0
        
        # Get list of manuscript directories
        manuscript_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        total_manuscripts = len(manuscript_dirs)
        
        # Set up progress bar
        pbar = tqdm(manuscript_dirs, desc="Processing manuscripts", unit="ms")
        
        for manuscript_dir in pbar:
            pbar.set_description(f"Processing {manuscript_dir.name}")
            
            metadata_file = manuscript_dir / 'standard_metadata.json'
            if args.skip_existing and metadata_file.exists():
                logger.info(f"Skipping {manuscript_dir.name} (standard_metadata.json exists)")
                skip_count += 1
                continue
            
            try:
                result = cataloguer.process_manuscript(manuscript_dir, args.notes)
                if result:
                    logger.info(f"Successfully processed {manuscript_dir.name}")
                    success_count += 1
                else:
                    logger.error(f"Failed to process {manuscript_dir.name}")
                    fail_count += 1
            except Exception as e:
                logger.error(f"Error processing {manuscript_dir.name}: {e}")
                fail_count += 1
                continue
            
            # Update progress bar postfix with counts
            pbar.set_postfix({
                'success': success_count,
                'failed': fail_count,
                'skipped': skip_count
            })
        
        logger.info(f"\nBatch processing complete:")
        logger.info(f"Total manuscripts: {total_manuscripts}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {fail_count}")
        if args.skip_existing:
            logger.info(f"Skipped: {skip_count}")
                
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")

if __name__ == "__main__":
    main()