"""Generate structured summaries and table of contents for manuscript transcriptions.

This module processes manuscript transcription data in two phases:
1. Generate a structured text summary and table of contents
2. Convert the structured text into JSON format for the transcription file
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import google.generativeai as genai
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CHUNK_PROMPT_PATH = Path("prompts/chunk_summary_prompt.txt")
FINAL_PROMPT_PATH = Path("prompts/final_summary_prompt.txt")

class ManuscriptSummarizer:
    def __init__(self, api_key: str, chunk_size: int = 20):
        """Initialize summarizer with API key and chunk size."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        self.chunk_size = chunk_size
        self.chunk_prompt = self._load_prompt(CHUNK_PROMPT_PATH)
        self.final_prompt = self._load_prompt(FINAL_PROMPT_PATH)

    def _load_prompt(self, path: Path) -> str:
        """Loads a prompt from a text file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {path}")
            raise  # Re-raise the FileNotFoundError
        
    def _chunk_pages(self, pages: Dict) -> List[Dict]:
        """Split pages into manageable chunks."""
        sorted_pages = sorted(pages.items(), key=lambda x: int(x[0]))
        chunks = []
        current_chunk = {}

        for num, data in sorted_pages:
            current_chunk[num] = data
            if len(current_chunk) >= self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = {}

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
    
    def _analyze_chunk(self, prompt: str, chunk_data: Dict,
                      previous_summary: Optional[str] = None) -> str:
        """Analyze a chunk of manuscript pages."""
        try:
            context = {
                'chunk_data': chunk_data,
                'previous_summary': previous_summary
            }
            response = self.model.generate_content([prompt, json.dumps(context)])
            return response.text
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return ""

    def _process_manuscript_chunk(self, chunk: Dict[str, Dict],
                                metadata: Dict, chunk_num: int,
                                previous_summary: Optional[str] = None) -> str:
        """Process a chunk of manuscript pages."""

        prompt = self.chunk_prompt.format(chunk_num=chunk_num, previous_summary=previous_summary if previous_summary else "None")


        chunk_data = {
            'metadata': metadata,
            'pages': [{
                'number': num,
                'summary': data.get('summary', ''),
                'keywords': data.get('keywords', ''),
                'marginalia': data.get('marginalia', []),
                'confidence': data.get('confidence', ''),
            } for num, data in sorted(chunk.items(), key=lambda x: int(x[0]))]
        }

        return self._analyze_chunk(prompt, chunk_data, previous_summary)

    def _create_json_metadata(self, transcription_path: Path) -> Tuple[Dict, Path]:
        """Generate JSON metadata and create structured text summary from it."""

        with open(transcription_path, 'r', encoding='utf-8') as f:
            transcription = json.load(f)

        metadata = transcription['metadata']
        pages = transcription['pages']

        # Process pages in chunks
        chunks = self._chunk_pages(pages)
        previous_analysis = None # More descriptive variable name
        chunk_summaries = []

        for i, chunk in enumerate(chunks, 1):
            summary = self._process_manuscript_chunk(chunk, metadata, i, previous_analysis)
            chunk_summaries.append(summary)
            previous_analysis = summary  # Update previous_analysis

        # Prepare data for the final prompt
        manuscript_content = json.dumps([  # Prepare manuscript_content as JSON string
            {
                'number': num,
                'revised_transcription': data.get('revised_transcription', ''),
                'summary': data.get('summary', ''),
                'keywords': data.get('keywords', []),
                'marginalia': data.get('marginalia', []),
                'page_number': data.get('page_number', num),
            } for num, data in sorted(pages.items(), key=lambda x: int(x[0]))
        ], indent=2)
        metadata_json = json.dumps(metadata, indent=2) # Prepare metadata as JSON string

        #  Use the loaded final prompt template
        final_prompt = self.final_prompt.format(
            chunk_summaries=chunk_summaries,
            metadata=metadata_json,  # Insert JSON string of metadata
            manuscript_content=manuscript_content  # Insert JSON string of manuscript content
        )

        llm_response = self._analyze_chunk(
            final_prompt,
            {'chunk_summaries': chunk_summaries, 'metadata': metadata}
        )
        json_data, markdown_summary = self._extract_json_and_create_markdown(
            llm_response, transcription_path)
        summary_path = transcription_path.parent / 'structured_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(markdown_summary)  # Write the markdown summary to file
        return json_data, summary_path

    def _extract_json_and_create_markdown(self, llm_response: str, transcription_path: Path) -> Tuple[Dict, str]:
        """Extract JSON and generate Markdown summary."""
        template = {
            "summary": {
                "title": "",
                "alternative_titles": [],
                "shelfmark": "",
                "repository": "",
                "date_range": [0, 0],
                "languages": [],
                "scribes": [],
                "physical_description": {
                    "material": "",
                    "dimensions": "",
                    "condition": "",
                    "layout": "",
                    "script_type": "",
                    "decoration": ""
                },
                "contents_summary": "",
                "historical_context": "",
                "significance": "",
                "themes": [],
                "provenance": []
            },
            "table_of_contents": []
        }
        try:
            # Clean up common issues
            cleaned_text = llm_response.strip()

            # Remove markdown code block markers
            cleaned_text = re.sub(r'^```(json)?\s*', '', cleaned_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r'```\s*$', '', cleaned_text, flags=re.MULTILINE)

            #Remove everything before opening bracket for json
            cleaned_text = re.sub(r'^[^\[{]*', '', cleaned_text)
            cleaned_text = re.sub(r',\s*([}\]])', r'\1', cleaned_text)
            cleaned_text = cleaned_text.replace('True', 'true').replace('False', 'false').replace('Null', 'null')
            logger.debug(f"Cleaned JSON Output: {cleaned_text}")

            json_data = json.loads(cleaned_text)
            markdown_summary = self._generate_markdown_from_json(json_data)
            return json_data, markdown_summary

        except json.JSONDecodeError as e:
            error_file = transcription_path.parent / f"{transcription_path.stem}.error.txt"
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(llm_response)  # Write the raw LLM response for inspection
            logger.error(f"JSON extraction failed, saved raw response to {error_file}: {e}")
            return template, ""

    def _generate_markdown_from_json(self, json_data: Dict) -> str:
        """Generate markdown summary from JSON data."""

        summary = json_data.get('summary', {})
        toc = json_data.get('table_of_contents', [])

        markdown = "**Document Summary:**\n\n"
        for key, value in summary.items():
            if isinstance(value, list):  # Check for lists FIRST
                markdown += f"* **{key.replace('_', ' ').title()}:** {', '.join(map(str, value))}\n"
            elif isinstance(value, dict):  # Handle nested dictionaries
                markdown += f"* **{key.replace('_', ' ').title()}:**\n"
                for sub_key, sub_value in value.items():
                    markdown += f"    * **{sub_key.replace('_', ' ').title()}:** {sub_value}\n"
            elif key == "date_range":  # Special handling AFTER isinstance checks
                if isinstance(value, list) and len(value) == 2:  # Check if it's a valid date range
                    start, end = value
                    markdown += f"* **Date Range:** {start}-{end}\n"  # Format date range
                else:
                    markdown += f"* **Date Range:** {value}\n"  # Handle unexpected date_range format
            else:
                markdown += f"* **{key.replace('_', ' ').title()}:** {value}\n"


        markdown += "\n**Table of Contents:**\n\n"
        for entry in toc:
            for key, value in entry.items():
                markdown += f"* **{key.replace('_', ' ').title()}:** {value}\n"
            markdown += "\n"  # Add extra newline between TOC entries

        return markdown

    def update_transcription(self, transcription_path: Path) -> None:
        """Process manuscript and update transcription file."""
        try:
            json_output, summary_path = self._create_json_metadata(transcription_path)

            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription = json.load(f)

            # Add top-level keys to transcription dictionary
            for key, value in json_output.items():
                transcription[key] = value

            transcription['last_updated'] = datetime.now().isoformat()
            transcription['total_pages'] = transcription.get('total_pages', 0)

            with open(transcription_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, indent=2, ensure_ascii=False)

            logger.info(f"Updated {transcription_path} with document summary and TOC")

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def extract_json(self, summary_path: Path) -> Dict:
        """Extract JSON from an existing structured summary file and update transcription."""
        return self._extract_json_from_summary(summary_path)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate JSON metadata and a markdown summary for a manuscript transcription.'
    )
    parser.add_argument('transcription', type=str, help='Path to the transcription.json file.')
    parser.add_argument('--chunk-size', type=int, default=20, help='Number of pages per chunk.')  # Keep chunk-size

    args = parser.parse_args()

    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        return

    summarizer = ManuscriptSummarizer(api_key, args.chunk_size)
    transcription_path = Path(args.transcription)

    summarizer.update_transcription(transcription_path)


if __name__ == "__main__":
    main()