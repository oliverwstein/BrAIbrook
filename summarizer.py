"""Generate structured summaries and table of contents for manuscript transcriptions.

This module processes manuscript transcription data to create comprehensive summaries and 
structured tables of contents. It handles manuscripts of varying lengths and types while
maintaining consistent quality and depth of analysis.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import google.generativeai as genai
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManuscriptSummarizer:
    def __init__(self, api_key: str, chunk_size: int = 200):
        """Initialize summarizer with API key and chunk size."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        self.chunk_size = chunk_size

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

    def _normalize_page_number(self, page_value: Any) -> int:
        """Convert any page number representation to an integer."""
        if isinstance(page_value, int):
            return page_value
        try:
            # Handle string representations, including possible bracketed numbers
            page_str = str(page_value)
            # Remove any non-numeric characters and convert to int
            page_num = int(''.join(c for c in page_str if c.isdigit()))
            return page_num
        except (ValueError, TypeError):
            logger.error(f"Could not normalize page number: {page_value}")
            return 0

    def _process_manuscript_chunk(self, chunk: Dict[str, Dict],
                                metadata: Dict, chunk_num: int,
                                previous_summary: Optional[str] = None) -> str:
        """Process a chunk of manuscript pages."""
        prompt = f"""Analyze this section (chunk {chunk_num}) of a manuscript. You must provide your analysis in a precise format that will be used to generate a table of contents.

        Format Requirements:
        - Respond only with analytical content that can be parsed into the required structure
        - Do not include explanatory text or meta-commentary
        - Maintain consistent formatting throughout the response
        - Use consistent section identifiers and notation

        Primary Analysis Goals:
        [Previous analysis goals section remains unchanged]

        Response Format:
        Your response must contain ONLY sections with clear labels that correspond to our JSON structure:

        Section: [Title]
        Level: [0-2]
        Page: [Number]
        Description: [Single sentence]
        Synopsis: [Detailed paragraph]
        
        You may include multiple such sections, but maintain this exact format for each.

        Previous Summary: {previous_summary if previous_summary else "None"}

        Primary Analysis Goals:
        1. Document Complete Manuscript Structure
        - Record preliminary material (title pages, tables, introductions)
        - Note navigational elements (contents lists, indices, cross-references)
        - Document end matter (colophons, tables, appendices)
        - Identify ownership marks and additions

        2. Map Internal Organization
        - Look for explicit statements of organization (e.g., "this work has seven parts...")
        - Identify formal divisions marked by rubrics, headings, or numerical markers
        - Note transitions signaled by linguistic markers or visual elements
        - Recognize when sections correspond to standard textual divisions (e.g., books of the Bible)

        2. Document Content Organization
        - Map how the text's argument or narrative develops
        - Note relationships between sections (continuation, contrast, elaboration)
        - Identify where new topics or themes are introduced
        - Track references to the text's own structure

        3. Structural Hierarchy
        The table of contents should reflect the text's own organizational principles:

        For Explicitly Structured Texts:
        - Level 0: Use the text's own major divisions
        - Level 1: Formal subdivisions within these parts
        - Level 2: Significant discussions or episodes within subdivisions

        For Implicitly Structured Texts:
        - Level 0: Major thematic or functional divisions
        - Level 1: Distinct subtopics or narrative units
        - Level 2: Notable passages or discussions

        For each table of contents entry provide:
        1. Title: Clear, descriptive heading
        2. Description: Brief orientation statement
        3. Synopsis: A detailed paragraph that captures:
           - Core content and themes
           - Notable features
           - Connection to manuscript context

        Additional Considerations:
        - Document significant changes in language, script, or presentation
        - Note marginalia and annotations
        - Record physical characteristics and variations
        - Identify connections between sections

        Previous Summary: {previous_summary if previous_summary else "None"}

        Focus on concrete observations and specific details that contribute to 
        understanding the manuscript as a whole."""

        chunk_data = {
            'metadata': metadata,
            'pages': [{
                'number': num,
                'summary': data.get('summary', ''),
                'keywords': data.get('keywords', []),
                'marginalia': data.get('marginalia', []),
                'confidence': data.get('confidence', 0),
            } for num, data in sorted(chunk.items(), key=lambda x: int(x[0]))]
        }

        return self._analyze_chunk(prompt, chunk_data, previous_summary)

    def _create_json_metadata(self, transcription_path: Path) -> Tuple[Dict, Path]:
        """Generate JSON metadata and create structured text summary."""
        with open(transcription_path, 'r', encoding='utf-8') as f:
            transcription = json.load(f)

        metadata = transcription['metadata']
        pages = transcription['pages']

        chunks = self._chunk_pages(pages)
        previous_summary = None
        chunk_summaries = []

        for i, chunk in enumerate(chunks, 1):
            summary = self._process_manuscript_chunk(chunk, metadata, i, previous_summary)
            chunk_summaries.append(summary)
            previous_summary = summary

        final_prompt = """
        Create a sophisticated scholarly analysis of this manuscript and its metadata, synthesizing the detailed 
        examinations provided. Focus on insights that would be valuable to researchers and scholars 
        familiar with medieval manuscripts and religious literature.

        Key Analysis Requirements:
        - Describe the manuscript's content clearly and comprehensively
        - Note meaningful physical and textual features
        - Explain the historical and cultural setting succinctly
        - Capture what makes this manuscript distinctive
        - Document significant aspects of its creation and use

        Summary Guidelines:
        - Write for an educated reader seeking to understand this specific manuscript
        - Focus on concrete details rather than scholarly implications
        - Explain historical context directly and naturally
        - Describe significance in terms of the manuscript's actual features and content
        - Avoid academic jargon and theoretical discussions

        Style Notes:
        - Use clear, direct language that informs rather than argues
        - Write in full sentences with natural flow
        - Focus on what is, not what might be
        - Describe features and patterns you can observe
        - Favor specific details over general claims

        Table of Contents Requirements:
        - Include material parts of the manuscript (front, title page, provenance, etc, but NOT things like blank pages.)
        - Verify that page numbers follow a logical sequence
        - Ensure major textual divisions are properly identified and ordered
        - Note significant changes in layout, hand, or decoration

        The table of contents should provide clear navigation while preserving the 
        depth of analysis from individual sections. Focus on creating a structure that
        serves both quick reference and detailed study.

        Use this JSON structure, with empty strings for missing text fields and empty 
        arrays for missing lists:

        {
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
          "table_of_contents": [
            {
              "title": "",
              "page_number": 0,
              "level": 0,
              "description": "",
              "synopsis": ""
            }
          ]
        }

        Chunk Analyses:
        {chunk_summaries}
        
        Metadata:
        {metadata}
        """

        llm_response = self._analyze_chunk(
            final_prompt,
            {'chunk_summaries': chunk_summaries, 'metadata': metadata}
        )
        json_data, markdown_summary = self._extract_json_and_create_markdown(
            llm_response, transcription_path)
        
        summary_path = transcription_path.parent / 'structured_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(markdown_summary)
            
        return json_data, summary_path

    def _validate_page_numbers(self, toc_entries: List[Dict]) -> None:
        """Validate page numbers for logical sequence and potential errors."""
        prev_page = -1
        for entry in toc_entries:
            try:
                # Convert page number to integer, handling possible string input
                page = int(str(entry.get('page_number', 0)))
                if page < prev_page and entry.get('level', 0) == 0:
                    logger.warning(
                        f"Possible page number error: {entry['title']} (page {page}) "
                        f"comes after page {prev_page}"
                    )
                prev_page = page
            except ValueError as e:
                logger.error(f"Invalid page number format in entry {entry['title']}: {e}")
                # Continue processing other entries rather than failing completely
                continue

    def _extract_json_and_create_markdown(self, llm_response: str, 
                                        transcription_path: Path) -> Tuple[Dict, str]:
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
            # Clean up response text
            cleaned_text = llm_response.strip()
            cleaned_text = re.sub(r'^```(json)?\s*', '', cleaned_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r'```\s*$', '', cleaned_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r'^[^\[{]*', '', cleaned_text)
            cleaned_text = re.sub(r',\s*([}\]])', r'\1', cleaned_text)
            cleaned_text = cleaned_text.replace('True', 'true').replace('False', 'false').replace('Null', 'null')
            
            json_data = json.loads(cleaned_text)
            
            # Clean and normalize the JSON data
            if 'table_of_contents' in json_data:
                for entry in json_data['table_of_contents']:
                    if 'page_number' in entry:
                        entry['page_number'] = self._normalize_page_number(entry['page_number'])
                
                self._validate_page_numbers(json_data['table_of_contents'])
            
            markdown_summary = self._generate_markdown_from_json(json_data)
            return json_data, markdown_summary

        except json.JSONDecodeError as e:
            error_file = transcription_path.parent / f"{transcription_path.stem}.error.txt"
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(llm_response)
            logger.error(f"JSON extraction failed, saved raw response to {error_file}: {e}")
            return template, ""

    def _generate_markdown_from_json(self, json_data: Dict) -> str:
        """Generate markdown summary from JSON data."""
        summary = json_data.get('summary', {})
        toc = json_data.get('table_of_contents', [])

        markdown = "# Document Summary\n\n"
        
        for key, value in summary.items():
            if isinstance(value, list):
                markdown += f"## {key.replace('_', ' ').title()}\n"
                markdown += ", ".join(map(str, value)) + "\n\n"
            elif isinstance(value, dict):
                markdown += f"## {key.replace('_', ' ').title()}\n"
                for sub_key, sub_value in value.items():
                    markdown += f"### {sub_key.replace('_', ' ').title()}\n"
                    markdown += f"{sub_value}\n\n"
            else:
                markdown += f"## {key.replace('_', ' ').title()}\n"
                markdown += f"{value}\n\n"

        markdown += "# Table of Contents\n\n"
        
        for entry in toc:
            level_indent = "  " * entry['level']
            markdown += f"{level_indent}* **{entry['title']}** (Page {entry['page_number']})\n"
            markdown += f"{level_indent}  {entry['description']}\n\n"
            if entry.get('synopsis'):
                markdown += f"{level_indent}  *Synopsis:* {entry['synopsis']}\n\n"

        return markdown

    def update_transcription(self, transcription_path: Path) -> None:
        """Process manuscript and update transcription file."""
        try:
            json_output, summary_path = self._create_json_metadata(transcription_path)

            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription = json.load(f)

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

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate structured summaries for manuscript transcriptions.'
    )
    parser.add_argument('transcription', type=str, 
                       help='Path to the transcription.json file.')
    parser.add_argument('--chunk-size', type=int, default=200,
                       help='Number of pages per chunk.')

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        return

    summarizer = ManuscriptSummarizer(api_key, args.chunk_size)
    transcription_path = Path(args.transcription)
    summarizer.update_transcription(transcription_path)

if __name__ == "__main__":
    main()