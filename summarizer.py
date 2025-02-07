"""
Manuscript summarizer that integrates with manuscript_server.py and transcription_manager.py
to generate comprehensive summaries and tables of contents.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import google.generativeai as genai
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManuscriptSummarizer:
    def __init__(self, api_key: str):
        """Initialize the summarizer with API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-pro-exp")
        
        self.prompt_path = Path("prompts/summary_prompt.txt")
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {self.prompt_path}")
            
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
            
        logger.info(f"Initialized summarizer with prompt template from {self.prompt_path}")

    def _extract_json_fields(self, text: str) -> Dict:
        """Extract JSON fields using robust pattern matching."""
        # Remove code block markers if present
        text = re.sub(r'^```(json)?\s*|\s*```\s*$', '', text, flags=re.MULTILINE)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.info("Direct JSON parsing failed, attempting field extraction")
            
        summary = {}
        
        # Extract summary section with raw strings for all regex patterns
        summary_match = re.search(r'"summary"\s*:\s*{([^}]+)}', text, re.DOTALL)
        if summary_match:
            summary_text = summary_match.group(1)
            
            fields = {
                "title": r'"title"\s*:\s*"([^"]+)"',
                "shelfmark": r'"shelfmark"\s*:\s*"([^"]+)"',
                "repository": r'"repository"\s*:\s*"([^"]+)"',
                "contents_summary": r'"contents_summary"\s*:\s*"([^"]+)"',
                "historical_context": r'"historical_context"\s*:\s*"([^"]+)"',
                "significance": r'"significance"\s*:\s*"([^"]+)"'
            }
            
            extracted_summary = {}
            for field, pattern in fields.items():
                match = re.search(pattern, summary_text)
                if match:
                    extracted_summary[field] = match.group(1)
            
            arrays = {
                "alternative_titles": r'"alternative_titles"\s*:\s*\[(.*?)\]',
                "languages": r'"languages"\s*:\s*\[(.*?)\]',
                "scribes": r'"scribes"\s*:\s*\[(.*?)\]',
                "themes": r'"themes"\s*:\s*\[(.*?)\]',
                "provenance": r'"provenance"\s*:\s*\[(.*?)\]'
            }
            
            for field, pattern in arrays.items():
                match = re.search(pattern, summary_text)
                if match:
                    items = re.findall(r'"([^"]+)"', match.group(1))
                    extracted_summary[field] = items
            
            date_match = re.search(r'"date_range"\s*:\s*\[(\d+)\s*,\s*(\d+)\]', summary_text)
            if date_match:
                extracted_summary["date_range"] = [
                    int(date_match.group(1)),
                    int(date_match.group(2))
                ]
            
            phys_desc_match = re.search(
                r'"physical_description"\s*:\s*{([^}]+)}',
                summary_text
            )
            if phys_desc_match:
                phys_desc = {}
                phys_fields = [
                    "material", "dimensions", "condition",
                    "layout", "script_type", "decoration"
                ]
                for field in phys_fields:
                    match = re.search(
                        rf'"{field}"\s*:\s*"([^"]+)"',
                        phys_desc_match.group(1)
                    )
                    if match:
                        phys_desc[field] = match.group(1)
                extracted_summary["physical_description"] = phys_desc
            
            summary["summary"] = extracted_summary
        
        # Extract table of contents
        toc_match = re.search(r'"table_of_contents"\s*:\s*\[(.*?)\](?=\s*})', text, re.DOTALL)
        if toc_match:
            toc_entries = []
            entries = re.finditer(
                r'{[^}]*"title":\s*"([^"]+)"[^}]*"page_number":\s*(\d+)[^}]*"level":\s*(\d+)[^}]*"description":\s*"([^"]+)"[^}]*"synopsis":\s*"([^"]+)"[^}]*}',
                toc_match.group(1),
                re.DOTALL
            )
            
            for entry in entries:
                toc_entries.append({
                    "title": entry.group(1),
                    "page_number": int(entry.group(2)),
                    "level": int(entry.group(3)),
                    "description": entry.group(4),
                    "synopsis": entry.group(5)
                })
            
            summary["table_of_contents"] = sorted(
                toc_entries,
                key=lambda x: x["page_number"]
            )
        
        return summary

    def update_transcription(self, transcription_path: Path) -> None:
        """Update the transcription file with generated summary."""
        try:
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription = json.load(f)
            if not transcription.get('pages'):
                raise ValueError("contents must not be empty")
            metadata = transcription['metadata']
            pages = {}
            
            for page_num, page_data in sorted(transcription['pages'].items(), key=lambda x: int(x[0])):
                pages[page_num] = {
                    'page_number': int(page_num),
                    'revised_transcription': page_data.get('revised_transcription', '')
                }
            # Format the prompt and generate content
            prompt = self.prompt_template.format(
                metadata=json.dumps(metadata, indent=2),
                manuscript_content=json.dumps(pages, indent=2)
            )
            # logger.info(f"Prompt: {prompt}")
            logger.info(f"Prompt tokens: {self.model.count_tokens(prompt)}")
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.9,
                    "top_p": 0.8,
                    "top_k": 40
                },
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            )
            logger.info("LLM Response Text:", response.text)

            result = self._extract_json_fields(response.text)
            if not result:
                raise ValueError("Failed to generate valid analysis")

            # Update transcription
            transcription.update(result)
            transcription['last_updated'] = datetime.now().isoformat()

            with open(transcription_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, indent=2, ensure_ascii=False)

        except ValueError as e:
            logger.error(f"Failed to process manuscript: {e}")
            raise
        except Exception as e:
            logger.error(f"Error updating transcription: {e}")
            raise