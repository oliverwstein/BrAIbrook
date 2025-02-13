import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from urllib.parse import quote
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_transcripts(transcripts_dir: str = "data/transcripts") -> Tuple[int, Dict[str, List[int]], int]:
    """
    Analyze transcript files to find pages with JSON parsing failures.
    """
    failed_pages = defaultdict(list)
    total_failed = 0
    total_pages = 0
    
    transcripts_path = Path(transcripts_dir)
    
    for manuscript_dir in transcripts_path.iterdir():
        if not manuscript_dir.is_dir():
            continue
            
        transcript_file = manuscript_dir / 'transcription.json'
        if not transcript_file.exists():
            continue
            
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
                
            manuscript_title = transcript_data.get('manuscript_title', manuscript_dir.name)
            pages = transcript_data.get('pages', {})
            
            total_pages += len(pages)
            
            for page_num, page_data in pages.items():
                if isinstance(page_data, dict):
                    notes = page_data.get('transcription_notes', '')
                    if notes == "Failed to parse JSON response":
                        failed_pages[manuscript_title].append(int(page_num))
                        total_failed += 1
                        
        except Exception as e:
            logger.error(f"Error processing {transcript_file}: {e}")
            
    return total_failed, dict(failed_pages), total_pages

def retranscribe_page(manuscript_title: str, page_number: int, api_url: str) -> bool:
    """Retranscribe a single page using the API."""
    try:
        encoded_title = quote(manuscript_title)
        url = f"{api_url}/manuscripts/{encoded_title}/pages/{page_number}/transcribe"
        
        response = requests.post(url)
        
        if response.status_code == 200:
            logger.info(f"Successfully retranscribed page {page_number}")
            return True
        else:
            logger.error(f"Failed to retranscribe page {page_number}: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error retranscribing page {page_number}: {str(e)}")
        return False

def repair_manuscript(manuscript_title: str, failed_pages: List[int], api_url: str = "http://127.0.0.1:5000") -> None:
    """Repair all failed pages in a manuscript."""
    logger.info(f"\nRepairing manuscript: {manuscript_title}")
    logger.info(f"Pages to repair: {sorted(failed_pages)}")
    
    for page_num in sorted(failed_pages):
        logger.info(f"Retranscribing page {page_num}")
        success = retranscribe_page(manuscript_title, page_num, api_url)
        
        if success:
            time.sleep(2)  # Add delay between requests
        else:
            logger.warning(f"Skipping remaining pages for {manuscript_title} due to error")
            break

def main():
    # Run analysis
    total_failed, failures_by_manuscript, total_pages = analyze_transcripts()
    
    # Print results
    print("\nTranscription Analysis Results")
    print("=" * 40)
    print(f"Total pages analyzed: {total_pages}")
    print(f"Total pages with JSON parsing failures: {total_failed}")
    print(f"Overall failure rate: {(total_failed/total_pages)*100:.2f}%")
    print("\nBreakdown by manuscript:")
    print("-" * 40)
    
    # Sort manuscripts by number of failures
    sorted_manuscripts = sorted(
        failures_by_manuscript.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    # Process each manuscript
    for manuscript, failed_pages in sorted_manuscripts:
        if failed_pages:  # Only show manuscripts with failures
            print(f"\n{manuscript}")
            print(f"Failed pages: {sorted(failed_pages)}")
            print(f"Total failures: {len(failed_pages)}")
            
            # Ask if user wants to repair this manuscript
            while True:
                response = input(f"\nDo you want to repair {len(failed_pages)} failed pages in this manuscript? (y/n): ").lower()
                if response in ['y', 'n']:
                    break
                print("Please enter 'y' or 'n'")
            
            if response == 'y':
                repair_manuscript(manuscript, failed_pages)
                print(f"Completed repair attempt for {manuscript}")
            else:
                print(f"Skipping repair for {manuscript}")

if __name__ == "__main__":
    main()