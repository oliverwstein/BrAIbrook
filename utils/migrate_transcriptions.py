import os
import json
import logging
from pathlib import Path
from gemini_transcribe import create_safe_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def migrate_transcriptions(old_transcripts_dir: str = "data/old_transcripts", 
                          new_transcripts_dir: str = "data/transcripts"):
    """Migrates transcriptions from a list-based format to a dictionary-based format."""
    
    old_transcripts_path = Path(old_transcripts_dir).absolute()
    new_transcripts_path = Path(new_transcripts_dir).absolute()
    
    if not old_transcripts_path.exists() or not old_transcripts_path.is_dir():
        logger.error(f"Old transcripts directory not found: {old_transcripts_path}")
        return
    
    logger.info(f"Starting migration from {old_transcripts_path} to {new_transcripts_path}")
    
    for old_manuscript_dir in old_transcripts_path.iterdir():
        if not old_manuscript_dir.is_dir():
            continue
            
        try:
            # Get manuscript title from directory
            manuscript_title = old_manuscript_dir.name
            
            # Get a safe title
            safe_title = create_safe_filename(manuscript_title)
            
            # Create the new folder for the transcript
            new_manuscript_dir = new_transcripts_path / safe_title
            os.makedirs(new_manuscript_dir, exist_ok = True)
           
            # Check for the old transcript file
            old_transcript_file = old_manuscript_dir / 'transcription.json'

            if old_transcript_file.exists():
                try:
                    # Load the old transcript data
                    with open(old_transcript_file, 'r', encoding='utf-8') as f:
                        old_data = json.load(f)

                    # Transform pages from list to dictionary keyed by page number
                    new_pages = {}
                    if isinstance(old_data.get('pages'), list):
                        for page in old_data.get('pages', []):
                          if page and 'page_number' in page:
                             new_pages[str(page['page_number'])] = page
                        
                        # Create the new transcript data
                    new_data = {
                        **old_data,  # Copy over existing data
                        'pages': new_pages  # Replace 'pages' with new dictionary
                    }
                    
                    # Write to the new transcript file
                    new_transcript_file = new_manuscript_dir / 'transcription.json'
                    with open(new_transcript_file, 'w', encoding='utf-8') as f:
                        json.dump(new_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Migrated {manuscript_title} transcription file successfully")
                
                except Exception as e:
                  logger.error(f"Error migrating {manuscript_title}: {e}")

            else:
                logger.warning(f"Transcription file not found for {manuscript_title}, skipping")

        except Exception as e:
            logger.error(f"Error processing manuscript directory {old_manuscript_dir}: {e}")
    
    logger.info("Transcription migration completed.")
    
if __name__ == '__main__':
    migrate_transcriptions()