#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path
import json
from PIL import Image
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_manuscript(manuscript_dir: Path, output_root: Path):
    """Processes a single manuscript and creates the output structure."""
    manuscript_id = manuscript_dir.name
    logger.info(f"Processing manuscript: {manuscript_id}")

    output_dir = output_root / manuscript_id
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    thumbs_dir = images_dir / "thumbs"
    web_dir = images_dir / "web"
    pages_dir = output_dir / "pages"

    images_dir.mkdir(exist_ok=True)
    thumbs_dir.mkdir(exist_ok=True)
    web_dir.mkdir(exist_ok=True)
    pages_dir.mkdir(exist_ok=True)


    # 1. Process Images (Create WebPs)
    for jpg_path in (manuscript_dir / "images").glob("*.jpg"):
        try:
            base_name = jpg_path.stem
            page_number = base_name.replace('page-', '').zfill(4)  # Consistent naming

            thumb_path = thumbs_dir / f"page-{page_number}.webp"
            web_path = web_dir / f"page-{page_number}.webp"

            create_webp(jpg_path, thumb_path, 300, 85)
            create_webp(jpg_path, web_path, 1600, 85)

        except OSError as e:
            logger.error(f"  Error processing image {jpg_path}: {e}")
            # Consider whether to continue or raise the exception

    # 2. Copy standard_metadata.json
    standard_metadata_src = manuscript_dir / "standard_metadata.json"
    standard_metadata_dest = output_dir / "standard_metadata.json"
    if standard_metadata_src.exists():
        shutil.copy2(standard_metadata_src, standard_metadata_dest)
        logger.info(f"  Copied standard_metadata.json for {manuscript_id}")
    else:
        logger.warning(f"  standard_metadata.json not found for {manuscript_id}")

    # 3. Process and Create Page Transcripts
    transcript_src = manuscript_dir / "transcript.json"
    if transcript_src.exists():
        try:
            with open(transcript_src, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)

            last_updated = transcript_data.get('last_updated', datetime.now().isoformat()) # Not used, but extracted
            pages_data = transcript_data.get('pages', {})

            for page_num, page_data in pages_data.items():
                page_number = str(page_num).zfill(4)

                # --- Create the page-specific directory ---
                page_dir = pages_dir / page_number
                page_dir.mkdir(exist_ok=True)

                page_doc = {
                    'transcript': {
                        'body': page_data.get('body', []),
                        'illustrations': page_data.get('illustrations', []),
                        'marginalia': page_data.get('marginalia', []),
                        'notes': page_data.get('notes', []),
                        'language': page_data.get('language', ''),
                        'transcription_notes': page_data.get('transcription_notes', '')
                    },
                    'last_updated': last_updated
                }

                # Save transcript.json *within* the page directory
                page_transcript_dest = page_dir / "transcript.json"
                with open(page_transcript_dest, 'w', encoding='utf-8') as outfile:
                    json.dump(page_doc, outfile, ensure_ascii=False, indent=2)
            logger.info(f"  Processed transcript.json for {manuscript_id}")

        except Exception as e:
            logger.error(f"  Error processing transcript for {manuscript_id}: {e}")
            # Consider whether to continue or raise the exception
    else:
        logger.warning(f"  transcript.json not found for {manuscript_id}")

        # 4. Create transcription status
    if transcript_src.exists(): # Only if it exists
        status_doc = {
            'total_pages': transcript_data.get('total_pages', 0),
            'transcribed_pages': len(transcript_data.get('pages', {})),
            'failed_pages': transcript_data.get('failed_pages', []),
            'last_updated': transcript_data.get('last_updated', datetime.now().isoformat()) #Not used
        }
        status_path = output_dir / "transcription_status.json"
        with open(status_path, 'w', encoding='utf-8') as f:
            json.dump(status_doc, f, ensure_ascii=False, indent=2)



def create_webp(input_path: Path, output_path: Path, width: int, quality: int):
    """Creates a WebP version of the image, handling errors."""
    try:
        with Image.open(input_path) as img:
            ratio = width / img.width
            height = int(img.height * ratio)
            resized = img.resize((width, height), Image.Resampling.LANCZOS)

            if resized.mode in ('RGBA', 'P'):
                resized = resized.convert('RGB')
            resized.save(output_path, format='WEBP', quality=quality, method=6) #Best lossless
    except OSError as e:
        logger.error(f"Error processing image {input_path}: {e}")
        #  Don't re-raise, just log.  Continue processing other images.
    except Exception as e:
        logger.error(f"Error with image {input_path}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare manuscript data for upload.')
    parser.add_argument('catalogue_dir', help='Path to the catalogue directory')
    parser.add_argument('output_dir', help='Path to the output directory (processed_catalogue)')
    args = parser.parse_args()

    catalogue_path = Path(args.catalogue_dir)
    output_path = Path(args.output_dir)

    if not catalogue_path.exists():
        logger.error(f"Catalogue directory not found: {catalogue_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    manuscript_dirs = [d for d in catalogue_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(manuscript_dirs)} manuscript directories to process")

    for manuscript_dir in manuscript_dirs:
        process_manuscript(manuscript_dir, output_path)

    logger.info("All manuscripts processed locally. Ready for upload.")

if __name__ == '__main__':
    main()