#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import subprocess
from typing import Dict, List, Optional
from google.cloud import storage
import logging
from PIL import Image
import io
import os
from datetime import datetime
import shutil
import aiofiles

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ManuscriptProcessor:
    def __init__(self, bucket_name: str = "ley-star", cdn_domain: Optional[str] = None, project_id: Optional[str] = None):
        self.bucket_name = bucket_name
        self.cdn_domain = cdn_domain or f"storage.googleapis.com/{bucket_name}"
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def process_manuscript(self, manuscript_dir: Path):
        """Process a single manuscript directory."""
        manuscript_id = manuscript_dir.name
        logger.info(f"Processing manuscript: {manuscript_id}")

        try:
            # Create temporary processing directories
            processed_dir = manuscript_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            (processed_dir / "thumbs").mkdir(exist_ok=True)
            (processed_dir / "web").mkdir(exist_ok=True)
            
            # Process images
            await self.process_images(manuscript_dir)

            # Upload standard metadata if exists
            standard_metadata_path = manuscript_dir / "standard_metadata.json"
            if standard_metadata_path.exists():
                await self.upload_file(
                    standard_metadata_path,
                    f"catalogue/{manuscript_id}/standard_metadata.json"
                )
            else:
                logger.warning(f"standard_metadata.json not found for {manuscript_id}")

            # Process and upload transcript if exists
            transcript_path = manuscript_dir / "transcript.json"
            if transcript_path.exists():
                await self.process_transcript(manuscript_id, transcript_path)
            else:
                logger.warning(f"transcript.json not found for {manuscript_id}")

            # Generate and upload transcription status
            if transcript_path.exists(): # Only if transcript exists
                await self.generate_transcription_status(manuscript_id, transcript_path)

        except Exception as e:
            logger.error(f"Error processing manuscript {manuscript_id}: {e}")
            raise

    async def process_images(self, manuscript_dir: Path):
        """Process and upload images for a manuscript."""
        manuscript_id = manuscript_dir.name # Get manuscript id
        image_tasks = []
        for jpg_path in (manuscript_dir / "images").glob("*.jpg"):
            # Construct the expected webp paths.
            base_name = jpg_path.stem
            page_number = base_name.replace('page-', '').zfill(4)
            thumb_dest = f"catalogue/{manuscript_id}/images/thumbs/page-{page_number}.webp"
            web_dest = f"catalogue/{manuscript_id}/images/web/page-{page_number}.webp"
            # Only add to tasks if *both* do not exist.
            if not (self.bucket.blob(thumb_dest).exists() and self.bucket.blob(web_dest).exists()):
                image_tasks.append(self.process_single_image(manuscript_dir, jpg_path))
        await asyncio.gather(*image_tasks)

    async def process_single_image(self, manuscript_dir: Path, jpg_path: Path):
        """Process a single image into thumb and web versions."""
        manuscript_id = manuscript_dir.name
        base_name = jpg_path.stem
        page_number = base_name.replace('page-', '').zfill(4)

        # Process images in parallel using ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        thumb_future = loop.run_in_executor(
            self.executor, self.create_webp, jpg_path, 300, 85
        )
        web_future = loop.run_in_executor(
            self.executor, self.create_webp, jpg_path, 1600, 85
        )

        try:  # Add a try...except block here as well
            thumb_data, web_data = await asyncio.gather(thumb_future, web_future)

            # Upload in parallel
            upload_tasks = [
                self.upload_bytes(
                    thumb_data,
                    f"catalogue/{manuscript_id}/images/thumbs/page-{page_number}.webp",
                    "image/webp"
                ),
                self.upload_bytes(
                    web_data,
                    f"catalogue/{manuscript_id}/images/web/page-{page_number}.webp",
                    "image/webp"
                )
            ]
            await asyncio.gather(*upload_tasks)
        except Exception as e:
            logger.error(f"Error processing or uploading image {jpg_path}: {e}")


    async def process_transcript(self, manuscript_id: str, transcript_path: Path):
        """Process and upload transcript data (concurrent page uploads)."""
        async with aiofiles.open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.loads(await f.read())

        last_updated = transcript_data.get('last_updated', datetime.now().isoformat())
        pages_data = transcript_data.get('pages', {})

        # Create a list of upload tasks
        upload_tasks = []
        for page_num, page_data in pages_data.items():
            page_number = str(page_num).zfill(4)

            # Create page-specific data structure
            page_doc = {
                'transcript': {
                    'body': page_data.get('body', []),
                    'illustrations': page_data.get('illustrations', []),
                    'marginalia': page_data.get('marginalia', []),
                    'notes': page_data.get('notes', []),
                    'language': page_data.get('language', ''),
                    'transcription_notes': page_data.get('transcription_notes', '')
                },
                'image_urls': {
                    'thumb': f"https://{self.cdn_domain}/catalogue/{manuscript_id}/images/thumbs/page-{page_number}.webp",
                    'web': f"https://{self.cdn_domain}/catalogue/{manuscript_id}/images/web/page-{page_number}.webp"
                },
                'last_updated': last_updated
            }

            upload_tasks.append(self.upload_json(
                page_doc,
                f"catalogue/{manuscript_id}/pages/{page_number}/transcript.json"
            ))

        await asyncio.gather(*upload_tasks) # Keep concurrent uploads

    async def generate_transcription_status(self, manuscript_id: str, transcript_path: Path):
        """Generate and upload transcription status document (using aiofiles)."""
        if not transcript_path.exists():
            return

        async with aiofiles.open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.loads(await f.read())

        status_doc = {
            'total_pages': transcript_data.get('total_pages', 0),
            'transcribed_pages': len(transcript_data.get('pages', {})),
            'failed_pages': transcript_data.get('failed_pages', []),
            'last_updated': transcript_data.get('last_updated', datetime.now().isoformat())
        }

        await self.upload_json(
            status_doc,
            f"catalogue/{manuscript_id}/transcription_status.json"
        )

    def create_webp(self, image_path: Path, width: int, quality: int) -> bytes:
        """Create a WebP version of an image with specified width, handling potential errors."""
        try:
            with Image.open(image_path) as img:
                ratio = width / img.width
                height = int(img.height * ratio)
                resized = img.resize((width, height), Image.Resampling.LANCZOS)

                if resized.mode in ('RGBA', 'P'):
                    resized = resized.convert('RGB')

                output = io.BytesIO()
                resized.save(output, format='WEBP', quality=quality, method=6)
                return output.getvalue()
        except OSError as e:  # Catch the OSError
            logger.error(f"Error processing image {image_path}: {e}")
            return b""  # Return an empty bytes object on error
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return b""

    async def upload_file(self, file_path: Path, destination: str):
        """Upload a file to Cloud Storage, checking for existence."""
        blob = self.bucket.blob(destination)
        if not blob.exists():
            blob.metadata = {'Cache-Control': 'public, max-age=86400'}
            await asyncio.to_thread(blob.upload_from_filename, str(file_path))
            await asyncio.to_thread(blob.patch)
            logger.info(f"Uploaded: {destination}")
        else:
            logger.info(f"Skipping upload (already exists): {destination}")

    async def upload_bytes(self, data: bytes, destination: str, content_type: str):
        """Upload bytes to Cloud Storage, checking for existence."""
        blob = self.bucket.blob(destination)
        if not blob.exists():
            blob.metadata = {'Cache-Control': 'public, max-age=86400'}
            await asyncio.to_thread(
                blob.upload_from_string,
                data,
                content_type=content_type
            )
            await asyncio.to_thread(blob.patch)
            logger.info(f"Uploaded: {destination}")
        else:
            logger.info(f"Skipping upload (already exists): {destination}")

    async def upload_json(self, data: Dict, destination: str):
        """Upload JSON data to Cloud Storage, checking for existence."""
        blob = self.bucket.blob(destination)
        if not blob.exists():
            blob.metadata = {'Cache-Control': 'public, max-age=86400'}
            await asyncio.to_thread(
                blob.upload_from_string,
                json.dumps(data, ensure_ascii=False, indent=2),
                content_type='application/json'
            )
            await asyncio.to_thread(blob.patch)
            logger.info(f"Uploaded: {destination}")
        else:
            logger.info(f"Skipping upload (already exists): {destination}")


    def cleanup_processed_files(self, manuscript_dir: Path):
        """Deletes the temporary 'processed' directory and its contents."""
        processed_dir = manuscript_dir / "processed"
        if processed_dir.exists():
            try:
                shutil.rmtree(processed_dir)
                logger.info(f"Cleaned up processed files for: {manuscript_dir.name}")
            except Exception as e:
                logger.error(f"Error cleaning up processed files for {manuscript_dir.name}: {e}")

async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process and upload manuscript data')
    parser.add_argument('catalogue_dir', help='Path to the catalogue directory')
    parser.add_argument('--bucket', default='ley-star', help='Cloud Storage bucket name')
    parser.add_argument('--cdn-domain', help='Custom CDN domain (optional)')
    parser.add_argument('--project', default='gen-lang-client-0604038741', help='Google Cloud Project ID')

    args = parser.parse_args()
    catalogue_path = Path(args.catalogue_dir)

    if not catalogue_path.exists():
        logger.error(f"Catalogue directory not found: {catalogue_path}")
        return

    processor = ManuscriptProcessor(args.bucket, args.cdn_domain, args.project)

    manuscript_dirs = [d for d in catalogue_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(manuscript_dirs)} manuscript directories to process")

    if not manuscript_dirs:
        logger.error(f"No manuscript directories found in {catalogue_path}")
        return

    tasks = []
    for manuscript_dir in manuscript_dirs:
        logger.info(f"Queuing manuscript directory: {manuscript_dir.name}")
        tasks.append(processor.process_manuscript(manuscript_dir))

    await asyncio.gather(*tasks)

    # --- Cleanup ALL processed folders (AFTER all manuscripts are processed) ---
    for manuscript_dir in manuscript_dirs:
        processed_dir = manuscript_dir / "processed"
        if processed_dir.exists():
            try:
                shutil.rmtree(processed_dir)
                logger.info(f"Cleaned up processed files for: {manuscript_dir.name}")
            except Exception as e:
                logger.error(f"Error cleaning up processed files for {manuscript_dir.name}: {e}")

    logger.info("All manuscripts processed successfully!")

if __name__ == '__main__':
    asyncio.run(main())