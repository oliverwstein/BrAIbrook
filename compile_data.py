import os
import json
import shutil
from pathlib import Path
import re
import hashlib

def create_manuscript_identifier(title: str, hash_length: int = 8) -> str:
    """
    Generate a consistent, readable, and alphabetically sortable manuscript identifier.
    """
    # Create a deterministic hash from the full title
    title_hash = hashlib.sha256(title.encode('utf-8')).hexdigest()
    
    # Convert hash to a lexicographically sortable representation
    sortable_hash = title_hash[:8]
    
    # Normalize the title for the filename
    normalized = re.sub(r'[^a-zA-Z0-9]+', '-', title[:20]).lower().strip('-')
    
    return f"{normalized}-{sortable_hash}"

def create_catalogue(raw_dir, transcripts_dir, catalogue_dir):
    """
    Create a catalogue of manuscripts by copying files from raw and transcripts directories.
    
    Args:
        raw_dir (str): Path to the raw manuscripts directory
        transcripts_dir (str): Path to the transcripts directory
        catalogue_dir (str): Path to the output catalogue directory
    """
    # Ensure catalogue directory exists
    os.makedirs(catalogue_dir, exist_ok=True)
    
    # Process each folder in the raw directory
    for folder in os.listdir(raw_dir):
        # Path to the source metadata
        metadata_path = os.path.join(raw_dir, folder, 'metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                # Read metadata to get the title
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Create manuscript identifier
                manuscript_id = create_manuscript_identifier(metadata.get('Title', folder))
                
                # Create manuscript catalogue folder
                catalogue_manuscript_path = os.path.join(catalogue_dir, manuscript_id)
                os.makedirs(catalogue_manuscript_path, exist_ok=True)
                
                # Copy metadata
                shutil.copy(metadata_path, os.path.join(catalogue_manuscript_path, 'metadata.json'))
                
                # Copy images
                images_src = os.path.join(raw_dir, folder)
                images_dest = os.path.join(catalogue_manuscript_path, 'images')
                os.makedirs(images_dest, exist_ok=True)
                
                # Copy image files
                for img_file in os.listdir(images_src):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        shutil.copy(
                            os.path.join(images_src, img_file), 
                            os.path.join(images_dest, img_file)
                        )
                
                # Try to copy transcript if it exists
                record_id = metadata.get('Record ID')
                if record_id:
                    for transcript_folder in os.listdir(transcripts_dir):
                        transcript_path = os.path.join(transcripts_dir, transcript_folder, 'transcription.json')
                        if os.path.exists(transcript_path):
                            try:
                                with open(transcript_path, 'r', encoding='utf-8') as f:
                                    transcript_data = json.load(f)
                                    if transcript_data.get('metadata', {}).get('Record ID') == record_id:
                                        shutil.copy(transcript_path, os.path.join(catalogue_manuscript_path, 'transcription.json'))
                                        break
                            except (json.JSONDecodeError, IOError):
                                continue
            
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error processing {folder}: {e}")

def main():
    raw_dir = 'data/raw'
    transcripts_dir = 'data/transcripts'
    catalogue_dir = 'data/catalogue'
    
    create_catalogue(raw_dir, transcripts_dir, catalogue_dir)
    print(f"Catalogue created in {catalogue_dir}")

if __name__ == "__main__":
    main()