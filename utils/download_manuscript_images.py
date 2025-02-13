import os
import json
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re

def download_image(url: str, filepath: str) -> None:
    """Download an image from a URL to a given filepath."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filepath, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def get_ids_sequential_images(base_id: str, max_attempts: int = 20) -> list:
    """Get sequential image URLs for IDS manifests by trying consecutive IDs."""
    base_url = f"https://ids.lib.harvard.edu/ids/iiif/{base_id}/full/full/0/default.jpg"
    urls = []
    
    # Try the base ID first
    response = requests.get(base_url)
    if response.status_code == 200:
        urls.append(base_url)
        base_num = int(base_id)
        
        # Try subsequent numbers
        for i in range(1, max_attempts):
            next_id = str(base_num + i)
            url = f"https://ids.lib.harvard.edu/ids/iiif/{next_id}/full/full/0/default.jpg"
            response = requests.head(url)  # Use HEAD request to check existence
            if response.status_code == 200:
                urls.append(url)
            else:
                break  # Stop when we hit a 404
    
    return urls

def download_iiif_images(manifest_id: str, manuscript_dir: str) -> None:
    """Download images from a IIIF manifest."""
    try:
        image_urls = []
        
        if 'ids:' in manifest_id:
            # Handle IDS manifest
            match = re.search(r'ids:(\d+)', manifest_id)
            if match:
                base_id = match.group(1)
                print(f"Processing IDS manifest with base ID: {base_id}")
                image_urls = get_ids_sequential_images(base_id)
        else:
            # Handle DRS manifest as before
            manifest_url = manifest_id
            print(f"Processing DRS manifest: {manifest_url}")
            response = requests.get(manifest_url)
            response.raise_for_status()
            manifest = response.json()

            if 'sequences' in manifest and manifest['sequences']:
                canvases = manifest['sequences'][0].get('canvases', [])
                for canvas in canvases:
                    if 'images' in canvas and canvas['images']:
                        for image in canvas['images']:
                            if 'resource' in image and '@id' in image['resource']:
                                image_url = image['resource']['@id']
                                if image_url.endswith('default.jpg'):
                                    image_urls.append(image_url)

        if not image_urls:
            print("No image URLs found")
            return

        print(f"Found {len(image_urls)} images")
        
        # Create image paths
        image_paths = [os.path.join(manuscript_dir, f"{i:04}.jpg") 
                      for i in range(len(image_urls))]

        # Download images using thread pool
        with ThreadPoolExecutor(max_workers=5) as executor:
            list(tqdm(
                executor.map(download_image, image_urls, image_paths),
                total=len(image_urls),
                desc="Downloading images"
            ))
            
        print(f"Successfully downloaded {len(image_urls)} images to {manuscript_dir}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading IIIF manifest or images: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from IIIF manifest: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def process_manuscript_folder(folder_path: str) -> None:
    """Process a single manuscript folder to download its images."""
    folder_path = Path(folder_path)
    
    # Check if metadata.json exists
    metadata_path = folder_path / 'metadata.json'
    if not metadata_path.exists():
        print(f"No metadata.json found in {folder_path}")
        return

    try:
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Extract IIIF manifest ID
        iiif_manifest_id = metadata.get('iiif_manifest_id')
        if not iiif_manifest_id:
            print(f"No IIIF manifest ID found in metadata for {folder_path}")
            return

        # Download images
        print(f"Processing manuscript in {folder_path}")
        print(f"Using manifest ID: {iiif_manifest_id}")
        download_iiif_images(iiif_manifest_id, str(folder_path))

    except Exception as e:
        print(f"Error processing {folder_path}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download images for a single manuscript folder')
    parser.add_argument('folder', help='Path to the manuscript folder containing metadata.json')
    
    args = parser.parse_args()
    process_manuscript_folder(args.folder)