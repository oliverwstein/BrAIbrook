#!/bin/bash

# --- Configuration ---
BUCKET_NAME="ley-star"
CATALOGUE_FOLDER="catalogue"

# --- Functions ---

# Function to process a single manuscript
process_manuscript() {
  local manuscript_dir="$1"
  local manuscript_id=$(basename "$manuscript_dir")

  echo "Processing manuscript: $manuscript_id"

  # Create temporary processing directories
  mkdir -p "$manuscript_dir/processed/thumbs"
  mkdir -p "$manuscript_dir/processed/web"

  # 1. Convert JPGs to WebP in two sizes
  find "$manuscript_dir/images" -name "*.jpg" -print0 | while IFS= read -r -d $'\0' jpg_file; do
    base_name=$(basename "${jpg_file%.jpg}")
    
    # Create thumbnail (300px wide)
    magick "$jpg_file" -strip -resize "300x>" -quality 85 -define webp:method=6 \
      "$manuscript_dir/processed/thumbs/${base_name}.webp"
    echo "  Created thumbnail: ${base_name}.webp"
    
    # Create web version (1600px wide)
    magick "$jpg_file" -strip -resize "1600x>" -quality 85 -define webp:method=6 \
      "$manuscript_dir/processed/web/${base_name}.webp"
    echo "  Created web version: ${base_name}.webp"
  done

  # 2. Upload processed images and metadata to GCS
  # Create the manuscript directory in the bucket
  gsutil -m cp "$manuscript_dir/standard_metadata.json" "gs://$BUCKET_NAME/$CATALOGUE_FOLDER/$manuscript_id/"
  
  # Upload thumbnails
  gsutil -m cp -r "$manuscript_dir/processed/thumbs/"*.webp \
    "gs://$BUCKET_NAME/$CATALOGUE_FOLDER/$manuscript_id/images/thumbs/"
  
  # Upload web versions
  gsutil -m cp -r "$manuscript_dir/processed/web/"*.webp \
    "gs://$BUCKET_NAME/$CATALOGUE_FOLDER/$manuscript_id/images/web/"
  
  echo "  Uploaded WebP images and metadata.json to GCS for: $manuscript_id"

  # 3. Set Cache-Control headers for CDN
  gsutil -m setmeta -h "Cache-Control:public, max-age=86400" \
    "gs://$BUCKET_NAME/$CATALOGUE_FOLDER/$manuscript_id/images/thumbs/*"
  gsutil -m setmeta -h "Cache-Control:public, max-age=86400" \
    "gs://$BUCKET_NAME/$CATALOGUE_FOLDER/$manuscript_id/images/web/*"
  
  # 4. Clean up processed files
  rm -rf "$manuscript_dir/processed"
  echo "  Cleaned up temporary files in: $manuscript_dir"
  echo "---"
  
  # 5. Print size information
  echo "Size information for $manuscript_id:"
  echo "  Original JPGs:"
  du -sh "$manuscript_dir/images"
  echo "  Uploaded to Cloud Storage:"
  gsutil du -sh "gs://$BUCKET_NAME/$CATALOGUE_FOLDER/$manuscript_id"
}

# --- Main Script ---

# Check if a directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <catalogue_directory>"
  exit 1
fi

# Check if the catalogue directory exists
if [ ! -d "$1" ]; then
  echo "Error: Catalogue directory '$1' not found."
  exit 1
fi

# Check if required commands are available
if ! command -v magick &> /dev/null; then
    echo "Error: ImageMagick is not installed. Please install it first."
    exit 1
fi

if ! command -v gsutil &> /dev/null; then
    echo "Error: gsutil is not installed. Please install Google Cloud SDK first."
    exit 1
fi

# Loop through all manuscript directories
find "$1" -maxdepth 1 -mindepth 1 -type d -print0 | while IFS= read -r -d $'\0' manuscript_dir; do
  process_manuscript "$manuscript_dir"
done

echo "Processing complete! All manuscripts have been converted and uploaded."
exit 0