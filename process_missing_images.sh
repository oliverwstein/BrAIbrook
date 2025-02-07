#!/bin/bash

# Directory containing manuscript folders
RAW_DIR="data/raw"

# Counter for manuscripts processed
count=0

# Check if the raw directory exists
if [ ! -d "$RAW_DIR" ]; then
    echo "Error: Directory $RAW_DIR not found"
    exit 1
fi

echo "Checking manuscript folders for missing images..."

# Loop through each subdirectory in the raw directory
for dir in "$RAW_DIR"/*/; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Check if metadata.json exists
        if [ -f "${dir}metadata.json" ]; then
            # Count image files (jpg, jpeg, png, tif, tiff)
            image_count=$(find "$dir" -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.tif" -o -name "*.tiff" \) | wc -l)
            
            # If there are no images but metadata exists
            if [ "$image_count" -eq 0 ]; then
                echo "Found manuscript without images: ${dir}"
                echo "Processing..."
                python download_manuscript_images.py "$dir"
                count=$((count + 1))
                echo "----------------------------------------"
            fi
        fi
    fi
done

echo "Processing complete. Found and processed $count manuscripts without images."