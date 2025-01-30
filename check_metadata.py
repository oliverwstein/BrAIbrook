import json
import os
import glob

def analyze_manuscript_data(data_dir):
    """
    Analyzes manuscript metadata in a directory, checks for iiif_manifest_id,
    and reports on manuscripts without it.

    Args:
      data_dir: The path to the root directory containing manuscript folders.
    """

    manuscripts_without_manifest = []

    for manuscript_folder in os.listdir(data_dir):
        manuscript_path = os.path.join(data_dir, manuscript_folder)

        # Skip if it's not a directory
        if not os.path.isdir(manuscript_path):
          continue

        metadata_file = os.path.join(manuscript_path, "metadata.json")

        # Ensure metadata file exists
        if not os.path.exists(metadata_file):
          print(f"Warning: No metadata.json found in {manuscript_path}")
          continue
        
        try:
          with open(metadata_file, "r", encoding="utf-8") as f:
              metadata = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON file: {metadata_file}. Skipping.")
            continue
        except Exception as e:
          print(f"Error reading metadata {metadata_file}: {e}. Skipping.")
          continue
        
        # Check for iiif_manifest_id
        if "iiif_manifest_id" not in metadata:
          # Count the number of jpgs
          jpg_count = len(glob.glob(os.path.join(manuscript_path, "*.jpg")))
          
          #Extract Title and Record ID
          title = metadata.get("Title", "N/A")
          record_id = metadata.get("Record ID", "N/A")
          
          #Add the data to the list
          manuscripts_without_manifest.append({
              "folder": manuscript_folder,
              "title": title,
              "record_id": record_id,
              "jpg_count": jpg_count
          })

    # Print the results
    if manuscripts_without_manifest:
      print("Manuscripts without 'iiif_manifest_id':")
      for item in manuscripts_without_manifest:
        print(f"- Folder: {item['folder']},\n Title: {item['title']}\n JPGs: {item['jpg_count']}")
    else:
      print("All manuscripts have 'iiif_manifest_id'.")

if __name__ == "__main__":
    data_directory = "data/raw"  # Change this path if your folder is somewhere else
    analyze_manuscript_data(data_directory)