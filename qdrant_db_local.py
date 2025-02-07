import os
import json
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import time

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_DIR = 'data/catalogue'
COLLECTION_NAME = 'manuscripts'
CHUNK_TYPE = 'paragraph'
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
MANUSCRIPT_EMBEDDING_TYPE = "summary_text"
CONTROL_POINT_ID = 0  # Define CONTROL_POINT_ID
# -----------------------

# Global counter for point IDs
point_id_counter = 1

def get_next_point_id():
    """Gets the next available point ID."""
    global point_id_counter
    next_id = point_id_counter
    point_id_counter += 1
    return next_id

def split_into_paragraphs(text):
    return text.split('\n\n')

def split_into_sentences(text):
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

def chunk_text(text, chunk_type='paragraph'):
    if chunk_type == 'sentence':
        return split_into_sentences(text)
    elif chunk_type == 'paragraph':
        return split_into_paragraphs(text)
    else:
        raise ValueError("Invalid chunk_type")

def concatenate_summary_fields(summary_dict):
    text_fields = ["title", "contents_summary", "historical_context", "significance"]
    text_parts = [str(summary_dict.get(field, "")) for field in text_fields]
    return " ".join(text_parts)

def add_point(client, collection_name, point_id, embedding, payload):
    """Adds a single point to Qdrant, handling potential errors."""
    try:
        client.upsert(
            collection_name=collection_name,
            points=[models.PointStruct(id=point_id, vector=embedding.tolist(), payload=payload)],
            wait=True
        )
    except Exception as e:
        print(f"Error upserting point with ID {point_id}: {e}")

def process_manuscript(manuscript_dir, client, model, chunk_type, manuscript_embedding_type):
    """Processes a single manuscript directory."""
    transcription_file = os.path.join(manuscript_dir, 'transcription.json')
    if not os.path.exists(transcription_file):
        print(f"Skipping {manuscript_dir} (no transcription.json)")
        return

    try:
        with open(transcription_file, 'r', encoding='utf-8') as f:
            manuscript_data = json.load(f)
    except (json.JSONDecodeError, OSError, IOError) as e:
        print(f"Error loading JSON from {transcription_file}: {e}")
        return

    manuscript_id = os.path.basename(manuscript_dir)
    points_to_upsert = []

    # --- Manuscript-Level Data ---
    summary_data = manuscript_data.get("summary", {})
    metadata = manuscript_data.get("metadata", {})

    if manuscript_embedding_type != "none":
        if manuscript_embedding_type == "summary_text":
            summary_text = summary_data.get("contents_summary", "")
        elif manuscript_embedding_type == "title":
            summary_text = manuscript_data.get("manuscript_title", "")
        elif manuscript_embedding_type == "summary_dict":
            summary_text = concatenate_summary_fields(summary_data)
        else:
            raise ValueError("Invalid MANUSCRIPT_EMBEDDING_TYPE")

        if summary_text:
            try:
                summary_embedding = model.encode(summary_text)
                payload = {
                    "manuscript_id": manuscript_id,
                    "type": "summary",
                    "title": manuscript_data.get("manuscript_title", ""),
                    "metadata": metadata,  # Store metadata directly
                    "summary": summary_data,  # Store summary directly
                }
                add_point(client, COLLECTION_NAME, get_next_point_id(), summary_embedding, payload)
            except Exception as e:
                print(f"Error processing summary for {manuscript_id}: {e}")

    # --- Table of Contents ---
    if "table_of_contents" in manuscript_data:
        for entry in manuscript_data["table_of_contents"]:
            if entry and "title" in entry:
                try:
                    toc_embedding = model.encode(entry["title"])
                    payload = {
                        "manuscript_id": manuscript_id,
                        "type": "toc_entry",
                        "page_number": int(entry.get("page_number", -1)),
                        "toc_title": entry["title"],
                        "toc_level": int(entry.get("level", 0)),
                        "toc_description": entry.get("description"),
                        "toc_synopsis": entry.get("synopsis")
                    }
                    add_point(client, COLLECTION_NAME, get_next_point_id(), toc_embedding, payload)
                except Exception as e:
                    print(f"Error processing TOC entry in {manuscript_id}: {e}")

    # --- Page-Level Data ---
    if "pages" in manuscript_data:
        for page_number, page_data in manuscript_data["pages"].items():
            page_number_int = int(page_number)  # Convert page_number to integer

            # Revised Transcription Chunks
            if "revised_transcription" in page_data:
                chunks = chunk_text(page_data["revised_transcription"], chunk_type)
                if chunks:
                    try:
                        embeddings = model.encode(chunks)
                        for i, embedding in enumerate(embeddings):
                            payload = {
                                "manuscript_id": manuscript_id,
                                "page_number": page_number_int,
                                "type": "section",
                            }
                            add_point(client, COLLECTION_NAME, get_next_point_id(), embedding, payload)
                    except Exception as e:
                        print(f"Error processing revised_transcription chunks on page {page_number} of {manuscript_id}: {e}")

            # Page Summary
            if "summary" in page_data and page_data["summary"]:
                try:
                    page_summary_embedding = model.encode(page_data["summary"])
                    payload = {
                        "manuscript_id": manuscript_id,
                        "page_number": page_number_int,
                        "type": "page_summary",
                    }
                    add_point(client, COLLECTION_NAME, get_next_point_id(), page_summary_embedding, payload)
                except Exception as e:
                    print(f"Error processing page summary on page {page_number} of {manuscript_id}: {e}")

            # Marginalia
            if "marginalia" in page_data and page_data["marginalia"]:
                try:
                    marginalia_text = " ".join(page_data["marginalia"])
                    if marginalia_text:
                        marginalia_embedding = model.encode([marginalia_text])[0]
                        payload = {
                            "manuscript_id": manuscript_id,
                            "page_number": page_number_int,
                            "type": "marginalia",
                        }
                        add_point(client, COLLECTION_NAME, get_next_point_id(), marginalia_embedding, payload)
                except Exception as e:
                    print(f"Error processing marginalia on page {page_number} of {manuscript_id}: {e}")

            # Content Notes
            if "content_notes" in page_data and page_data["content_notes"]:
                try:
                    content_notes_embedding = model.encode(page_data["content_notes"])
                    payload = {
                        "manuscript_id": manuscript_id,
                        "page_number": page_number_int,
                        "type": "content_notes",
                    }
                    add_point(client, COLLECTION_NAME, get_next_point_id(), content_notes_embedding, payload)
                except Exception as e:
                    print(f"Error processing content_notes on page {page_number} of {manuscript_id}: {e}")

def main():
    global point_id_counter
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    model = SentenceTransformer(MODEL_NAME)

    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists")
    except:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created")
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=CONTROL_POINT_ID,
                    vector=[0.0] * 384,  # Dummy vector of correct dimension
                    payload={
                        "type": "control",
                        "last_processed_timestamp": 0,
                        "processed_manuscripts": {},
                    },
                )
            ],
            wait=True
        )
        print(f"Control point with ID {CONTROL_POINT_ID} created.")

    # Load processed manuscripts and last_processed_timestamp from the control point
    try:
        control_point = client.retrieve(collection_name=COLLECTION_NAME, ids=[CONTROL_POINT_ID], with_payload=True)
        if control_point:
            last_processed_timestamp = control_point[0].payload["last_processed_timestamp"]
            processed_manuscripts = control_point[0].payload["processed_manuscripts"]
            #Resume ID counter
            largest_id = client.count(collection_name=COLLECTION_NAME, exact=False).count #Get count, not exact
            if largest_id > 0:
                point_id_counter = largest_id +1
            print(f"Starting Point ID Counter at {point_id_counter}")

        else: #Handle the case of no control point
            last_processed_timestamp = 0
            processed_manuscripts = {}
            print("No control point found. Starting fresh.")
    except Exception as e:
        print(f"Error retrieving control point: {e}, defaulting to 0")
        last_processed_timestamp = 0
        processed_manuscripts = {}


    current_timestamp = time.time()

    for manuscript_dir in tqdm(os.listdir(DATA_DIR), desc="Manuscripts"):
        full_manuscript_dir = os.path.join(DATA_DIR, manuscript_dir)
        if not os.path.isdir(full_manuscript_dir):
            continue

        manuscript_id = os.path.basename(full_manuscript_dir)
        dir_last_modified_time = os.path.getmtime(full_manuscript_dir)

        if manuscript_id in processed_manuscripts and processed_manuscripts[manuscript_id] == dir_last_modified_time:
            print(f"Skipping {manuscript_dir} (already processed and unchanged)")
            continue

        process_manuscript(full_manuscript_dir, client, model, CHUNK_TYPE, MANUSCRIPT_EMBEDDING_TYPE)
        processed_manuscripts[manuscript_id] = dir_last_modified_time


    # Update the control point with the current timestamp and processed manuscripts
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=CONTROL_POINT_ID,
                    vector=[0.0] * 384,  # Dummy vector
                    payload={
                        "type": "control",
                        "last_processed_timestamp": current_timestamp,
                        "processed_manuscripts": processed_manuscripts,
                    },
                )
            ],
            wait=True
        )
        print(f"Control point updated with timestamp: {current_timestamp}")
    except Exception as e:
        print(f"Error updating control point: {e}")
    print("Qdrant database creation complete!")

if __name__ == "__main__":
    main()