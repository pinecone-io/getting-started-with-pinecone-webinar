"""Assistant load command implementation."""

from pathlib import Path
from tqdm import tqdm
from shared import utils


def upload_files(assistant):
    """Upload JSON files to the Pinecone Assistant.
    
    Finds and uploads all JSON files in the data directory, including split parts.
    
    Args:
        assistant: Pinecone Assistant object
        
    Returns:
        tuple: List of upload responses
    """
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "steam_dataset_2025_csv"
    
    # Find all JSON files (including split parts)
    # Sort to ensure consistent order: applications first, then reviews
    applications_files = sorted(data_dir.glob("applications*.json"))
    reviews_files = sorted(data_dir.glob("reviews*.json"))
    
    #all_files = list(applications_files) + list(reviews_files)
    all_files =  list(reviews_files)
    
    if not all_files:
        print("No JSON files found to upload")
        return tuple([])
    
    responses = []
    
    # Upload files with progress tracking
    for file_path in tqdm(all_files, desc="Uploading files", unit="file"):
        filename = file_path.name
        tqdm.write(f"Uploading {filename}...")
        try:
            response = assistant.upload_file(
                file_path=str(file_path),
                timeout=None
            )
            responses.append(response)
            tqdm.write(f"✓ Completed {filename}")
        except Exception as e:
            tqdm.write(f"✗ Failed to upload {filename}: {str(e)}")
            raise
    
    return tuple(responses)


def main():
    """Execute the assistant-load command."""
    # Get the Pinecone assistant
    assistant = utils.get_assistant()

    # Convert JSON if needed
    utils.convert_steam_datasets_to_json()

    # Upload files to assistant
    upload_files(assistant)


if __name__ == "__main__":
    main()
