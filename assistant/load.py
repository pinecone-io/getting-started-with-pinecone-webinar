"""Assistant load command implementation."""

from pathlib import Path
from tqdm import tqdm
from shared import utils


def upload_files(assistant):
    """Upload JSON files to the Pinecone Assistant.
    
    Args:
        assistant: Pinecone Assistant object
        
    Returns:
        tuple: (applications_response, reviews_response)
    """
    project_root = Path(__file__).parent.parent
    files_to_upload = [
        project_root / "data" / "steam_dataset_2025_csv" / "applications.json",
        project_root / "data" / "steam_dataset_2025_csv" / "reviews.json"
    ]
    
    responses = []
    
    # Upload files with progress tracking
    for file_path in tqdm(files_to_upload, desc="Uploading files", unit="file"):
        filename = file_path.name
        tqdm.write(f"Uploading {filename}...")
        response = assistant.upload_file(
            file_path=str(file_path),
            timeout=None
        )
        responses.append(response)
        tqdm.write(f"✓ Completed {filename}")
    
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
