"""Shared utility functions."""

import json
import pandas as pd
from pathlib import Path
import os
import time
import pinecone


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    path = Path(file_path)
    # Check if the file exists
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, low_memory=False)

    print(f"Loaded {len(df)} rows from {file_path}")

    # Return the DataFrame
    return df


def load_steam_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both Steam dataset CSV files and return as a tuple of DataFrames.
    
    Returns:
        tuple: (applications_df, reviews_df)
    """
    # Make the paths to the CSV files relative to the project root
    project_root = Path(__file__).parent.parent
    applications_path = project_root / "data" / "steam_dataset_2025_csv" / "applications.csv"
    reviews_path = project_root / "data" / "steam_dataset_2025_csv" / "reviews.csv"
    
    # Load the CSV files into pandas DataFrames
    applications_df = load_csv(str(applications_path))
    reviews_df = load_csv(str(reviews_path))
    
    # Return the DataFrames as a tuple
    return applications_df, reviews_df


def get_indexes() -> tuple[pinecone.Pinecone.Index, pinecone.Pinecone.Index]:
    """
    Get or create the Pinecone indexes.
    
    Returns:
        tuple: (dense_index, sparse_index)
    """
    pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    dense_index_name = os.getenv('PINECONE_DENSE_INDEX')
    sparse_index_name = os.getenv('PINECONE_SPARSE_INDEX')
    
    # Create dense index if it doesn't exist
    if not pc.has_index(dense_index_name):
        pc.create_index_for_model(
            name=dense_index_name,
            cloud="aws",
            region="us-east-1", 
            embed={
                "model":"llama-text-embed-v2",
                "field_map":{"text": "text"}
            }
        )
        # Wait for index to be ready
        while not pc.describe_index(dense_index_name).status['ready']:
            time.sleep(1)

        print(f"Created dense index: {dense_index_name}")
    
    # Create sparse index if it doesn't exist
    if not pc.has_index(sparse_index_name):
        pc.create_index_for_model(
            name=sparse_index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "pinecone-sparse-english-v0",
                "field_map": {"text": "text"}
            }
        )
        # Wait for index to be ready
        while not pc.describe_index(sparse_index_name).status['ready']:
            time.sleep(1)
        print(f"Created sparse index: {sparse_index_name}")
    
    # Get the indexes
    dense_index = pc.Index(dense_index_name)
    sparse_index = pc.Index(sparse_index_name)
    
    return dense_index, sparse_index


def get_assistant():
    """
    Get or create the Pinecone Assistant with instructions about the data structure.
    
    Returns:
        Pinecone Assistant object
    """
    pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    assistant_name = os.getenv('PINECONE_ASSISTANT_NAME')
    
    # Comprehensive instructions explaining the data structure and relationships
    instructions = """You are a helpful assistant that answers questions about Steam games and reviews.

IMPORTANT: Understanding the Data Structure

The data consists of two types of records with a parent-child relationship:

1. APPLICATION RECORDS (Parent records):
   - Each application represents a Steam game
   - Key fields: appid (unique identifier), name, release_date, short_description, metacritic_score, recommendations_total
   - Example: {"appid": 10, "name": "Counter-Strike", "release_date": "2000-11-01", "short_description": "Play the world's number 1 online action game...", "metacritic_score": 88.0, "recommendations_total": 161854.0}

2. RECOMMENDATION/REVIEW RECORDS (Child records):
   - Each recommendation is a user review for a specific game
   - Key fields: recommendationid (unique identifier), appid (links to parent application), review_text, weighted_vote_score, timestamp_created, author_playtime_forever
   - The appid field in each recommendation record links it to its parent application
   - Example: {"recommendationid": 10000000, "appid": 264220, "review_text": "What's a crap. This game costs 2 euro...", "weighted_vote_score": 0.45990565, "timestamp_created": 1399059965}

RELATIONSHIP: Applications have a 1-to-many relationship with recommendations. Each application (game) can have many recommendations (reviews), and each recommendation belongs to exactly one application via the appid field.

When answering questions about games (e.g., "Which game is the most action-packed?", "What's the scariest game?", "Which game is the most emotional?", "What games have the best reviews?"):
- You MUST consider ALL reviews for each game, not just individual reviews
- Aggregate and analyze reviews by their appid to understand the overall sentiment, themes, and characteristics of each game
- When comparing games, look at the collective body of reviews for each game, not just single reviews
- Use the appid field to group reviews together by their parent application
- Consider metrics like weighted_vote_score across all reviews for a game when making assessments
- Look for patterns and themes across multiple reviews for the same game to form comprehensive assessments

Always provide comprehensive answers that take into account the full set of reviews for each game when making comparisons or recommendations."""
    
    # Check if assistant exists, create if it doesn't
    try:
        pc.assistant.describe_assistant(assistant_name)
    except Exception:
        # Assistant doesn't exist, create it
        pc.assistant.create_assistant(
            assistant_name=assistant_name,
            instructions=instructions,
            region="us"
        )
        print(f"Created assistant: {assistant_name}")
    
    # Get the assistant object
    assistant = pc.assistant.Assistant(assistant_name=assistant_name)
    
    return assistant


def convert_csv_to_json(csv_path: str, json_path: str = None, max_size_mb: float = 350.0) -> list[str]:
    """Convert a CSV file to JSON format, splitting into multiple files if needed.
    
    Args:
        csv_path: Path to the CSV file
        json_path: Path template for output JSON files (defaults to same directory with .json extension). If file is split, parts will be named: filename_part1.json, filename_part2.json, etc.
        max_size_mb: Maximum size in MB for each JSON file (default: 350.0)
        
    Returns:
        List of paths to created JSON files
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Default JSON path: same directory, same name, .json extension
    if json_path is None:
        json_path = csv_file.with_suffix('.json')
    else:
        json_path = Path(json_path)
    
    # Load CSV
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Convert to records format (list of dictionaries)
    records = df.to_dict('records')
    
    # Replace NaN/NaT values with None (which becomes null in JSON)
    # This ensures all values are JSON-serializable
    def clean_record(record):
        cleaned = {}
        for key, value in record.items():
            if pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        return cleaned
    
    records = [clean_record(r) for r in records]
    
    # Estimate size per record by converting a sample
    max_size_bytes = max_size_mb * 1024 * 1024
    if len(records) > 0:
        # Sample first record to estimate size
        sample_json = json.dumps([records[0]], ensure_ascii=False)
        sample_size = len(sample_json.encode('utf-8'))
        # Use 90% of max size to leave margin for JSON overhead
        records_per_file = max(1, int((max_size_bytes * 0.9) / sample_size))
    else:
        records_per_file = len(records)
    
    json_files = []
    total_records = len(records)
    
    # Split records into chunks and write JSON files
    if total_records <= records_per_file:
        # Single file - no splitting needed
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False)
        json_files.append(str(json_path))
        print(f"Converted {total_records} rows from {csv_file.name} to {json_path.name}")
    else:
        # Split into multiple files
        num_parts = (total_records + records_per_file - 1) // records_per_file
        base_name = json_path.stem
        base_dir = json_path.parent
        
        for part_num in range(num_parts):
            start_idx = part_num * records_per_file
            end_idx = min(start_idx + records_per_file, total_records)
            chunk_records = records[start_idx:end_idx]
            
            # Create filename: original_name_part1.json, original_name_part2.json, etc.
            part_filename = base_dir / f"{base_name}_part{part_num + 1}.json"
            
            with open(part_filename, 'w', encoding='utf-8') as f:
                json.dump(chunk_records, f, ensure_ascii=False)
            
            json_files.append(str(part_filename))
            print(f"Converted {len(chunk_records)} rows (part {part_num + 1}/{num_parts}) from {csv_file.name} to {part_filename.name}")
    
    return json_files


def convert_steam_datasets_to_json() -> tuple[list[str], list[str]]:
    """Convert both Steam dataset CSV files to JSON format, splitting if needed.
    
    Returns:
        tuple: (list of applications_json_paths, list of reviews_json_paths)
    """
    project_root = Path(__file__).parent.parent
    applications_csv = project_root / "data" / "steam_dataset_2025_csv" / "applications.csv"
    reviews_csv = project_root / "data" / "steam_dataset_2025_csv" / "reviews.csv"
    
    applications_json_files = convert_csv_to_json(str(applications_csv))
    reviews_json_files = convert_csv_to_json(str(reviews_csv))
    
    return applications_json_files, reviews_json_files
