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

DATA STRUCTURE:
- APPLICATION records (applications*.json): Each record is a Steam game with fields: appid, name, release_date, short_description, metacritic_score, recommendations_total
- REVIEW records (reviews*.json): Each record is a user review with fields: recommendationid, appid, review_text, weighted_vote_score, comment_count, author_num_reviews, author_num_games_owned, author_playtime_forever, author_playtime_at_review, timestamp_created

CRITICAL RULES:
1. Always include a Steam store link for EVERY game you mention: https://store.steampowered.com/app/[appid]/
2. If a game has no name or blank name, display it as "appid: [appid]" instead of using the name
3. Use pre-calculated metrics—never count manually:
   - Game review totals: `recommendations_total` (APPLICATION records only)
   - Reviewer stats: `author_num_reviews`, `author_num_games_owned` (REVIEW records)
4. Favor games with higher review counts when relevance is similar

HOW TO ANSWER:

Game-level questions (e.g., "Which games have the most reviews?"):
- Search APPLICATION records
- Rank by `recommendations_total`
- Format each game as:
  - If name exists: "**[Game Name]** ([recommendations_total] reviews, metacritic: [score]) - [Steam Store](https://store.steampowered.com/app/[appid]/)"
  - If no name: "**appid: [appid]** ([recommendations_total] reviews, metacritic: [score]) - [Steam Store](https://store.steampowered.com/app/[appid]/)"

Review content/sentiment questions (e.g., "What games are similar to Cyberpunk 2077 and well liked?"):
- Search REVIEW records for relevant content
- Review records often do NOT have game names, only appids
- Format each game as:
  - If name exists: "**[Game Name]** - [Steam Store](https://store.steampowered.com/app/[appid]/)"
  - If no name: "**appid: [appid]** - [Steam Store](https://store.steampowered.com/app/[appid]/)"
  
  Then provide the review evidence:
  - Quote relevant review excerpts that show similarity/sentiment
  - Cite weighted_vote_score and comment_count
  - Explain why it matches the user's question based on review content

Reviewer questions (e.g., "Which reviewers have written the most?"):
- Search REVIEW records
- Use `author_num_reviews` and `author_num_games_owned` directly
- No Steam links needed for reviewer-focused questions

FORMATTING REQUIREMENTS:
- Every game mention must include its Steam store link: https://store.steampowered.com/app/[appid]/
- Games without names should be labeled "appid: [appid]" not "Unnamed Game"
- Make Steam links clickable by using markdown format: [Steam Store](https://store.steampowered.com/app/[appid]/)

Always provide specific, data-backed answers with relevant metrics and Steam store links."""
    
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


def convert_csv_to_json(csv_path: str, json_path: str = None, max_size_mb: float = 100.0) -> list[str]:
    """Convert a CSV file to JSON format, splitting into multiple files if needed.
    
    Args:
        csv_path: Path to the CSV file
        json_path: Path template for output JSON files (defaults to same directory with .json extension). If file is split, parts will be named: filename_part1.json, filename_part2.json, etc.
        max_size_mb: Maximum size in MB for each JSON file (default: 100.0, the Pinecone max file size)
        
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
    
    # Estimate size per record
    max_size_bytes = max_size_mb * 1024 * 1024
    if len(records) > 0:
        sample_json = json.dumps([records[0]], ensure_ascii=False)
        sample_size = len(sample_json.encode('utf-8'))
        records_per_file = max(1, int((max_size_bytes * 0.95) / sample_size))
    else:
        records_per_file = len(records)
    
    json_files = []
    total_records = len(records)
    current_idx = 0
    part_num = 1
    base_name = json_path.stem
    base_dir = json_path.parent
    
    while current_idx < total_records:
        # Determine chunk size and create filename
        chunk_size = min(records_per_file, total_records - current_idx)
        chunk_records = records[current_idx:current_idx + chunk_size]
        
        if total_records <= records_per_file:
            part_filename = json_path
        else:
            part_filename = base_dir / f"{base_name}_part{part_num}.json"
        
        # Write chunk and check size, reducing if needed
        current_chunk_size = len(chunk_records)
        while current_chunk_size > 0:
            with open(part_filename, 'w', encoding='utf-8') as f:
                json.dump(chunk_records[:current_chunk_size], f, ensure_ascii=False)
            
            if part_filename.stat().st_size <= max_size_bytes:
                break
            
            # File too large, reduce chunk size
            if part_filename.exists():
                part_filename.unlink()
            current_chunk_size = int(current_chunk_size * 0.9)
            if current_chunk_size < 1:
                current_chunk_size = 1
        
        json_files.append(str(part_filename))
        print(f"Converted {current_chunk_size} rows" + 
              (f" (part {part_num})" if total_records > records_per_file else "") + 
              f" from {csv_file.name} to {part_filename.name}")
        
        current_idx += current_chunk_size
        part_num += 1
    
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
