"""Database load command implementation."""

from datetime import datetime
import pandas as pd
import tiktoken
import os
import pinecone
from tqdm import tqdm

from shared import utils


def applications_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the applications DataFrame.
    
    Args:
        df: The applications DataFrame
        
    Returns:
        Transformed DataFrame with selected columns and combined_text column for embedding
    """
    # The columns we want in the new dataframe
    columns_to_extract = [
        'appid', 
        'name', 
        'release_date', 
        'short_description', 
        'metacritic_score', 
        'recommendations_total'
    ]
    
    # Create new dataframe with selected columns
    new_applications_df = df[columns_to_extract].copy()

    # Fill NaN values with appropriate defaults for Pinecone metadata
    # Numeric fields: use 0 as default
    new_applications_df['metacritic_score'] = new_applications_df['metacritic_score'].fillna(0)
    new_applications_df['recommendations_total'] = new_applications_df['recommendations_total'].fillna(0)
    # String fields: use empty string as default
    new_applications_df['name'] = new_applications_df['name'].fillna('')
    new_applications_df['release_date'] = new_applications_df['release_date'].astype(str)
    new_applications_df['short_description'] = new_applications_df['short_description'].fillna('')
    # appid should never be NaN, but handle it just in case
    new_applications_df['appid'] = new_applications_df['appid'].fillna(0)
    
    # Create combined_text
    new_applications_df['combined_text'] = (
        'AppID: ' + new_applications_df['appid'].astype(str) + '.\n' +
        'Name: ' + new_applications_df['name'].astype(str) + '.\n' +
        'Description: ' + new_applications_df['short_description'].astype(str) + '.\n' +
        'Released: ' + new_applications_df['release_date'].astype(str) + '.\n' +
        'Metacritic Score: ' + new_applications_df['metacritic_score'].astype(str) + '.\n' +
        'Recommendations: ' + new_applications_df['recommendations_total'].astype(str) + '.\n'
    )
    
    return new_applications_df


def reviews_transform(df: pd.DataFrame, app_df: pd.DataFrame) -> pd.DataFrame:
    """Transform the reviews DataFrame.
    
    Args:
        df: The reviews DataFrame
        app_df: The applications DataFrame
        
    Returns:
        Transformed DataFrame
    """
    # The columns we want in the new dataframe
    columns_to_extract = [
        'recommendationid',
        'appid',
        'author_num_games_owned',
        'author_num_reviews',
        'author_playtime_forever',
        'author_playtime_at_review',
        'review_text',
        'timestamp_created',
        'weighted_vote_score',
        'comment_count'
    ]
    
    # Create new dataframe with selected columns
    new_reviews_df = df[columns_to_extract].copy()

    # Fill NaN values with appropriate defaults for Pinecone metadata
    # Numeric fields: use 0 as default
    new_reviews_df['author_num_games_owned'] = new_reviews_df['author_num_games_owned'].fillna(0)
    new_reviews_df['author_num_reviews'] = new_reviews_df['author_num_reviews'].fillna(0)
    new_reviews_df['author_playtime_forever'] = new_reviews_df['author_playtime_forever'].fillna(0)
    new_reviews_df['author_playtime_at_review'] = new_reviews_df['author_playtime_at_review'].fillna(0)
    new_reviews_df['weighted_vote_score'] = new_reviews_df['weighted_vote_score'].fillna(0)
    new_reviews_df['comment_count'] = new_reviews_df['comment_count'].fillna(0)
    # String fields: use empty string as default
    new_reviews_df['review_text'] = new_reviews_df['review_text'].fillna('')
    
    # Convert Unix timestamp to readable date string
    def format_timestamp(timestamp):
        """Convert Unix timestamp to readable date string"""
        if pd.isna(timestamp) or timestamp == '':
            return ''
        try:
            # Convert to numeric if it's a string
            ts = float(timestamp)
            # Convert Unix timestamp to datetime
            dt = datetime.fromtimestamp(ts)
            # Format as readable date string
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError, OSError):
            return str(timestamp)  # Return original if conversion fails
    
    new_reviews_df['timestamp_created'] = new_reviews_df['timestamp_created'].apply(format_timestamp)
    
    # appid and recommendationid should never be NaN, but handle it just in case
    new_reviews_df['appid'] = new_reviews_df['appid'].fillna(0)
    new_reviews_df['recommendationid'] = new_reviews_df['recommendationid'].fillna(0)
    
    # Look up application name from applications dataframe
    app_name_lookup = dict(zip(app_df['appid'], app_df['name']))
    new_reviews_df['name'] = new_reviews_df['appid'].map(app_name_lookup).fillna('')
    
    # Create combined_text
    new_reviews_df['combined_text'] = (
        'AppId: ' + new_reviews_df['appid'].astype(str) + '.\n' +
        'Name: ' + new_reviews_df['name'].astype(str) + '.\n' +
        "RecommendationId: " + new_reviews_df['recommendationid'].astype(str) + '.\n' +
        "Review Created At: " + new_reviews_df['timestamp_created'].astype(str) + '.\n' +
        "Review Weighted Vote Score: " + new_reviews_df['weighted_vote_score'].astype(str) + '.\n' +
        "Review Comment Count: " + new_reviews_df['comment_count'].astype(str) + '.\n' +
        "Author Number of Games Owned: " + new_reviews_df['author_num_games_owned'].astype(str) + '.\n' +
        "Author Number of Reviews: " + new_reviews_df['author_num_reviews'].astype(str) + '.\n' +
        "Author Playtime Forever: " + new_reviews_df['author_playtime_forever'].astype(str) + '.\n' +
        "Author Playtime At Review: " + new_reviews_df['author_playtime_at_review'].astype(str) + '.\n' +
        "Review Text: " + new_reviews_df['review_text'].astype(str) + '.\n'
    )
    
    return new_reviews_df


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> list[str]:
    """
    Chunk text into smaller pieces with overlap for embedding.
    
    Args:
        text: Text content to chunk
        chunk_size: Target chunk size in tokens (defaults to CHUNK_SIZE env var or 1740)
        chunk_overlap: Overlap between chunks in tokens (defaults to CHUNK_OVERLAP env var or 205)
        
    Returns:
        List of chunked text strings
    """
    # Get chunk size and overlap from environment variables with defaults
    if chunk_size is None:
        chunk_size = int(os.getenv('CHUNK_SIZE', '1740'))
    if chunk_overlap is None:
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '205'))
    
    if not text.strip():
        return []
    
    # Initialize tokenizer (same as OpenAI models)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Tokenize the entire text
    tokens = tokenizer.encode(text)

    # Create chunks with overlap
    chunks = []
    start_idx = 0
    
    # If text fits in one chunk, return it
    if len(tokens) <= chunk_size:
        chunks.append(text)
    else:
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = start_idx + chunk_size
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode tokens back to text
            chunk_text_str = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text_str)
            
            # Move start index forward, accounting for overlap
            start_idx = end_idx - chunk_overlap
            
            # Prevent infinite loop if overlap is too large
            if chunk_overlap >= chunk_size:
                break
    
    return chunks


def _upsert_batch_resilient(dense_index, sparse_index, batch, record_type="record"):
    """
    Upsert a batch of records with resilience to individual failures.
    
    If batch upsert fails, tries individual records to identify and skip problematic ones.
    This prevents a single invalid vector from causing the entire batch or process to fail.
    
    Args:
        dense_index: Pinecone dense index object
        sparse_index: Pinecone sparse index object
        batch: List of vector records to upsert
        record_type: Type of record for error messages (e.g., "application", "review")
        
    Returns:
        Tuple of (successful_count, skipped_count, skipped_ids)
    """
    successful_count = 0
    skipped_count = 0
    skipped_ids = []
    
    # Filter out any records with empty or whitespace-only text before upserting
    filtered_batch = []
    for record in batch:
        text = record.get('text', '')
        if text and text.strip():
            filtered_batch.append(record)
        else:
            skipped_count += 1
            skipped_ids.append(record.get('_id', 'unknown'))
    
    # Skip empty batches
    if not filtered_batch:
        return successful_count, skipped_count, skipped_ids
    
    # Try batch upsert first (more efficient)
    try:
        dense_index.upsert_records("__default__", filtered_batch)
        sparse_index.upsert_records("__default__", filtered_batch)
        successful_count = len(filtered_batch)
        return successful_count, skipped_count, skipped_ids
    except pinecone.exceptions.PineconeApiException as e:
        # Check if this is an embedding-related error (empty sparse vector, invalid input, etc.)
        error_msg = str(e).lower()
        is_embedding_error = (
            "empty sparse vector" in error_msg or
            "invalid input" in error_msg or
            "invalid upsert" in error_msg or
            "embedding" in error_msg
        )
        
        if is_embedding_error:
            # Batch failed due to embedding issue - try each record individually
            # to identify and skip the problematic ones
            for record in filtered_batch:
                try:
                    dense_index.upsert_records("__default__", [record])
                    sparse_index.upsert_records("__default__", [record])
                    successful_count += 1
                except pinecone.exceptions.PineconeApiException as individual_error:
                    # Skip this problematic record
                    skipped_count += 1
                    record_id = record.get('_id', 'unknown')
                    skipped_ids.append(record_id)
                    # Don't print individual errors to avoid spam - will report summary at end
                    pass
                except Exception as individual_error:
                    # For non-embedding errors (network, auth, etc.), skip this record
                    # but don't fail the entire process
                    skipped_count += 1
                    record_id = record.get('_id', 'unknown')
                    skipped_ids.append(record_id)
                    pass
        else:
            # Re-raise if it's a different type of error (network, auth, rate limit, etc.)
            # These are more serious and should stop the process
            raise
    except Exception as e:
        # For unexpected errors, re-raise to fail fast
        # This includes network errors, authentication errors, etc.
        raise
    
    return successful_count, skipped_count, skipped_ids


def upsert_applications(df: pd.DataFrame, dense_index: pinecone.Pinecone.Index, sparse_index: pinecone.Pinecone.Index) -> int:
    """
    Upsert application records to Pinecone with chunked descriptions.
    
    For each row in the dataframe:
    - Chunks the combined_text
    - Creates a vector for each chunk (embedded)
    - Includes all other dataframe fields as metadata
    
    Args:
        df: Applications DataFrame
        dense_index: Pinecone dense index object for upserting
        sparse_index: Pinecone sparse index object for upserting
        
    Returns:
        Number of vectors upserted
    """
    vectors = []
    
    # Progress bar for processing rows
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing applications"):
        # Prepare metadata from dataframe row (exclude short_description)
        metadata = {
            'appid': int(row['appid']),
            'name': str(row['name']),
            'release_date': str(row['release_date']),
            'metacritic_score': float(row['metacritic_score']),
            'recommendations_total': int(row['recommendations_total']),
        }

        # Chunk the combined_text for this row
        application_text = row['combined_text'] if pd.notna(row['combined_text']) else ''
        chunks = chunk_text(application_text)
        
        # If no chunks, skip this row
        if not chunks:
            continue
        
        # Create a vector for each chunk
        for chunk_idx, chunk_text_content in enumerate(chunks):
            # Generate unique ID: appid#chunk_idx
            vector_id = f"{row['appid']}#{chunk_idx}"
            
            # Prepare vector record with text for integrated embeddings
            vector_record = {
                '_id': vector_id,
                'text': chunk_text_content,  # For integrated embeddings
            }
            
            # Add metadata to the record
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = chunk_idx
            chunk_metadata['total_chunks'] = len(chunks)
            
            # Add metadata to vector record
            vector_record.update(chunk_metadata)
            vectors.append(vector_record)
    
    # Upsert vectors in batches
    if not vectors:
        return 0
    
    # Batch size for upserting; 96 is the maximum for records that use integrated embeddings
    batch_size = 96
    total_upserted = 0
    total_skipped = 0
    all_skipped_ids = []
    
    # Progress bar for upserting batches
    for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting application vector batches"):
        batch = vectors[i:i + batch_size]
        
        successful, skipped, skipped_ids = _upsert_batch_resilient(
            dense_index, sparse_index, batch, "application"
        )
        total_upserted += successful
        total_skipped += skipped
        all_skipped_ids.extend(skipped_ids)
    
    # Report skipped records if any
    if total_skipped > 0:
        print(f"\nWarning: Skipped {total_skipped} application vectors due to invalid embeddings")
        print(f"Skipped IDs: {', '.join(all_skipped_ids)}")
    
    return total_upserted


def upsert_reviews(df: pd.DataFrame, dense_index: pinecone.Pinecone.Index, sparse_index: pinecone.Pinecone.Index) -> int:
    """
    Upsert review records to Pinecone with chunked descriptions.
    
    For each row in the dataframe:
    - Chunks the combined_text
    - Creates a vector for each chunk (embedded)
    - Includes all other dataframe fields as metadata
    
    Args:
        df: Reviews DataFrame
        dense_index: Pinecone dense index object for upserting
        sparse_index: Pinecone sparse index object for upserting
        
    Returns:
        Number of vectors upserted
    """
    vectors = []
    
    # Progress bar for processing rows
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
        # Prepare metadata from dataframe row (exclude review_text since it's chunked)
        metadata = {
            'appid': int(row['appid']),
            'name': str(row['name']),
            'recommendationid': int(row['recommendationid']),
            'author_num_games_owned': int(row['author_num_games_owned']),
            'author_num_reviews': int(row['author_num_reviews']),
            'author_playtime_forever': int(row['author_playtime_forever']),
            'author_playtime_at_review': int(row['author_playtime_at_review']),
            'weighted_vote_score': float(row['weighted_vote_score']),
            'comment_count': int(row['comment_count']),
            'timestamp_created': str(row['timestamp_created']),
        }

        # Chunk the combined_text for this row
        review_text = row['combined_text'] if pd.notna(row['combined_text']) else ''
        chunks = chunk_text(review_text)
        
        # If no chunks, skip this row
        if not chunks:
            continue
        
        # Create a vector for each chunk
        for chunk_idx, chunk_text_content in enumerate(chunks):
            # Generate unique ID: appid#recommendationid#chunk_idx
            vector_id = f"{row['appid']}#{row['recommendationid']}#{chunk_idx}"
            
            # Prepare vector record with text for integrated embeddings
            vector_record = {
                '_id': vector_id,
                'text': chunk_text_content,  # For integrated embeddings
            }
            
            # Add metadata to the record
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = chunk_idx
            chunk_metadata['total_chunks'] = len(chunks)
            
            # Add metadata to vector record
            vector_record.update(chunk_metadata)
            vectors.append(vector_record)
    
    # Upsert vectors in batches
    if not vectors:
        return 0
    
    # Batch size for upserting; 96 is the maximum for records that use integrated embeddings
    batch_size = 96
    total_upserted = 0
    total_skipped = 0
    all_skipped_ids = []
    
    # Progress bar for upserting batches
    for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting review vector batches"):
        batch = vectors[i:i + batch_size]
        
        successful, skipped, skipped_ids = _upsert_batch_resilient(
            dense_index, sparse_index, batch, "review"
        )
        total_upserted += successful
        total_skipped += skipped
        all_skipped_ids.extend(skipped_ids)
    
    # Report skipped records if any
    if total_skipped > 0:
        print(f"\nWarning: Skipped {total_skipped} review vectors due to invalid embeddings")
        print(f"Skipped IDs: {', '.join(all_skipped_ids)}")
    
    return total_upserted


def main():
    """Execute the database-load command."""
    # Get the Pinecone indexes
    dense_index, sparse_index = utils.get_indexes()

    # Load the Steam dataset CSV files into pandas DataFrames
    applications_df, reviews_df = utils.load_steam_datasets()

    # Transform the dataframes for upserting into Pinecone
    applications_df = applications_transform(applications_df)
    reviews_df = reviews_transform(reviews_df, applications_df)

    # Upsert applications with chunked descriptions
    upserted_app_count = upsert_applications(applications_df, dense_index, sparse_index)
    print(f"Successfully upserted {upserted_app_count} application vectors")

    # Upsert reviews with chunked descriptions
    upserted_rev_count = upsert_reviews(reviews_df, dense_index, sparse_index)
    print(f"Successfully upserted {upserted_rev_count} review vectors")


if __name__ == "__main__":
    main()
