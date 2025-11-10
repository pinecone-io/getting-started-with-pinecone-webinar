# getting-started-with-pinecone-webinar
Repository with example code and data used in the monthly Getting Started with Pinecone Webinar.

### Prerequisites
  * Python >=3.9 (built on v3.13.8)
  * Pinecone account ([Sign up here](https://app.pinecone.io/?sessionType=signup))
    * Create a project and copy your API key

### Using Pre-Loaded Environments (Optional)
**Want to try it out without setting up your own environment?**

You can use the shared API key provided in `example.env` to access the pre-loaded database indexes and assistant:
- **API Key**: `pcsk_2go2xm_EAXpMTvVHud6PP3od6iCB5NsCY3PC9smXUQktmh2eVaoDbAhCjqp7Fw5Yqitjqr`
- This API key provides read-only access to:
  - Both database indexes (`getting-started-webinar-dense` and `getting-started-webinar-sparse`) with pre-loaded Steam game data
  - The Pinecone Assistant (`getting-started-webinar-assistant`) with uploaded Steam game data

**Web Interface:** You can also chat with the assistant directly in your browser at [https://getting-started-with-pinecone-webinar.vercel.app/](https://getting-started-with-pinecone-webinar.vercel.app/)

Copy the API key from `example.env` into your `.env` file and you can start querying the database or chatting with the assistant immediately, without running the data load commands.

## Walkthrough

### Getting your environment ready
  1. Clone this repo: `https://github.com/pinecone-io/getting-started-with-pinecone-webinar.git`
  2. Get the data
     1. Download the CSV dataset from Kaggle at: [https://www.kaggle.com/datasets/crainbramp/steam-dataset-2025-multi-modal-gaming-analytics?resource=download-directory&select=steam_dataset_2025_csv_package_v1](https://www.kaggle.com/datasets/crainbramp/steam-dataset-2025-multi-modal-gaming-analytics?resource=download-directory&select=steam_dataset_2025_csv_package_v1)
     2. Unzip the download (should unzip into a folder named `steam_dataset_2025_csv`)
     3. Copy the `steam_dataset_2025_csv` folder to the data folder in the root of this repo
  3. Create a Python virtual environment: `python -m venv .venv`
  4. Activate the virtual environment: `source .venv/bin/activate` (or `.\venv\Scripts\activate` on Windows)
  5. Install Python requirements: `pip install -r requirements.txt`
  6. Copy `example.env` to `.env`: `cp example.env .env`
  7. Edit `.env` and add your Pinecone API key

### Using Pinecone Database

#### Loading Data
Load Steam game applications and reviews into Pinecone indexes:

```bash
python pc-webinar.py database-load
```

This command will:
- Create Pinecone indexes automatically if they don't exist:
  - **Dense index**: Uses `llama-text-embed-v2` model for integrated semantic embeddings
  - **Sparse index**: Uses `pinecone-sparse-english-v0` model integrated keyword embeddings
- Load the Steam dataset CSV files (applications.csv and reviews.csv)
- Transform the data into text for upserting
- Chunk the text using token-based chunking with overlap
- Upsert all chunks with metadata to both dense and sparse indexes using integrated embeddings

**Note:** This data load will take several hours:
  * It's a lot of records (hundreds of thousands of applications and over a million reviews)
  * The upsert uses integrated embedding, which makes embedding model calls for every vector it upserts
  * Progress bars will show real-time progress for both applications and reviews

**Resilience:** The load process is resilient to individual vector failures. If any vectors produce invalid embeddings (e.g., empty sparse vectors), they will be automatically skipped and the process will continue. A warning message will be displayed at the end listing any skipped vector IDs.

**Output:** The command prints the number of vectors successfully upserted for both applications and reviews.

#### Querying the Database
Query the Pinecone database using hybrid, semantic, or lexical search:

```bash
# Hybrid search (default) - queries both dense and sparse indexes, then reranks results
python pc-webinar.py database-query "action games with good graphics"

# Semantic search - uses only the dense index for semantic similarity
python pc-webinar.py database-query "action games with good graphics" --mode semantic

# Lexical search - uses only the sparse index for keyword matching
python pc-webinar.py database-query "action games with good graphics" --mode lexical

# Specify number of results to return
python pc-webinar.py database-query "strategy games" --top-k 20

# Combine options
python pc-webinar.py database-query "indie puzzle games" -m semantic -t 5
```

**Search Modes:**
- **hybrid** (default): Queries both dense and sparse indexes, merges and deduplicates results, then reranks using `cohere-rerank-3.5` for the most relevant results
- **semantic**: Uses only the dense index for semantic similarity search
- **lexical**: Uses only the sparse index for keyword-based search

**Arguments:**
- `query` (required): The search query string
- `-m, --mode` (optional): Search mode - `hybrid`, `semantic`, or `lexical` (default: `hybrid`)
- `-t, --top-k` (optional): Number of results to return (default: `10`)

### Using Pinecone Assistant

#### Loading Data for Assistant
Load Steam game applications and reviews into the Pinecone Assistant:

```bash
python pc-webinar.py assistant-load
```

This command will:
- Convert the Steam dataset CSV files (applications.csv and reviews.csv) to JSON format and save them in the same directory as the CSV files
- Get or create the Pinecone Assistant (if it doesn't exist)
- Upload the JSON files to the assistant for use in chat completions

**Note:** The CSV files are automatically converted to JSON format before upload, as the Pinecone Assistant doesn't accept CSV files but does accept JSON files.

**Note:** This data load will take several hours:
  * They are big files
  * The upload automatically creates dense and sparse indexes and uses hybrid search (none of this is visible to the user)
  * Progress bars will only show if a file has been uploaded or not, progress isn't real-time

#### Prompting the Assistant
Chat with the Pinecone Assistant using the uploaded Steam game data:

```bash
# Use default model (gpt-4o)
python pc-webinar.py assistant-prompt "What are the most popular Steam games?"

# Specify a different model using short flag
python pc-webinar.py assistant-prompt "Tell me about Counter-Strike" -m claude-3-5-sonnet

# Using the full --model flag
python pc-webinar.py assistant-prompt "What games have the best reviews?" --model gemini-2.5-pro
```

**Arguments:**
- `prompt` (required): The prompt/question to ask the assistant
- `-m, --model` (optional): Model to use for the assistant (default: `gpt-4o`)

**Available Models:**
- `gpt-4o` (default)
- `gpt-4.1`
- `o4-mini`
- `claude-3-5-sonnet`
- `claude-3-7-sonnet`
- `gemini-2.5-pro`

## Configuration

The project uses environment variables configured in `.env`:

- `PINECONE_API_KEY` (required): Your Pinecone API key
- `PINECONE_DENSE_INDEX` (optional): Name for the dense index (default: `getting-started-webinar-dense`)
- `PINECONE_SPARSE_INDEX` (optional): Name for the sparse index (default: `getting-started-webinar-sparse`)
- `PINECONE_ASSISTANT_NAME` (optional): Name for the Pinecone Assistant (default: `getting-started-webinar-assistant`)
- `CHUNK_SIZE` (optional): Chunk size in tokens for text processing (default: `1740`)
- `CHUNK_OVERLAP` (optional): Overlap between chunks in tokens (default: `205`)

**Index Configuration:**
- Dense index uses `llama-text-embed-v2` model with integrated embeddings
- Sparse index uses `pinecone-sparse-english-v0` model with integrated embeddings

## Dataset

We are using 2 CSV files from the [Steam Dataset 2025: Multi-Modal Gaming Analytics](https://www.kaggle.com/datasets/crainbramp/steam-dataset-2025-multi-modal-gaming-analytics) dataset from Kaggle.
   * Github repo: [vintagedon/steam-dataset-2025](https://github.com/vintagedon/steam-dataset-2025)

- `applications.csv`: Contains all game application data from Steam
- `reviews.csv`: Contains all game review data from Steam

## Commands Summary

| Command | Description | Arguments |
|---------|-------------|-----------|
| `database-load` | Load Steam data into Pinecone indexes | None |
| `database-query` | Query Pinecone indexes | `query` (required), `-m/--mode` (optional), `-t/--top-k` (optional) |
| `assistant-load` | Load Steam data into Pinecone Assistant | None |
| `assistant-prompt` | Chat with Pinecone Assistant | `prompt` (required), `-m/--model` (optional) |
