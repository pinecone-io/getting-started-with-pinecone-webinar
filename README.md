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
  2. Create a Python virtual environment: 
     - On macOS/Linux: `python3 -m venv .venv`
     - On Windows: `python -m venv .venv` (or `python3` if available)
  3. Activate the virtual environment: 
     - On macOS/Linux: `source .venv/bin/activate`
     - On Windows: `.\venv\Scripts\activate`
  4. Install Python requirements: `pip install -r requirements.txt`
  
  **Note:** On macOS/Linux, use `python3` and `pip3` before creating the virtual environment. After activating the virtual environment, you can use `python` and `pip` (they will point to the virtual environment versions).
  5. Get the data (choose one method):
     
     **Option A: Automated download (recommended)**
     1. Set up Kaggle API credentials:
        - Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
        - Scroll to 'API' section and click 'Create New Token'
        - Download the `kaggle.json` file
        - Place it at `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables)
     2. Run the download command:
        ```bash
        python3 pc-webinar.py data-download
        ```
     
     **Option B: Manual download**
     1. Download the CSV dataset from Kaggle at: [https://www.kaggle.com/datasets/crainbramp/steam-dataset-2025-multi-modal-gaming-analytics?resource=download-directory&select=steam_dataset_2025_csv_package_v1](https://www.kaggle.com/datasets/crainbramp/steam-dataset-2025-multi-modal-gaming-analytics?resource=download-directory&select=steam_dataset_2025_csv_package_v1)
     2. Unzip the download (should unzip into a folder named `steam_dataset_2025_csv`)
     3. Copy the `steam_dataset_2025_csv` folder to the `data` folder in the root of this repo
  6. Copy `example.env` to `.env`: `cp example.env .env`
  7. Edit `.env` and add your Pinecone API key

### Using Pinecone Database

#### Loading Data
Load Steam game applications and reviews into Pinecone indexes:

```bash
python3 pc-webinar.py database-load
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
python3 pc-webinar.py database-query "action games with good graphics"

# Semantic search - uses only the dense index for semantic similarity
python3 pc-webinar.py database-query "action games with good graphics" --mode semantic

# Lexical search - uses only the sparse index for keyword matching
python3 pc-webinar.py database-query "action games with good graphics" --mode lexical

# Specify number of results to return
python3 pc-webinar.py database-query "strategy games" --top-k 20

# Combine options
python3 pc-webinar.py database-query "indie puzzle games" -m semantic -t 5
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
python3 pc-webinar.py assistant-load
```

This command will:
- Convert the Steam dataset CSV files (applications.csv and reviews.csv) to JSON format and save them in the same directory as the CSV files
- Automatically split large JSON files into 100MB chunks if needed (files are named `applications_part1.json`, `reviews_part1.json`, etc. when split)
- Get or create the Pinecone Assistant (if it doesn't exist)
- Upload all JSON files (including split parts) to the assistant for use in chat completions

**Note:** The CSV files are automatically converted to JSON format before upload, as the Pinecone Assistant doesn't accept CSV files but does accept JSON files. Large files (over 100MB) are automatically split into multiple smaller JSON files to ensure successful uploads.

**Note:** This data load will take several hours:
  * They are big files (reviews.csv has over 1 million records)
  * Large files are automatically split into 100MB chunks to avoid upload size limits
  * The upload automatically creates dense and sparse indexes and uses hybrid search (none of this is visible to the user)
  * Progress bars will show file-level progress for each JSON file being uploaded

#### Prompting the Assistant
Chat with the Pinecone Assistant using the uploaded Steam game data:

```bash
# Use default model (gpt-4o)
python3 pc-webinar.py assistant-prompt "What are the most popular Steam games?"

# Specify a different model using short flag
python3 pc-webinar.py assistant-prompt "Tell me about Counter-Strike's review sentiment" -m claude-3-5-sonnet

# Using the full --model flag
python3 pc-webinar.py assistant-prompt "What games have the best reviews?" --model gemini-2.5-pro
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
| `data-download` | Download Steam dataset from Kaggle | None |
| `database-load` | Load Steam data into Pinecone indexes | None |
| `database-query` | Query Pinecone indexes | `query` (required), `-m/--mode` (optional), `-t/--top-k` (optional) |
| `assistant-load` | Load Steam data into Pinecone Assistant | None |
| `assistant-prompt` | Chat with Pinecone Assistant | `prompt` (required), `-m/--model` (optional) |
