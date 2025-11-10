# null

# Pinecone quick reference for agents

> **Official docs**: [https://docs.pinecone.io/](https://docs.pinecone.io/) - For complete API reference, advanced features, and detailed guides.

This guide covers critical gotchas, best practices, and common patterns specific to this project. For anything not covered here, consult the official Pinecone documentation.

***

## ⚠️ Critical: Installation & SDK

**ALWAYS use the current SDK:**

```bash  theme={null}
pip install pinecone          # ✅ Correct (current SDK)
pip install pinecone-client   # ❌ WRONG (deprecated, old API)
```

**Current API (2025):**

```python  theme={null}
from pinecone import Pinecone  # ✅ Correct import
```

***

## 🔧 CLI vs SDK: When to Use Which

**Use the Pinecone CLI for:**

* ✅ **Creating indexes** - `pc index create`
* ✅ **Deleting indexes** - `pc index delete`
* ✅ **Configuring indexes** - `pc index configure` (replicas, deletion protection)
* ✅ **Listing indexes** - `pc index list`
* ✅ **Describing indexes** - `pc index describe`
* ✅ **Creating API keys** - `pc api-key create`
* ✅ **One-off inspection** - Checking stats, configuration
* ✅ **Development setup** - All initial infrastructure setup

**Use the Python SDK for:**

* ✅ **Data operations in application code** - upsert, query, search, delete RECORDS
* ✅ **Runtime checks** - `pc.has_index()`, `index.describe_index_stats()`
* ✅ **Automated workflows** - Any data operations that run repeatedly
* ✅ **Production data access** - Reading and writing vectors/records

**❌ NEVER use SDK for:**

* Creating, deleting, or configuring indexes in application code
* One-time administrative tasks

### Installing the Pinecone CLI

**macOS (Homebrew):**

```bash  theme={null}
brew tap pinecone-io/tap
brew install pinecone-io/tap/pinecone

# Upgrade later
brew update && brew upgrade pinecone
```

**Other platforms:**
Download from [GitHub Releases](https://github.com/pinecone-io/cli/releases) (Linux, Windows, macOS)

### CLI Authentication

Choose one method:

**Option 1: User login (recommended for development)**

```bash  theme={null}
pc login
pc target -o "my-org" -p "my-project"
```

**Option 2: API key**

```bash  theme={null}
export PINECONE_API_KEY="your-api-key"
# Or: pc auth configure --global-api-key <api-key>
```

**Option 3: Service account**

```bash  theme={null}
export PINECONE_CLIENT_ID="your-client-id"
export PINECONE_CLIENT_SECRET="your-client-secret"
```

**Full CLI reference:** [https://docs.pinecone.io/reference/cli/command-reference](https://docs.pinecone.io/reference/cli/command-reference)

***

## Quickstarts

> **Important for all quickstarts**: Execute all steps completely. Keep setup minimal (directories, venv, dependencies only). Do not expect the user to satisfy any prerequisites except providing API keys. For summaries, use only README.md and SUMMARY.md.

When you are asked to help get started with Pinecone, ask the user to choose an option:

* Quick test: Create an index, upsert data, and perform semantic search.

* Choose a use case:
  * Search: Build a semantic search system that returns ranked results from your knowledge base. This pattern is ideal for search interfaces where users need a list of relevant documents with confidence scores.

  * RAG: Build a multi-tenant RAG (Retrieval-Augmented Generation) system that retrieves relevant context per tenant and feeds it to an LLM to generate answers. Each tenant (organization, workspace, or user) has isolated data stored in separate Pinecone namespaces. This pattern is ideal for knowledge bases, customer support platforms, and collaborative workspaces.

  * Recommendations: Build a recommendation engine that suggests similar items based on semantic similarity. This pattern is ideal for e-commerce, content platforms, and user personalization systems.

Based on the choice, use the appropriate pattern.

### Setup Prerequisites (all quickstarts)

Before starting any quickstart, complete these steps:

1. **Set up Python environment**: Create project directory, virtual environment, and install Pinecone SDK
2. **Install CLI**: Run `pc version` to check. If not installed: `brew tap pinecone-io/tap && brew install pinecone-io/tap/pinecone` (macOS) or download from [GitHub releases](https://github.com/pinecone-io/cli/releases). If already installed, upgrade: `brew update && brew upgrade pinecone`
3. **Configure API key**: Ask user for Pinecone API key, set as `PINECONE_API_KEY` env variable, then run `pc auth configure --api-key $PINECONE_API_KEY`
4. **For RAG quickstart only**: Also obtain and set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`

### Quick test

Complete [Setup Prerequisites](#setup-prerequisites-all-quickstarts) first.

**Step 1. Implement semantic search**

1. Create an index called "agentic-quickstart-test" with an integrated embedding model that can handle text documents. Use the Pinecone CLI for this. Use the API key env variable to authenticate.

   ```bash  theme={null}
   pc index create -n agentic-quickstart-test -m cosine -c aws -r us-east-1 --model llama-text-embed-v2 --field_map text=content
   ```

2. Prepare a sample dataset of factual statements from different domains like history, physics, technology, and music and upsert the dataset into a new namespace in the index:

   ```python  theme={null}
   from pinecone import Pinecone

   # Initialize Pinecone client
   api_key = os.getenv("PINECONE_API_KEY")
   if not api_key:
       raise ValueError("PINECONE_API_KEY environment variable not set")

   pc = Pinecone(api_key=api_key)

   records = [
       { "_id": "rec1", "content": "The Eiffel Tower was completed in 1889 and stands in Paris, France.", "category": "history" },
       { "_id": "rec2", "content": "Photosynthesis allows plants to convert sunlight into energy.", "category": "science" },
       { "_id": "rec5", "content": "Shakespeare wrote many famous plays, including Hamlet and Macbeth.", "category": "literature" },
       { "_id": "rec7", "content": "The Great Wall of China was built to protect against invasions.", "category": "history" },
       { "_id": "rec15", "content": "Leonardo da Vinci painted the Mona Lisa.", "category": "art" },
       { "_id": "rec17", "content": "The Pyramids of Giza are among the Seven Wonders of the Ancient World.", "category": "history" },
       { "_id": "rec21", "content": "The Statue of Liberty was a gift from France to the United States.", "category": "history" },
       { "_id": "rec26", "content": "Rome was once the center of a vast empire.", "category": "history" },
       { "_id": "rec33", "content": "The violin is a string instrument commonly used in orchestras.", "category": "music" },
       { "_id": "rec38", "content": "The Taj Mahal is a mausoleum built by Emperor Shah Jahan.", "category": "history" },
       { "_id": "rec48", "content": "Vincent van Gogh painted Starry Night.", "category": "art" },
       { "_id": "rec50", "content": "Renewable energy sources include wind, solar, and hydroelectric power.", "category": "energy" }
   ]

   # Target the index
   dense_index = pc.Index("agentic-quickstart-test")

   # Upsert the records into a namespace
   dense_index.upsert_records("example-namespace", records)
   ```

3. Search the dense index for ten records that are most semantically similar to the query, “Famous historical structures and monuments”:

   ```python  theme={null}
   # Wait for the upserted vectors to be indexed
   import time
   time.sleep(10)

   # View stats for the index
   stats = dense_index.describe_index_stats()
   print(stats)

   # Define the query
   query = "Famous historical structures and monuments"

   # Search the dense index
   results = dense_index.search(
       namespace="example-namespace",
       query={
           "top_k": 10,
           "inputs": {
               'text': query
           }
       }
   )

   # Print the results
   for hit in results['result']['hits']:
       print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['content']:<50}")
   ```

4. Show the search results to the user. Most of the results will be about historical structures and monuments. However, a few unrelated statements will be included as well and ranked high in the list, for example, a statement about Shakespeare. Don't show the literal results in your terminal. Print the important result details in the chat.

5. To get a more accurate ranking, search again but this time rerank the initial results based on their relevance to the query:

   ```python  theme={null}
   # Search the dense index and rerank results
   reranked_results = dense_index.search(
       namespace="example-namespace",
       query={
           "top_k": 10,
           "inputs": {
               'text': query
           }
       },
       rerank={
           "model": "bge-reranker-v2-m3",
           "top_n": 10,
           "rank_fields": ["content"]
       }   
   )

   # Print the reranked results
   for hit in reranked_results['result']['hits']:
       print(f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['content']}, category: {hit['fields']['category']}")

   # Access search results
   # IMPORTANT: With reranking, use dict-style access for hit object
   for hit in results.result.hits:
       doc_id = hit["_id"]              # Dict access for id
       score = hit["_score"]            # Dict access for score
       content = hit.fields["content"]  # hit.fields is also a dict
       metadata = hit.fields.get("metadata_field", "")  # Use .get() for optional fields
   ```

6. Show the search results to the user. All of the most relevant results about historical structures and monuments will now be ranked highest. Again, don't show the literal results in your terminal. Print the important result details in the chat.

### Build a semantic search system

Complete [Setup Prerequisites](#setup-prerequisites-all-quickstarts) first.

**Step 1. Build a semantic search system**

1. Create an index called "agentic-quickstart-search" with an integrated embedding model that can handle text documents. Use the Pinecone CLI for this. Use the API key env variable to authenticate.

   ```bash  theme={null}
   pc index create -n agentic-quickstart-search -m cosine -c aws -r us-east-1 --model llama-text-embed-v2 --field_map text=content
   ```

2. Create 20 unique documents with metadata. Each document should cover a unique foundational AI/ML concept.

3. Store the documents in the Pinecone index. Be sure to use the `upsert_records()` method not the `upsert()` method.

4. Create a search function that:

   * Uses semantic search to find relevant documents
   * Includes reranking with the hosted bge-reranker-v2-m3 model
   * Allows filtering by metadata
   * Returns well-formatted results
   * Uses production-ready error handling patterns

   Be sure to use the `search()` method, not the `query()` method.

5. Then search the knowledge base with 3 sample queries.

6. Show the search results to the user. Don't show the literal results in your terminal. Print the important result details in the chat.

7. Provide a summary of what you did including:
   * The production-ready patterns you used
   * A concise explanation of the generated code

### Build a multi-tenant RAG system

Complete [Setup Prerequisites](#setup-prerequisites-all-quickstarts) first (including step 4 for LLM API keys).

This example builds an **Email Management & Search Platform** where each user has isolated access to their own email mailbox—ensuring privacy and data segregation. Each person's email is indexed in its own namespace and they have access only to that namespace.

**Step 1. Build a RAG system**

1. Create an index called "agentic-quickstart-rag" with an integrated embedding model that can handle text documents. Use the Pinecone CLI for this. Use the API key env variable to authenticate.

   ```bash  theme={null}
   pc index create -n agentic-quickstart-rag -m cosine -c aws -r us-east-1 --model llama-text-embed-v2 --field_map text=content
   ```

2. Create 20 unique email messages with metadata across four categories:

   * **Work Correspondence** (5 emails): Project updates, meeting notes, team announcements
   * **Project Management** (5 emails): Task assignments, progress reports, deadline reminders
   * **Client Communications** (5 emails): Client requests, proposals, feedback
   * **Administrative** (5 emails): HR notices, policy updates, expense reports

   Each email should include metadata fields:

   * `message_type`: "work", "project", "client", "admin"
   * `priority`: "high", "medium", "low"
   * `from_domain`: "internal", "client", "vendor"
   * `date_received`: ISO date string
   * `has_attachments`: true or false

3. Store the emails in the Pinecone index using separate namespaces for each user (e.g., `user_alice`, `user_bob`). Be sure to use the `upsert_records()` method not the `upsert()` method.

4. Create a RAG function that:

   * Takes a user query and user identifier as input
   * Searches ONLY the specified user's namespace to ensure data isolation
   * Retrieves relevant emails using semantic search
   * Reranks results with the hosted bge-reranker-v2-m3 model (prioritizing by priority and message\_type)
   * Constructs a prompt with the retrieved email content
   * Sends the prompt to an LLM (use OpenAI GPT-4 or Anthropic Claude)
   * Returns the generated answer with source citations including sender, date, and priority level

   The RAG system should:

   * **Enforce namespace isolation** - never return emails from other users
   * Handle context window limits intelligently
   * Include metadata in citations (message type, date received, priority)
   * Flag high-priority emails in the response
   * Gracefully handle missing or insufficient email context

   Be sure to use the `search()` method, not the `query()` method.

5. Then answer 3 sample questions as a user querying their email mailbox:
   * "What updates did I receive about the quarterly project?"
   * "Show me all client feedback we've received this month"
   * "Find high-priority emails from my team about the presentation"

6. Give the user insight into the process. Show the search results from Pinecone as well as the answers from the LLM. Don't show the literal results and answers in your terminal. Print the important result and asnwer details in the chat.

7. Provide a summary of what you did including:
   * The production-ready patterns you used
   * How namespace isolation ensures privacy and data segregation
   * A concise explanation of the generated code

### Build a recommendation engine

Complete [Setup Prerequisites](#setup-prerequisites-all-quickstarts) first.

**Step 1. Build a recommendation engine**

1. Create an index called "agentic-quickstart-recommendations" with an integrated embedding model that can handle text documents. Use the Pinecone CLI for this. Use the API key env variable to authenticate.

   ```bash  theme={null}
   pc index create -n agentic-quickstart-recommendations -m cosine -c aws -r us-east-1 --model llama-text-embed-v2 --field_map text=content
   ```

2. Create 20 diverse product listings with rich metadata.

3. Store the product listings in the Pinecone index. Be sure to use the `upsert_records()` method not the `upsert()` method.

4. Create a recommendation engine that:

   * Takes a product ID as input and finds similar items.
   * Uses vector similarity to find semantically related products.
   * Allows filtering by category, price range, and other attributes.
   * Implements diversity strategies to limit results per category and score spreading.
   * Aggregates multi-item preferences to generate recommendations.
   * Returns well-formatted recommendations with similarity scores.

   Be sure to use the `search()` method, not the `query()` method.

5. Then test the recommendation engine with 3 sample products.

6. Show the search results to the user. For each test, explain why these recommendations make sense based on the similarity scores and filters. Don't show the literal results in your terminal. Print the important result details in the chat.

7. Provide a summary of what you did including:
   * The production-ready patterns you used
   * A concise explanation of the generated code

***

## Index creation

> **⚠️ Use CLI (`pc index create`), NOT SDK in application code. See [CLI vs SDK](#cli-vs-sdk-when-to-use-which).**

### Index creation with integrated embeddings (preferred)

```python  theme={null}
# Create index with integrated embedding model
index_name = "my-index"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",  # or "gcp", "azure"
        region="us-east-1",  # choose based on location
        embed={
            "model": "llama-text-embed-v2",  # recommended for most cases
            "field_map": {"text": "content"}  # maps search field to record field
        }
    )

index = pc.Index(index_name)
```

### Available embedding models (current)

* `llama-text-embed-v2`: High-performance, configurable dimensions, recommended for most use cases
* `multilingual-e5-large`: For multilingual content, 1024 dimensions
* `pinecone-sparse-english-v0`: For keyword/hybrid search scenarios

## Data operations

### Upserting records (text with integrated embeddings)

```python  theme={null}
# Indexes with integrated embeddings
records = [
    {
        "_id": "doc1",
        "content": "Your text content here",  # must match field_map
        "category": "documentation",
        "created_at": "2025-01-01",
        "priority": "high"
    }
]

# Always use namespaces
namespace = "user_123"  # e.g., "knowledge_base", "session_456"
index.upsert_records(namespace, records)
```

### Updating records

```python Python theme={null}
# Update existing records (use same upsert operation with existing IDs)
updated_records = [
    {
        "_id": "doc1",  # existing record ID
        "content": "Updated content here",
        "category": "updated_docs",  # can change metadata
        "last_modified": "2025-01-15"
    }
]

# Partial updates - only changed fields need to be included
partial_update = [
    {
        "_id": "doc1",
        "category": "urgent",  # only updating category field
        "priority": "high"     # adding new field
    }
]

index.upsert_records(namespace, updated_records)
```

### Fetching records

```python Python theme={null}
# Fetch single record
result = index.fetch(namespace=namespace, ids=["doc1"])
if result.records:
    record = result.records["doc1"]
    print(f"Content: {record.fields.content}")
    print(f"Metadata: {record.metadata}")

# Fetch multiple records
result = index.fetch(namespace=namespace, ids=["doc1", "doc2", "doc3"])
for record_id, record in result.records.items():
    print(f"ID: {record_id}, Content: {record.fields.content}")

# Fetch with error handling
def safe_fetch(index, namespace, ids):
    try:
        result = index.fetch(namespace=namespace, ids=ids)
        return result.records
    except Exception as e:
        print(f"Fetch failed: {e}")
        return {}
```

### Listing record IDs

```python Python theme={null}
# List all record IDs (paginated)
def list_all_ids(index, namespace, prefix=None):
    """List all record IDs with optional prefix filter"""
    all_ids = []
    pagination_token = None

    while True:
        result = index.list(
            namespace=namespace,
            prefix=prefix,  # filter by ID prefix
            limit=1000,
            pagination_token=pagination_token
        )

        all_ids.extend([record.id for record in result.records])

        if not result.pagination or not result.pagination.next:
            break
        pagination_token = result.pagination.next

    return all_ids

# Usage
all_record_ids = list_all_ids(index, "user_123")
docs_only = list_all_ids(index, "user_123", prefix="doc_")
```

***

## Search operations

### Semantic search with reranking (best practice)

```python  theme={null}
def search_with_rerank(index, namespace, query_text, top_k=5):
    """Standard search pattern - always rerank for production"""
    results = index.search(
        namespace=namespace,
        query={
            "top_k": top_k * 2,  # more candidates for reranking
            "inputs": {
                "text": query_text  # must match index config
            }
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": top_k,
            "rank_fields": ["content"]
        }
    )
    return results
```

### Lexical search (keyword-based)

```python Python theme={null}
# Basic lexical search
def lexical_search(index, namespace, query_text, top_k=5):
    """Keyword-based search using sparse embeddings"""
    results = index.search(
        namespace=namespace,
        query={
            "inputs": {"text": query_text},
            "top_k": top_k
        }
    )
    return results

# Lexical search with required terms
def lexical_search_with_required_terms(index, namespace, query_text, required_terms, top_k=5):
    """Results must contain specific required words"""
    results = index.search(
        namespace=namespace,
        query={
            "inputs": {"text": query_text},
            "top_k": top_k,
            "match_terms": required_terms  # results must contain these terms
        }
    )
    return results

# Lexical search with reranking
def lexical_search_with_rerank(index, namespace, query_text, top_k=5):
    """Lexical search with reranking for better relevance"""
    results = index.search(
        namespace=namespace,
        query={
            "inputs": {"text": query_text},
            "top_k": top_k * 2  # get more candidates for reranking
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": top_k,
            "rank_fields": ["content"]
        }
    )
    return results

# Example usage
search_results = lexical_search_with_required_terms(
    index,
    "knowledge_base",
    "machine learning algorithms neural networks",
    required_terms=["algorithms"]  # must contain "algorithms"
)
```

### Metadata filtering

```python  theme={null}
# Simple filters
filter_criteria = {"category": "documentation"}

# Complex filters
filter_criteria = {
    "$and": [
        {"category": {"$in": ["docs", "tutorial"]}},
        {"priority": {"$ne": "low"}},
        {"created_at": {"$gte": "2025-01-01"}}
    ]
}

results = index.search(
    namespace=namespace,
    query={
        "top_k": 10,
        "inputs": {"text": query_text},
        "filter": filter_criteria  # Filter goes inside query object
    }
)
```

### Supported filter operators

* `$eq`: equals
* `$ne`: not equals
* `$gt`, `$gte`: greater than, greater than or equal
* `$lt`, `$lte`: less than, less than or equal
* `$in`: in list
* `$nin`: not in list
* `$exists`: field exists
* `$and`, `$or`: logical operators

***

## 🚨 Common Mistakes (Must Avoid)

### 1. **Nested Metadata** (will cause API errors)

```python  theme={null}
# ❌ WRONG - nested objects not allowed
bad_record = {
    "_id": "doc1",
    "user": {"name": "John", "id": 123},  # Nested
    "tags": [{"type": "urgent"}]  # Nested in list
}

# ✅ CORRECT - flat structure only
good_record = {
    "_id": "doc1",
    "user_name": "John",
    "user_id": 123,
    "tags": ["urgent", "important"]  # String lists OK
}
```

### 2. **Batch Size Limits** (will cause API errors)

```python  theme={null}
# Text records: MAX 96 per batch, 2MB total
# Vector records: MAX 1000 per batch, 2MB total

# ✅ CORRECT - respect limits
for i in range(0, len(records), 96):
    batch = records[i:i + 96]
    index.upsert_records(namespace, batch)
```

### 3. **Missing Namespaces** (causes data isolation issues)

```python  theme={null}
# ❌ WRONG - no namespace
index.upsert_records(records)  # Old API pattern

# ✅ CORRECT - always use namespaces
index.upsert_records("user_123", records)
index.search(namespace="user_123", query=params)
index.delete(namespace="user_123", ids=["doc1"])
```

### 4. **Skipping Reranking** (reduces search quality)

```python  theme={null}
# ⚠️ OK but not optimal
results = index.search(namespace="ns", query={"top_k": 5, "inputs": {"text": "query"}})

# ✅ BETTER - always rerank in production
results = index.search(
    namespace="ns",
    query={"top_k": 10, "inputs": {"text": "query"}},
    rerank={"model": "bge-reranker-v2-m3", "top_n": 5, "rank_fields": ["content"]}
)
```

### 5. **Hardcoded API Keys**

```python  theme={null}
# ❌ WRONG
pc = Pinecone(api_key="pc-abc123...")

# ✅ CORRECT
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
```

***

## Key Constraints

| Constraint          | Limit                                      | Notes                             |
| ------------------- | ------------------------------------------ | --------------------------------- |
| Metadata per record | 40KB                                       | Flat JSON only, no nested objects |
| Text batch size     | 96 records                                 | Also 2MB total per batch          |
| Vector batch size   | 1000 records                               | Also 2MB total per batch          |
| Query response size | 4MB                                        | Per query response                |
| Metadata types      | strings, ints, floats, bools, string lists | No nested structures              |
| Consistency         | Eventually consistent                      | Wait \~1-5s after upsert          |

***

## Error Handling (Production)

### Error Types

* **4xx (client errors)**: Fix your request - DON'T retry (except 429)
* **429 (rate limit)**: Retry with exponential backoff
* **5xx (server errors)**: Retry with exponential backoff

### Simple Retry Pattern

```python  theme={null}
import time
from pinecone.exceptions import PineconeException

def exponential_backoff_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except PineconeException as e:
            status_code = getattr(e, 'status', None)

            # Only retry transient errors
            if status_code and (status_code >= 500 or status_code == 429):
                if attempt < max_retries - 1:
                    delay = min(2 ** attempt, 60)  # Exponential backoff, cap at 60s
                    time.sleep(delay)
                else:
                    raise
            else:
                raise  # Don't retry client errors (4xx except 429)

# Usage
exponential_backoff_retry(lambda: index.upsert_records(namespace, records))
```

***

## Common Operations Cheat Sheet

### Index Management

**⚠️ Important**: For administrative tasks (create, configure, delete indexes), prefer the **Pinecone CLI** over the SDK. Use the SDK only when you need to check index existence or get stats programmatically in your application code.

**Use CLI for these operations:**

```bash  theme={null}
# Create index with integrated embeddings (recommended, one-time setup)
pc index create --name my-index --dimension 1536 --metric cosine \
  --cloud aws --region us-east-1 \
  --model llama-text-embed-v2 \
  --field_map text=content

# Create serverless index without integrated embeddings (if you need custom embeddings)
pc index create-serverless --name my-index --dimension 1536 --metric cosine \
  --cloud aws --region us-east-1

# List indexes
pc index list

# Describe index
pc index describe --name my-index

# Configure index
pc index configure --name my-index --replicas 3

# Delete index
pc index delete --name my-index
```

**Use SDK only for programmatic checks in application code:**

```python  theme={null}
# Check if index exists (in application startup)
if pc.has_index("my-index"):
    index = pc.Index("my-index")

# Get stats (for monitoring/metrics)
stats = index.describe_index_stats()
print(f"Total vectors: {stats.total_vector_count}")
print(f"Namespaces: {list(stats.namespaces.keys())}")
```

**❌ Avoid in application code:**

```python  theme={null}
# Don't create indexes in application code - use CLI instead
pc.create_index(...)  # Use: pc index create ...
pc.create_index_for_model(...)  # Use: pc index create ... (with --model flag)

# Don't delete indexes in application code - use CLI instead
pc.delete_index("my-index")  # Use: pc index delete --name my-index

# Don't configure indexes in application code - use CLI instead
pc.configure_index("my-index", replicas=3)  # Use: pc index configure ...
```

### Data Operations

```python  theme={null}
# Fetch records
result = index.fetch(namespace="ns", ids=["doc1", "doc2"])
for record_id, record in result.vectors.items():
    print(f"{record_id}: {record.values}")

# List all IDs (paginated)
all_ids = []
pagination_token = None
while True:
    result = index.list(namespace="ns", limit=1000, pagination_token=pagination_token)
    all_ids.extend([r['id'] for r in result.vectors])
    if not result.pagination or not result.pagination.next:
        break
    pagination_token = result.pagination.next

# Delete records
index.delete(namespace="ns", ids=["doc1", "doc2"])

# Delete entire namespace
index.delete(namespace="ns", delete_all=True)
```

### Search with Filters

```python  theme={null}
# Metadata filtering - IMPORTANT: Only include "filter" key if you have filters
# Don't set filter to None - omit the key entirely
results = index.search(
    namespace="ns",
    query={
        "top_k": 10,
        "inputs": {"text": "query"},
        "filter": {
            "$and": [
                {"category": {"$in": ["docs", "tutorial"]}},
                {"priority": {"$ne": "low"}},
                {"created_at": {"$gte": "2025-01-01"}}
            ]
        }
    },
    rerank={"model": "bge-reranker-v2-m3", "top_n": 5, "rank_fields": ["content"]}
)

# Search without filters - omit the "filter" key
results = index.search(
    namespace="ns",
    query={
        "top_k": 10,
        "inputs": {"text": "query"}
        # No filter key at all
    },
    rerank={"model": "bge-reranker-v2-m3", "top_n": 5, "rank_fields": ["content"]}
)

# Dynamic filter pattern - conditionally add filter to query dict
query_dict = {
    "top_k": 10,
    "inputs": {"text": "query"}
}
if has_filters:  # Only add filter if it exists
    query_dict["filter"] = {"category": {"$eq": "docs"}}

results = index.search(namespace="ns", query=query_dict, rerank={...})

# Filter operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, $and, $or
```

***

## Recommended Patterns

### Namespace Strategy

```python  theme={null}
# Multi-user apps
namespace = f"user_{user_id}"

# Session-based
namespace = f"session_{session_id}"

# Content-based
namespace = "knowledge_base"
namespace = "chat_history"
```

### Batch Processing

```python  theme={null}
def batch_upsert(index, namespace, records, batch_size=96):
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        exponential_backoff_retry(
            lambda: index.upsert_records(namespace, batch)
        )
        time.sleep(0.1)  # Rate limiting
```

### Environment Config

```python  theme={null}
class PineconeClient:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY required")
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = os.getenv("PINECONE_INDEX", "default-index")

    def get_index(self):
        return self.pc.Index(self.index_name)
```

***

## Embedding Models (2025)

**Integrated embeddings** (recommended - Pinecone handles embedding):

* `llama-text-embed-v2`: High-performance, recommended for most cases
* `multilingual-e5-large`: Multilingual content (1024 dims)
* `pinecone-sparse-english-v0`: Keyword/hybrid search

**Use integrated embeddings** - don't generate vectors manually unless you have a specific reason.

***

## Official Documentation Resources

For advanced features not covered in this quick reference:

* **API reference**: [https://docs.pinecone.io/reference/api/introduction](https://docs.pinecone.io/reference/api/introduction)
* **Bulk imports** (S3/GCS): [https://docs.pinecone.io/guides/index-data/import-data](https://docs.pinecone.io/guides/index-data/import-data)
* **Hybrid search**: [https://docs.pinecone.io/guides/search/hybrid-search](https://docs.pinecone.io/guides/search/hybrid-search)
* **Back ups** (backup/restore): [https://docs.pinecone.io/guides/manage-data/backups-overview](https://docs.pinecone.io/guides/manage-data/backups-overview)
* **Error handling**: [https://docs.pinecone.io/guides/production/error-handling](https://docs.pinecone.io/guides/production/error-handling)
* **Database limits**: [https://docs.pinecone.io/reference/api/database-limits](https://docs.pinecone.io/reference/api/database-limits)
* **Production monitoring**: [https://docs.pinecone.io/guides/production/monitoring](https://docs.pinecone.io/guides/production/monitoring)
* **Python SDK docs**: [https://sdk.pinecone.io/python/index.html](https://sdk.pinecone.io/python/index.html)

***

## Quick Troubleshooting

| Issue                                | Solution                                                      |
| ------------------------------------ | ------------------------------------------------------------- |
| `ModuleNotFoundError: pinecone.grpc` | Wrong SDK - reinstall with `pip install pinecone`             |
| `Metadata too large` error           | Check 40KB limit, flatten nested objects                      |
| `Batch too large` error              | Reduce to 96 records (text) or 1000 (vectors)                 |
| Search returns no results            | Check namespace, wait for indexing (\~5s), verify data exists |
| Rate limit (429) errors              | Implement exponential backoff, reduce request rate            |
| Nested metadata error                | Flatten all metadata - no nested objects allowed              |

***

**Remember**: Always use namespaces, always rerank, always handle errors with retry logic.
