#!/bin/bash
# Build ChromaDB for DebateQA

set -e

echo "=========================================="
echo "🏗️  Building ChromaDB Index for DebateQA"
echo "=========================================="

# Set environment variables
export GCP_VERTEXAI_PROJECT_NUMBER="${GCP_VERTEXAI_PROJECT_NUMBER:-your-gcp-project-number}"
export GCP_VERTEXAI_REGION="${GCP_VERTEXAI_REGION:-us-central1}"
export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-True}"

echo "✅ Environment variables set"
echo ""

# Get script directory and CS329x directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CS329X_DIR points to CS329x_Final
CS329X_DIR="$(dirname "$SCRIPT_DIR")"

# Check if corpus exists
CORPUS_PATH="$CS329X_DIR/DebateQA_inference/DebateQA_corpus.jsonl"
if [ ! -f "$CORPUS_PATH" ]; then
    echo "❌ ERROR: Corpus file not found at $CORPUS_PATH"
    exit 1
fi

echo "✅ Found corpus file"
echo ""

# Create chroma_db directory
CHROMA_PATH="$CS329X_DIR/DebateQA_inference/chroma_db"
mkdir -p "$CHROMA_PATH"
echo "✅ Created directory: $CHROMA_PATH"
echo ""

# Build the database
echo "🔨 Building ChromaDB (this will take a few minutes)..."
echo ""

python -c "
import asyncio
import json
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from tinker_cookbook.recipes.tool_use.search.embedding import get_gemini_client, get_gemini_embedding
from tqdm import tqdm

async def build():
    # Load corpus
    corpus_path = Path('$CORPUS_PATH')
    print(f'📂 Loading corpus from {corpus_path}')
    
    documents = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            documents.append(doc)
    
    print(f'✅ Loaded {len(documents)} documents (corpus chunks)')
    
    # Create Chroma client
    chroma_path = Path('$CHROMA_PATH')
    print(f'🗄️  Creating ChromaDB at {chroma_path}')
    
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Create collection
    collection_name = 'debateqa_corpus'
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={'description': 'DebateQA corpus'}
    )
    
    print(f'✅ Created collection: {collection_name}')
    
    # Get Gemini client
    print('🔑 Initializing Gemini client...')
    gemini_client = get_gemini_client()
    
    # Embed in batches
    batch_size = 50
    print(f'🧮 Embedding {len(documents)} documents in batches of {batch_size}...')
    
    for i in tqdm(range(0, len(documents), batch_size), desc='Embedding'):
        batch = documents[i:i + batch_size]
        batch_ids = [doc['id'] for doc in batch]
        batch_contents = [doc['contents'] for doc in batch]
        
        # Get embeddings
        embeddings = await get_gemini_embedding(
            gemini_client,
            batch_contents,
            model='gemini-embedding-001',
            embedding_dim=768,
            task_type='RETRIEVAL_DOCUMENT',
        )
        
        # Add to collection
        collection.add(
            documents=batch_contents,
            embeddings=embeddings,
            ids=batch_ids,
        )
        
        await asyncio.sleep(0.5)  # Rate limiting
    
    # Verify
    count = collection.count()
    print(f'')
    print('='*80)
    print('✅ SUCCESS! ChromaDB built successfully!')
    print('='*80)
    print(f'Collection: {collection_name}')
    print(f'Documents: {count}')
    print(f'Path: {chroma_path}')
    print('='*80)

asyncio.run(build())
"

echo ""
echo "=========================================="
echo "✅ ChromaDB Build Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Start Chroma server:"
echo "     chroma run --host localhost --port 8000 --path $CHROMA_PATH"
echo ""
echo "  2. Run training with DebateQA dataset"
echo ""
