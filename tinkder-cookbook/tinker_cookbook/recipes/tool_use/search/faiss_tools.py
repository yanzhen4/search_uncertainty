"""FAISS-based search tool client for ResearchyQA."""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import chz
import faiss
import numpy as np
from tinker_cookbook.recipes.tool_use.search.embedding import (
    get_gemini_client,
    get_gemini_embedding,
)
from tinker_cookbook.recipes.tool_use.search.tools import (
    EmbeddingConfig,
    RetrievalConfig,
    ToolClientInterface,
)
from tinker_cookbook.renderers import Message, ToolCall

logger = logging.getLogger(__name__)


@chz.chz
class FAISSToolClientConfig:
    index_path: str
    corpus_path: str
    retrieval_config: RetrievalConfig = RetrievalConfig()
    max_retries: int = 10
    initial_retry_delay: int = 1


class FAISSToolClient(ToolClientInterface):
    """Tool client that uses FAISS for vector search."""

    def __init__(
        self,
        index: faiss.Index,
        corpus: dict[str, str],
        gemini_client: Any,
        retrieval_config: RetrievalConfig,
        max_retries: int,
        initial_retry_delay: int,
    ):
        self.index = index
        self.corpus = corpus  # Maps document IDs to content
        self.gemini_client = gemini_client
        self.n_results = retrieval_config.n_results
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.embedding_config = retrieval_config.embedding_config

    @staticmethod
    def create(config: FAISSToolClientConfig) -> "FAISSToolClient":
        """Create a FAISS tool client from config."""
        # Load FAISS index
        logger.info(f"Loading FAISS index from {config.index_path}")
        index = faiss.read_index(config.index_path)
        logger.info(f"Index loaded: {index.ntotal} vectors")

        # Load corpus
        logger.info(f"Loading corpus from {config.corpus_path}")
        corpus = {}
        with open(config.corpus_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                doc = json.loads(line)
                # Store both by index (for FAISS retrieval) and by ID
                corpus[str(idx)] = doc["contents"]
                if "id" in doc:
                    corpus[doc["id"]] = doc["contents"]

        logger.info(f"Corpus loaded: {len(corpus)} documents")

        # Create Gemini client
        gemini_client = get_gemini_client()

        return FAISSToolClient(
            index=index,
            corpus=corpus,
            gemini_client=gemini_client,
            retrieval_config=config.retrieval_config,
            max_retries=config.max_retries,
            initial_retry_delay=config.initial_retry_delay,
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search",
                "title": "Knowledge Base search",
                "description": "Searches a knowledge base for relevant information based on the given query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A list of fully-formed semantic queries. The tool will return search results for each query.",
                        }
                    },
                    "required": ["query_list"],
                },
                "outputSchema": {
                    "type": "string",
                    "description": "The search results in JSON format",
                },
            }
        ]

    async def _get_embeddings_with_retry(self, query_list: list[str]) -> list[list[float]]:
        """Get embeddings from Gemini with retry logic."""
        return await get_gemini_embedding(
            self.gemini_client,
            query_list,
            self.embedding_config.model_name,
            self.embedding_config.embedding_dim,
            self.embedding_config.task_type,
        )

    async def _search_faiss(
        self, query_embeddings: list[list[float]]
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Search FAISS index for nearest neighbors."""
        # Convert to numpy array
        query_array = np.array(query_embeddings, dtype=np.float32)

        # Normalize if using inner product (FAISS convention)
        if isinstance(self.index, faiss.IndexFlatIP):
            faiss.normalize_L2(query_array)

        # Search
        distances, indices = self.index.search(query_array, self.n_results)

        return indices.tolist(), distances.tolist()

    async def invoke(self, tool_call: ToolCall) -> list[Message]:
        """Invoke the search tool with a query."""
        if tool_call["name"] != "search":
            return [Message(role="tool", content=f"Error: Unknown tool '{tool_call['name']}'")]

        if not isinstance(tool_call["args"], dict) or "query_list" not in tool_call["args"]:
            return [
                Message(role="tool", content="Error invoking search tool: query_list is required")
            ]

        query_list = tool_call["args"]["query_list"]
        if (
            not isinstance(query_list, list)
            or len(query_list) == 0
            or not all(isinstance(q, str) and len(q.strip()) > 0 for q in query_list)
        ):
            return [
                Message(
                    role="tool",
                    content="Error: query_list must be a list of non-empty strings",
                )
            ]

        try:
            # Get embeddings
            query_embeddings = await self._get_embeddings_with_retry(query_list)

            # Search FAISS
            indices, distances = await self._search_faiss(query_embeddings)

            # Build response
            message_content = ""
            for query, doc_indices in zip(query_list, indices):
                message_content += f"Query: {query}\n"
                for doc_i, idx in enumerate(doc_indices):
                    # Get document content by index
                    doc_id = str(idx)
                    if doc_id in self.corpus:
                        content = self.corpus[doc_id]
                        message_content += f"Document {doc_i + 1}:\n{content}\n"
                    else:
                        logger.warning(f"Document index {idx} not found in corpus")
                        message_content += f"Document {doc_i + 1}: [Not found]\n"

            return [Message(role="tool", content=message_content)]

        except Exception as e:
            logger.error(f"Error during search: {repr(e)}")
            return [Message(role="tool", content=f"Error during search: {str(e)}")]







