### Min
### Provides text embedding and vector search functionality for MCP server

from typing import Dict, Any, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
from pathlib import Path
import gc


class EmbeddingTool:
    def __init__(self):
        self.model = None
        self.index = None
        self.texts = []
        self.current_model_name = None
        self.batch_size = 100

    @property
    def name(self) -> str:
        return "embedding"

    @property
    def description(self) -> str:
        return "Text embedding and vector search tool for MCP server with batch processing"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of texts to vectorize"
                },
                "model_name": {
                    "type": "string",
                    "default": "all-MiniLM-L6-v2",
                    "description": "Model name to use for embedding"
                },
                "cache_folder": {
                    "type": "string",
                    "default": "./models",
                    "description": "Model cache folder"
                },
                "operation": {
                    "type": "string",
                    "enum": ["embed", "create_index", "search", "batch_embed", "batch_create_index", "save_index",
                             "load_index"],
                    "description": "Operation type to perform"
                },
                "query": {
                    "type": "string",
                    "description": "Query text for similarity search"
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of top results to return"
                },
                "batch_size": {
                    "type": "integer",
                    "default": 100,
                    "description": "Batch size for processing large datasets"
                },
                "save_path": {
                    "type": "string",
                    "description": "Path to save the vector index"
                },
                "load_path": {
                    "type": "string",
                    "description": "Path to load an existing vector index"
                }
            },
            "required": ["operation"]
        }

    def get_tool_info(self) -> Dict[str, Any]:
        ### MCP server needs this to register the tool
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> Optional[str]:
        ### check if inputs are valid before doing anything
        operation = inputs.get("operation")

        if operation in ["embed", "batch_embed"]:
            if "texts" not in inputs:
                return "Missing required parameter: texts"
            if not isinstance(inputs["texts"], list) or len(inputs["texts"]) == 0:
                return "texts must be a non-empty list"

        elif operation in ["create_index", "batch_create_index"]:
            if "texts" not in inputs:
                return "Missing required parameter: texts"
            if not isinstance(inputs["texts"], list) or len(inputs["texts"]) == 0:
                return "texts must be a non-empty list"

        elif operation == "search":
            if "query" not in inputs:
                return "Missing required parameter: query"
            if self.index is None:
                return "No vector index available. Please create index first."

        elif operation == "save_index":
            if "save_path" not in inputs:
                return "Missing required parameter: save_path"
            if self.index is None:
                return "No index to save. Please create index first."

        elif operation == "load_index":
            if "load_path" not in inputs:
                return "Missing required parameter: load_path"

        else:
            return f"Invalid operation: {operation}"

        return None

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ### main entry point - validate first, then do the work
        validation_error = self.validate_inputs(inputs)
        if validation_error:
            return {"error": validation_error}

        operation = inputs["operation"]

        try:
            if operation == "embed":
                return self._embed_texts(inputs)
            elif operation == "batch_embed":
                return self._batch_embed_texts(inputs)
            elif operation == "create_index":
                return self._create_vector_index(inputs)
            elif operation == "batch_create_index":
                return self._batch_create_vector_index(inputs)
            elif operation == "search":
                return self._search_similar(inputs)
            elif operation == "save_index":
                return self._save_index(inputs)
            elif operation == "load_index":
                return self._load_index(inputs)
            else:
                return {"error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"error": f"Operation failed: {str(e)}"}

    def _load_model(self, model_name: str, cache_folder: str = "./models") -> bool:
        ### load the embedding model if not already loaded
        try:
            if self.model is None or model_name != self.current_model_name:
                if self.model is not None:
                    del self.model
                    gc.collect()

                self.model = SentenceTransformer(
                    model_name,
                    cache_folder=cache_folder,
                    use_auth_token=False
                )
                self.current_model_name = model_name
            return True
        except Exception as e:
            return False

    def _embed_texts(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ### turn text into vectors
        texts = inputs["texts"]
        model_name = inputs.get("model_name", "all-MiniLM-L6-v2")
        cache_folder = inputs.get("cache_folder", "./models")

        if not self._load_model(model_name, cache_folder):
            return {"error": f"Failed to load model: {model_name}"}

        embeddings = self.model.encode(texts, convert_to_numpy=True)

        return {
            "success": True,
            "operation": "embed",
            "embeddings": embeddings.tolist(),
            "embedding_dim": embeddings.shape[1],
            "num_texts": len(texts),
            "model_name": model_name
        }

    def _perform_batch_embedding(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        ### process texts in batches to avoid memory issues
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)

            del batch_embeddings
            gc.collect()

        return np.vstack(all_embeddings)

    def _batch_embed_texts(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ### vectorize large text lists in batches
        texts = inputs["texts"]
        model_name = inputs.get("model_name", "all-MiniLM-L6-v2")
        cache_folder = inputs.get("cache_folder", "./models")
        batch_size = inputs.get("batch_size", 100)

        if not self._load_model(model_name, cache_folder):
            return {"error": f"Failed to load model: {model_name}"}

        try:
            embeddings = self._perform_batch_embedding(texts, batch_size)
            return {
                "success": True,
                "operation": "batch_embed",
                "embeddings": embeddings.tolist(),
                "embedding_dim": embeddings.shape[1],
                "num_texts": len(texts),
                "model_name": model_name,
                "batch_size": batch_size,
                "num_batches": (len(texts) + batch_size - 1) // batch_size
            }
        except Exception as e:
            return {"error": f"Failed to batch embed texts: {str(e)}"}

    def _create_vector_index(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ### build index for fast searching later
        texts = inputs["texts"]
        model_name = inputs.get("model_name", "all-MiniLM-L6-v2")
        cache_folder = inputs.get("cache_folder", "./models")

        embed_result = self._embed_texts({
            "texts": texts,
            "model_name": model_name,
            "cache_folder": cache_folder,
            "operation": "embed"
        })

        if not embed_result.get("success"):
            return embed_result

        embeddings = np.array(embed_result["embeddings"])
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))
        self.texts = texts

        return {
            "success": True,
            "operation": "create_index",
            "index_created": True,
            "num_vectors": len(texts),
            "embedding_dim": embeddings.shape[1],
            "model_name": model_name
        }

    def _batch_create_vector_index(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ### build index for large datasets using batch processing
        texts = inputs["texts"]
        model_name = inputs.get("model_name", "all-MiniLM-L6-v2")
        cache_folder = inputs.get("cache_folder", "./models")
        batch_size = inputs.get("batch_size", 100)

        if not self._load_model(model_name, cache_folder):
            return {"error": f"Failed to load model: {model_name}"}

        try:
            first_batch = texts[:min(batch_size, len(texts))]
            first_embeddings = self.model.encode(first_batch, convert_to_numpy=True)
            embedding_dim = first_embeddings.shape[1]

            self.index = faiss.IndexFlatL2(embedding_dim)

            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                all_embeddings.append(batch_embeddings)

                self.index.add(batch_embeddings.astype(np.float32))

                del batch_embeddings
                gc.collect()

            self.texts = texts

            return {
                "success": True,
                "operation": "batch_create_index",
                "index_created": True,
                "num_vectors": len(texts),
                "embedding_dim": embedding_dim,
                "model_name": model_name,
                "batch_size": batch_size,
                "num_batches": (len(texts) + batch_size - 1) // batch_size
            }
        except Exception as e:
            return {"error": f"Failed to batch create vector index: {str(e)}"}

    def _search_similar(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ### find similar texts in the index
        query = inputs["query"]
        top_k = inputs.get("top_k", 5)

        if self.index is None or self.model is None:
            return {"error": "No index or model available. Please create index first."}

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1 and idx < len(self.texts):
                results.append({
                    "rank": i + 1,
                    "text": self.texts[idx],
                    "distance": float(distance),
                    "index": int(idx)
                })

        return {
            "success": True,
            "operation": "search",
            "query": query,
            "top_k": top_k,
            "results": results
        }

    def _save_index(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ### save the current vector index to disk
        try:
            save_path = inputs["save_path"]

            faiss.write_index(self.index, f"{save_path}.faiss")

            metadata = {
                "texts": self.texts,
                "model_name": self.current_model_name,
                "embedding_dim": self.index.d
            }

            with open(f"{save_path}.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "index_saved": True,
                "save_path": save_path,
                "num_vectors": len(self.texts)
            }
        except Exception as e:
            return {"error": f"Failed to save index: {str(e)}"}

    def _load_index(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ### load a vector index from disk
        try:
            load_path = inputs["load_path"]

            self.index = faiss.read_index(f"{load_path}.faiss")

            with open(f"{load_path}.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.texts = metadata["texts"]
            self.current_model_name = metadata["model_name"]

            return {
                "success": True,
                "index_loaded": True,
                "load_path": load_path,
                "num_vectors": len(self.texts),
                "model_name": self.current_model_name,
                "embedding_dim": self.index.d
            }
        except Exception as e:
            return {"error": f"Failed to load index: {str(e)}"}

    def get_status(self) -> Dict[str, Any]:
        ### tell MCP server what's going on with this tool
        return {
            "model_loaded": self.model is not None,
            "current_model": self.current_model_name,
            "index_created": self.index is not None,
            "num_indexed_texts": len(self.texts) if self.texts else 0
        }
### #%#