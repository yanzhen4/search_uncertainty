# --- Retrieval server setup for Colab ---

import json
import torch
import faiss
import numpy as np
from typing import List, Optional
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nest_asyncio
import threading
from pyngrok import ngrok
from tqdm import tqdm

nest_asyncio.apply()

# ------------------------------
# Config class
# ------------------------------
class Config:
    def __init__(
        self, 
        retrieval_method: str = "dense", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size

# ------------------------------
# Helper functions
# ------------------------------
def load_corpus(corpus_path: str):
    return load_dataset('json', data_files=corpus_path, split="train")

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]

def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval().cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

# ------------------------------
# Encoder
# ------------------------------
class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.model, self.tokenizer = load_model(model_path, use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        # Prefix for certain models
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            query_list = [prefix + q for q in query_list]
        if "bge" in self.model_name.lower() and is_query:
            query_list = [f"Represent this sentence for searching relevant passages: {q}" for q in query_list]

        inputs = self.tokenizer(query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        output = self.model(**inputs, return_dict=True)
        query_emb = pooling(output.pooler_output, output.last_hidden_state, inputs['attention_mask'], self.pooling_method)
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy().astype(np.float32, order="C")
        del inputs, output
        torch.cuda.empty_cache()
        return query_emb

# ------------------------------
# Dense Retriever
# ------------------------------
class DenseRetriever:
    def __init__(self, config):
        self.config = config
        self.index = faiss.read_index(config.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
        self.corpus = load_corpus(config.corpus_path)
        self.encoder = Encoder(
            model_name=config.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            batch_queries = query_list[start_idx:start_idx+self.batch_size]
            emb = self.encoder.encode(batch_queries)
            batch_scores, batch_idxs = self.index.search(emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i*num:(i+1)*num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)
            del emb, batch_scores, batch_idxs, batch_results
            torch.cuda.empty_cache()
        if return_score:
            return results, scores
        else:
            return results

# ------------------------------
# FastAPI setup
# ------------------------------
app = FastAPI()

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False

