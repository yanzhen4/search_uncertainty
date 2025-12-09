# CS329x Final Project: Search-Augmented QA with LLM-as-a-Judge

Training and evaluation of search-augmented question-answering models using LLM-as-a-Judge reinforcement learning on **ResearchyQA** (factual questions) and **DebateQA** (controversial topics).

Our data is available at [data](https://drive.google.com/drive/folders/1XQu-TkRrkjraHDpTXXMIMt_91G3GeP-z?usp=sharing).
## Overview

Models learn to:
- Use a Wikipedia search tool strategically (2-4 searches per question)
- Cite documents explicitly by number (e.g., "According to Document 1...")
- Present balanced perspectives from multiple sources
- Ground all claims in retrieved evidence

An LLM judge (fine-tuned Qwen3-30B) evaluates responses during training based on search strategy, citation quality, and balanced reasoning.

## Quick Start

### Setup
```bash
# Install dependencies
pip install tinker
pip install -e .[dev]

# Set environment variables
export TINKER_API_KEY="your-api-key"
export WANDB_API_KEY="your-wandb-key"
export GCP_VERTEXAI_PROJECT_NUMBER="your-project-number"
export GCP_VERTEXAI_REGION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="True"
```

### Build ChromaDB (one-time setup)
```bash
# For training
cd CS329x_Final/ResearchyQA_training && bash build_chroma.sh
cd CS329x_Final/DebateQA_training && bash build_chroma.sh

# For evaluation
cd CS329x_Final/ResearchyQA_inference && bash build_chroma.sh
cd CS329x_Final/DebateQA_inference && bash build_chroma.sh
```

## Training

### 1. Start Chroma Server (port 8000)
```bash
bash CS329x_Final/ResearchyQA_training/start_server_researchy_train.sh
# OR
bash CS329x_Final/DebateQA_training/start_server_researchy_train.sh
```

### 2. Run Training
```bash
# ResearchyQA (200 questions)
bash CS329x_Final/ResearchyQA_training/train_researchyqa_llm_judge_search_cite.sh

# DebateQA (200 questions)
bash CS329x_Final/DebateQA_training/train_debateqa_llm_judge_search_cite.sh
```

**Key hyperparameters** (edit in scripts):
- Policy model: `Qwen/Qwen3-4B-Instruct-2507`
- Judge model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Learning rate: `4e-5`, Batch size: `4`, Group size: `4`, LoRA rank: `32`
- Quality threshold: `0.6`, Max trajectory tokens: `8192`

## Evaluation

### 1. Start Chroma Server
```bash
# ResearchyQA (port 8001)
bash CS329x_Final/ResearchyQA_inference/start_server_researchy_inference.sh

# DebateQA (port 8001)
bash CS329x_Final/DebateQA_inference/start_server_researchy_inference.sh
```

### 2. Run Evaluation
```bash
# ResearchyQA (100 questions)
bash CS329x_Final/ResearchyQA_inference/eval_researchyqa_llm_judge_search_cite.sh

# DebateQA (100 questions)
bash CS329x_Final/DebateQA_inference/eval_debateqa_llm_judge_search_cite.sh
```

**Set checkpoint in script**:
- `CHECKPOINT="tinker://your-checkpoint-id/sampler_weights/final"` (or `""` for base model)
- `DOC_MAX_LENGTH="2000"` (chars per document, or `""` for full docs)

## Evaluation Rubric

Models are scored 0-1 on three aspects:

1. **Search Strategy**: Multiple targeted searches (2-4), well-formulated queries, iterative refinement
2. **Citation Quality**: Explicit numbered references ("Document 1"), all claims grounded, no speculation
3. **Balanced Perspective**: Multiple viewpoints, acknowledges debates, comprehensive coverage

**Score ranges**:
- 0.8-1.0: Excellent multi-search strategy, strong citations, balanced
- 0.6-0.7: Good search usage, mostly cited, generally balanced
- 0.4-0.5: Adequate searches, weaker citations, some unsupported claims
- 0.2-0.3: Limited search, poor grounding, missing citations
- 0.0-0.1: Poor quality, failed to cite properly

## Output Files

- **Training**: Checkpoints in Tinker cloud, W&B dashboard, `metrics.jsonl`, trajectory logs
- **Evaluation**: 
  - `outputs/eval_results/predictions_llm_judge_YYYYMMDD_HHMMSS.jsonl` (full responses)
  - `outputs/eval_results/metrics_llm_judge_YYYYMMDD_HHMMSS.json` (aggregated scores)

## Troubleshooting

**Chroma server not running?**
```bash
# Check server
curl -s http://localhost:8000/api/v1/heartbeat  # Training/DebateQA inference
curl -s http://localhost:8001/api/v1/heartbeat  # ResearchyQA inference

# Stop servers
ps aux | grep chroma  # Find PID
kill <PID>
```

**ChromaDB not built?**
```bash
cd <dataset_dir>
bash build_chroma.sh
```

## Datasets

- **ResearchyQA**: Factual Wikipedia questions (200 train, 100 eval)
- **DebateQA**: Controversial topics requiring balanced perspectives (200 train, 100 eval)

## Related Code

- `tinker_cookbook/recipes/tool_use/search/train_llm_judge.py` - Training loop
- `tinker_cookbook/recipes/tool_use/search/llm_judge_env.py` - Environment & dataset builder
- `tinker_cookbook/recipes/tool_use/search/search_env.py` - Search environment
