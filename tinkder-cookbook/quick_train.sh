#!/bin/bash
# Quick training script with all environment variables

# Set environment variables
export TINKER_API_KEY="${TINKER_API_KEY:-sk-your-api-key-here}"
export WANDB_API_KEY="${WANDB_API_KEY:-b53a8b344440d37f80e675cd93858227bab887a7}"
export GCP_VERTEXAI_PROJECT_NUMBER="${GCP_VERTEXAI_PROJECT_NUMBER:-24436122332}"
export GCP_VERTEXAI_REGION="${GCP_VERTEXAI_REGION:-us-central1}"
export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-1}"

echo "🚀 Starting ResearchyQA Training (40 steps, ~20 min)"
echo ""

# Run training with medium configuration (40 steps)
python -m tinker_cookbook.recipes.tool_use.search.train_researchyqa \
    batch_size=2 \
    max_tokens=1024 \
    group_size=4 \
    learning_rate=4e-5 \
    lora_rank=32 \
    bleu_threshold=0.3 \
    wandb_project=researchyqa-training

echo ""
echo "✅ Training complete! Run ./quick_eval.sh to evaluate."



