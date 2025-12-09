# ResearchyQA Training Scripts

Quick reference for training and evaluating your ResearchyQA model.

## 🎯 Quick Start (3 Steps)

### Step 1: Start Chroma DB
In a **separate terminal**, run:
```bash
cd /Users/yanzhenshen/Desktop/CS329x/tinker-cookbook
chroma run --host localhost --port 8000 --path ResearchyQA/chroma_db
```

Keep this running during training!

### Step 2: Train the model
```bash
./quick_train.sh
```

This will train for 40 steps (~20 minutes).

### Step 3: Evaluate the model
```bash
./quick_eval.sh
```

That's it! 🎉

---

## 📁 Available Scripts

### Training Scripts

#### `quick_train.sh` - Fast & Simple
```bash
./quick_train.sh
```
- **No prompts** - just runs
- **40 steps** (~20 minutes)
- All environment variables included
- Best for: Quick iterations

#### `train_researchyqa.sh` - Interactive
```bash
./train_researchyqa.sh
```
- **Interactive menu** - choose duration
- Options: 20, 40, 80, or 200+ steps
- Checks prerequisites
- Best for: Customized training

### Evaluation Script

#### `quick_eval.sh` - Evaluate Trained Model
```bash
./quick_eval.sh
```
- Loads your latest checkpoint
- Runs on 20 test questions
- Computes BLEU scores
- Saves results to `outputs/eval_results/`

---

## 🔧 Environment Variables

All scripts automatically set these variables:

```bash
# Tinker API (required)
export TINKER_API_KEY="sk-your-api-key-here"

# Weights & Biases (optional)
export WANDB_API_KEY="b53a8b344440d37f80e675cd93858227bab887a7"

# Google Cloud / Vertex AI (for embeddings)
export GCP_VERTEXAI_PROJECT_NUMBER="24436122332"
export GCP_VERTEXAI_REGION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="1"
```

**Note**: If your `TINKER_API_KEY` is already set in your shell, the scripts will use that instead.

---

## 📊 Training Options

### Quick Train (`quick_train.sh`)
- Fixed configuration: 40 steps
- Batch size: 2
- Expected time: ~20 minutes

### Interactive Train (`train_researchyqa.sh`)
Choose from:
1. **Quick test**: 20 steps (~10 min)
2. **Medium run**: 40 steps (~20 min) ← **Recommended**
3. **Long run**: 80 steps (~40 min)
4. **Very long run**: ~200 steps (~90 min)
5. **Custom**: Specify your own parameters

---

## 📈 After Training

### Find Your Checkpoint
```bash
tail -1 outputs/researchyqa/*/checkpoints.jsonl
```

Look for:
```json
{
  "sampler_path": "tinker://[ID]/sampler_weights/final"
}
```

### View Training Metrics
```bash
# Local metrics file
cat outputs/researchyqa/*/metrics.jsonl | tail -5

# Or view in W&B
# https://wandb.ai/your-username/researchyqa-training
```

### Evaluate Your Model
```bash
./quick_eval.sh
```

Or with custom checkpoint:
```bash
python eval_researchyqa.py \
    --checkpoint "tinker://YOUR_ID/sampler_weights/final" \
    --model-name "Qwen/Qwen3-4B-Instruct-2507"
```

---

## 🎯 Training Parameters (Advanced)

If you want to customize training beyond the scripts:

```bash
python -m tinker_cookbook.recipes.tool_use.search.train_researchyqa \
    batch_size=2 \              # Questions per batch (80/batch_size = steps)
    max_tokens=1024 \           # Total token budget
    group_size=4 \              # Rollouts per question
    learning_rate=4e-5 \        # Learning rate
    lora_rank=32 \              # LoRA rank
    bleu_threshold=0.3 \        # Reward threshold
    wandb_project=researchyqa-training
```

### Parameter Effects

| Parameter | Default | Effect |
|-----------|---------|--------|
| `batch_size` | 2 | Smaller = more steps per epoch |
| `max_tokens` | 1024 | Higher = longer training |
| `group_size` | 4 | More rollouts = more diverse |
| `learning_rate` | 4e-5 | Higher = faster learning |
| `bleu_threshold` | 0.3 | Lower = easier rewards |

**Steps per epoch** = 80 questions ÷ `batch_size`

Examples:
- `batch_size=4` → 20 steps per epoch
- `batch_size=2` → 40 steps per epoch
- `batch_size=1` → 80 steps per epoch

---

## 📂 Output Locations

### Training Outputs
```
outputs/researchyqa/
└── researchyqa_[model]_[config]_[timestamp]/
    ├── config.json           # Training configuration
    ├── metrics.jsonl         # Per-step metrics
    ├── checkpoints.jsonl     # Checkpoint paths
    └── [other logs]
```

### Evaluation Outputs
```
outputs/eval_results/
├── metrics_[timestamp].json        # Aggregate metrics
└── predictions_[timestamp].jsonl   # Per-question predictions
```

---

## 🐛 Troubleshooting

### "Connection refused" (Chroma)
**Problem**: Chroma DB is not running

**Solution**:
```bash
# In another terminal
chroma run --host localhost --port 8000 --path ResearchyQA/chroma_db
```

### "Invalid API key"
**Problem**: TINKER_API_KEY not set or incorrect

**Solution**: Set it before running:
```bash
export TINKER_API_KEY=your-real-api-key
./quick_train.sh
```

### "Questions file not found"
**Problem**: ResearchyQA_questions_with_answers_100.jsonl missing

**Solution**: Generate answers first:
```bash
python generate_answers_gpt4o.py
```

### Training BLEU stays at 0
**Problem**: Model isn't learning

**Solutions**:
1. Train longer (use option 3 or 4 in `train_researchyqa.sh`)
2. Lower `bleu_threshold` to 0.2
3. Increase `learning_rate` to 1e-4
4. Check that Chroma DB is working: `python test_researchyqa_setup.py`

---

## 📚 Additional Resources

- **Detailed training guide**: `RESEARCHYQA_TRAINING_GUIDE.md`
- **Evaluation guide**: `EVALUATION_GUIDE.md`
- **Quick eval guide**: `EVALUATION_QUICKSTART.md`
- **Training outputs**: `TRAINING_OUTPUTS_GUIDE.md`

---

## 🎯 Recommended Workflow

### For Quick Testing
```bash
# Terminal 1: Start Chroma
chroma run --host localhost --port 8000 --path ResearchyQA/chroma_db

# Terminal 2: Train & Evaluate
./quick_train.sh      # Train (40 steps, ~20 min)
./quick_eval.sh       # Evaluate
```

### For Better Results
```bash
# Terminal 1: Start Chroma
chroma run --host localhost --port 8000 --path ResearchyQA/chroma_db

# Terminal 2: Longer training
./train_researchyqa.sh   # Choose option 3 or 4 (80-200 steps)
./quick_eval.sh          # Evaluate
```

### For Custom Experiments
```bash
# Edit train_researchyqa.sh or run directly:
python -m tinker_cookbook.recipes.tool_use.search.train_researchyqa \
    batch_size=1 \
    max_tokens=5000 \
    learning_rate=1e-4

# Then evaluate
./quick_eval.sh
```

---

**Happy training!** 🚀



