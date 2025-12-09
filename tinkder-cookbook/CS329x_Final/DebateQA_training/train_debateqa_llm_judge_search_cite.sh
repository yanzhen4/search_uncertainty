#!/bin/bash

set -e

echo "=========================================="
echo "🚀 DebateQA Training (LLM Judge - LARGE)"
echo "=========================================="
echo ""

export TINKER_API_KEY="your-tinker-api-key"
export WANDB_API_KEY="your-wandb-api-key"
export GCP_VERTEXAI_PROJECT_NUMBER="your-gcp-project-number"
export GCP_VERTEXAI_REGION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="True"

MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
LORA_RANK=32

JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
JUDGE_MODEL_PATH="tinker://2b33e94d-bdea-4c21-bdb7-2af143b79c8e/sampler_weights/final"

LEARNING_RATE=4e-5
BATCH_SIZE=4
GROUP_SIZE=4
SEED=42
MAX_TOKENS=1024
EVAL_EVERY=0
MAX_TRAJECTORY_TOKENS=8192
QUALITY_THRESHOLD=0.6

QUESTIONS_FILE="CS329x_Final/DebateQA_training/DebateQA_questions_200.jsonl"

MODEL_SHORT=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/\//-/g')
JUDGE_SHORT=$(echo "$JUDGE_MODEL" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M")
RUN_NAME="debateqa_search_cite_${MODEL_SHORT}_bs${BATCH_SIZE}_gs${GROUP_SIZE}_judge-${JUDGE_SHORT}_thresh${QUALITY_THRESHOLD}_seed${SEED}_traj$((MAX_TRAJECTORY_TOKENS/1024))k_lr${LEARNING_RATE}_rank${LORA_RANK}_${TIMESTAMP}"

RUBRIC='Evaluate the answer based on the following aspects:

1. **Search Tool Usage & Strategy**: How effectively did the model use the search tool? High-quality responses demonstrate strategic search behavior:
   - Multiple searches (typically 2-4) covering different aspects of the question
   - Well-formulated queries that target specific information needs
   - Follow-up searches that build upon initial results to fill gaps
   - Appropriate search breadth for complex or multi-faceted questions

2. **Document-Based Reasoning & Citation**: How well does the answer use and cite retrieved documents?
   - Explicit document citations using numbered references (e.g., "According to Document 1...", "Document 2 indicates...")
   - All claims are grounded in specific retrieved documents, not speculation
   - Proper attribution showing which document supports each claim
   - Synthesizes information from multiple documents with clear citations
   - Does NOT make unsupported claims or add information not found in the retrieved documents

3. **Balanced Perspective & Completeness**: For topics with multiple viewpoints:
   - Acknowledges different perspectives from different documents (e.g., "Document 1 argues... while Document 3 contends...")
   - Presents controversial topics fairly by citing multiple document sources
   - Indicates when documents show debates or disagreements
   - Addresses multiple aspects of complex questions using evidence from retrieved documents

Scoring guidance:
- 0.8-1.0: Excellent search strategy, strong document-based reasoning with explicit citations (e.g., "Document 1", "Document 2"), balanced perspective
- 0.6-0.7: Good search usage, mostly document-based with citations, generally balanced
- 0.4-0.5: Adequate searches but weaker citation practice, some unsupported claims or vague references
- 0.2-0.3: Limited search effectiveness, weak document grounding, missing or poor citations
- 0.0-0.1: Poor quality, failed to cite documents properly, or made claims without document support'

SYSTEM_PROMPT='
You are an expert research assistant who uses a Wikipedia search tool to find accurate, well-sourced information. You MUST base your answers on the retrieved documents and cite them properly.

Tool calling: Execute the tool by wrapping calls in <tool_call>...</tool_call>

The search tool you have access to:
```
{
    "name": "search",
    "title": "Wikipedia search",
    "description": "Searches Wikipedia for relevant information based on the given query.",
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
```

How to approach research questions:

1. **Plan your searches**: Think about what information you need. Formulate 1-3 initial search queries that target key aspects of the question.

2. **Execute searches**: Call the search tool with your queries:
   <tool_call>{"name": "search", "args": {"query_list": ["query1", "query2", ...]}}</tool_call>
   
   The tool will return numbered documents (Document 1, Document 2, Document 3, etc.) for each query.

3. **Assess and iterate**: Review the results. Do you have enough information? If not, perform follow-up searches to fill gaps or explore different angles.

4. **Consider multiple perspectives**: For complex or controversial topics (politics, ethics, debates), search for different viewpoints using queries like "arguments for X", "criticisms of X", "perspectives on X".

5. **Answer ONLY based on retrieved documents**: Your answer MUST be grounded in the documents returned by the search tool. Do NOT make claims without document support.

6. **Cite documents by number**: When referencing information, explicitly cite which document(s) you are using. Use formats like:
   - "According to Document 1, ..."
   - "Document 2 indicates that ..."
   - "As stated in Document 3 and Document 5, ..."
   - "Based on the information in Document 1, ..."

7. **Present balanced views**: When documents present multiple perspectives, acknowledge them: "Document 1 argues... while Document 3 contends..." or "Documents 2 and 4 suggest... however, Document 5 points out...".

8. **Format your answer**: Present your final answer after the "Answer:" prefix with proper document citations.

Example workflow:
Question: "Between 2020 and 2025, which year did New York City see the most population growth and how did San Francisco population change in that year?"

Think: I need NYC and SF population data for 2020-2025.
<tool_call>{"name": "search", "args": {"query_list": ["New York City population growth 2020-2025", "San Francisco population change 2020-2025"]}}</tool_call>

[Tool returns: Document 1 with NYC data, Document 2 with SF data]

(After results) Think: Document 1 shows NYC peak was 2024. Need more specific SF 2024 data.
<tool_call>{"name": "search", "args": {"query_list": ["San Francisco population 2024"]}}</tool_call>

[Tool returns: Document 3 with detailed SF 2024 data]

Answer: According to Document 1, New York City saw the most population growth in 2024 with an increase of [X]. Document 3 indicates that San Francisco'\''s population changed by [Y] in 2024, showing [increase/decrease].
'

echo "📋 Configuration:"
echo "  Dataset: DebateQA LARGE (200 questions - ALL used for training)"
echo "  Policy: $MODEL_NAME (LoRA rank $LORA_RANK)"
echo "  Judge: $JUDGE_MODEL"
echo "  Judge Checkpoint: $JUDGE_MODEL_PATH"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE, Group Size: $GROUP_SIZE"
echo "  Seed: $SEED"
echo "  Data: $QUESTIONS_FILE"
echo "  W&B Run: $RUN_NAME"
echo ""

if ! curl -s http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
    echo "⚠️  Chroma DB not running!"
    echo "Start: chroma run --host localhost --port 8000 --path CS329x_Final/DebateQA_training/chroma_db"
    exit 1
fi

if [ ! -f "$QUESTIONS_FILE" ]; then
    echo "❌ Questions file not found: $QUESTIONS_FILE"
    exit 1
fi

echo "🚀 Starting training on DebateQA LARGE dataset (200 questions - ALL for training)..."
echo ""

python -m tinker_cookbook.recipes.tool_use.search.train_llm_judge \
    model_name="$MODEL_NAME" \
    lora_rank=$LORA_RANK \
    judge_model_name="$JUDGE_MODEL" \
    judge_model_path="$JUDGE_MODEL_PATH" \
    judge_rubric="$RUBRIC" \
    system_prompt="$SYSTEM_PROMPT" \
    learning_rate=$LEARNING_RATE \
    batch_size=$BATCH_SIZE \
    group_size=$GROUP_SIZE \
    seed=$SEED \
    max_tokens=$MAX_TOKENS \
    eval_every=$EVAL_EVERY \
    max_trajectory_tokens=$MAX_TRAJECTORY_TOKENS \
    quality_threshold=$QUALITY_THRESHOLD \
    questions_path="$QUESTIONS_FILE" \
    wandb_name="$RUN_NAME"

echo ""
echo "✅ Training complete!"
