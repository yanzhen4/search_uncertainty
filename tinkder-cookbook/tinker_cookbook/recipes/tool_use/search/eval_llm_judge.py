"""
Evaluate a trained ResearchyQA model using LLM-as-a-Judge.

This script:
1. Loads a trained checkpoint from Tinker
2. Runs inference on all evaluation data
3. Uses LLM judge to evaluate response quality (same as training)
4. Saves predictions and evaluation results

Usage:
    python CS329x_Final/eval_researchyqa_llm_judge.py \
        --checkpoint "tinker://xxx/sampler_weights/final" \
        --eval-data CS329x_Final/ResearchyQA_inference/ResearchyQA_eval_100.jsonl \
        --output-dir outputs/eval_results
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List
import argparse
from datetime import datetime

import tinker
from tinker_cookbook import renderers, model_info
from tinker_cookbook.recipes.tool_use.search.llm_judge_env import (
    load_researchyqa_data,
    ResearchyQADatum,
    LLMJudgeScorer,
)
from tinker_cookbook.recipes.tool_use.search.search_env import SearchEnv
from tinker_cookbook.recipes.tool_use.search.tools import (
    ChromaToolClient,
    ChromaToolClientConfig,
    EmbeddingConfig,
    RetrievalConfig,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm.asyncio import tqdm as atqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchyQALLMJudgeEvaluator:
    """Evaluator for ResearchyQA using a trained model with LLM judge scoring."""
    
    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        chroma_tool_client: ChromaToolClient,
        llm_judge_scorer: LLMJudgeScorer,
        renderer: renderers.Renderer,
        fewshot_prefix: list,
        quality_threshold: float = 0.6,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        doc_max_length: int = None,
    ):
        self.sampling_client = sampling_client
        self.chroma_tool_client = chroma_tool_client
        self.llm_judge_scorer = llm_judge_scorer
        self.renderer = renderer
        self.quality_threshold = quality_threshold
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fewshot_prefix = fewshot_prefix
        self.doc_max_length = doc_max_length
    
    def _truncate_documents(self, content: str, max_length: int) -> str:
        """
        Truncate document content while preserving structure.
        Truncates each document to max_length characters.
        """
        lines = content.split('\n')
        truncated_lines = []
        current_doc_lines = []
        current_doc_length = 0
        
        for line in lines:
            if line.startswith('Document '):
                # Start of a new document, finalize previous one
                if current_doc_lines:
                    truncated_lines.extend(current_doc_lines)
                    if current_doc_length > max_length:
                        truncated_lines.append(f"... (truncated, {current_doc_length - max_length} chars omitted)")
                
                # Reset for new document
                current_doc_lines = [line]
                current_doc_length = 0
            elif line.startswith('Query:'):
                # Query line, always include
                truncated_lines.append(line)
            else:
                # Document content line
                if current_doc_length + len(line) + 1 <= max_length:
                    current_doc_lines.append(line)
                    current_doc_length += len(line) + 1
                elif current_doc_length < max_length:
                    # Partial line to reach max_length
                    remaining = max_length - current_doc_length
                    current_doc_lines.append(line[:remaining])
                    current_doc_length = max_length
        
        # Finalize last document
        if current_doc_lines:
            truncated_lines.extend(current_doc_lines)
            if current_doc_length > max_length:
                truncated_lines.append(f"... (truncated, {current_doc_length - max_length} chars omitted)")
        
        return '\n'.join(truncated_lines)
    
    async def evaluate_single(self, datum: ResearchyQADatum) -> Dict:
        """
        Evaluate a single question.
        
        Returns a dict with:
            - question: The question
            - question_id: Question ID
            - num_searches: Number of search calls made
            - model_response: Full model response
            - llm_judge_score: LLM judge score (0.0-1.0)
            - judge_explanation: Judge's reasoning (truncated)
            - quality_pass: Whether score >= threshold
            - turns: Number of turns (conversation steps)
            - retrieved_contexts: List of all contexts retrieved from searches (each truncated to 2000 chars)
            - initial_model_input: The initial prompt sent to the model (for debugging)
        """
        # Build initial prompt
        messages = self.fewshot_prefix + [
            {"role": "user", "content": datum.question}
        ]
        
        model_input = self.renderer.build_generation_prompt(messages)
        
        # Save initial model input as readable text for debugging
        # Decode the tokens back to text
        try:
            tokenizer = self.renderer.tokenizer
            if hasattr(model_input, 'chunks') and len(model_input.chunks) > 0:
                # Get tokens from the first chunk
                tokens = model_input.chunks[0].tokens
                initial_model_input = tokenizer.decode(tokens)
            else:
                initial_model_input = str(model_input)
        except Exception as e:
            # Fallback to string representation if decoding fails
            initial_model_input = str(model_input)
        
        # Interactive loop (model can call search tool)
        max_turns = 3  # Reduced from 5 to 3 to prevent context overflow
        turn_count = 0
        num_searches = 0
        final_response = None
        retrieved_contexts = []  # Store all retrieved contexts
        
        for turn in range(max_turns):
            turn_count += 1
            
            # Sample from model
            try:
                sampling_params = tinker.SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=self.renderer.get_stop_sequences(),
                )
                response = await self.sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                )
                assistant_response = response.sequences[0].tokens
            except Exception as e:
                logger.error(f"Sampling failed for question {datum.question_id}: {e}")
                return {
                    "question": datum.question,
                    "question_id": datum.question_id,
                    "num_searches": num_searches,
                    "model_response": "ERROR",
                    "llm_judge_score": 0.0,
                    "judge_explanation": f"Error: {str(e)}",
                    "quality_pass": False,
                    "turns": turn_count,
                    "retrieved_contexts": [],
                    "error": str(e),
                    "initial_model_input": initial_model_input,
                }
            
            # Parse response
            message, parse_success = self.renderer.parse_response(assistant_response)
            messages.append(message)
            
            # Check if it's a tool call
            if parse_success and "tool_calls" in message:
                tool_call = message["tool_calls"][0]
                
                if tool_call["name"] == "search":
                    # Execute search
                    try:
                        # tool_call structure: {'name': 'search', 'args': {'query_list': [...]}}
                        search_results = await self.chroma_tool_client.invoke(tool_call)
                        
                        # Increment search counter
                        num_searches += 1
                        
                        # Store retrieved context for judge (add to list)
                        if search_results and "content" in search_results[0]:
                            retrieved_contexts.append(search_results[0]["content"][:2000])
                        
                        # Truncate documents if doc_max_length is specified
                        if self.doc_max_length is not None and search_results:
                            truncated_results = []
                            for msg in search_results:
                                if msg.get("role") == "tool" and "content" in msg:
                                    # Truncate the content
                                    truncated_content = self._truncate_documents(msg["content"], self.doc_max_length)
                                    truncated_msg = msg.copy()
                                    truncated_msg["content"] = truncated_content
                                    truncated_results.append(truncated_msg)
                                else:
                                    truncated_results.append(msg)
                            search_results = truncated_results
                        
                        # Add tool response to messages
                        messages.extend(search_results)
                        
                        # Build next prompt
                        model_input = self.renderer.build_generation_prompt(messages)
                        
                        continue  # Continue conversation
                    except Exception as e:
                        logger.error(f"Search failed: {e}")
                        final_response = "ERROR: Search failed"
                        break
                else:
                    # Unknown tool
                    final_response = "ERROR: Unknown tool"
                    break
            else:
                # No tool call - this is the final answer
                final_response = message.get("content", "")
                break
        
        if final_response is None:
            final_response = "ERROR: Max turns reached"
        
        # Evaluate with LLM judge (no reference answers needed, use full response)
        try:
            # Combine all retrieved contexts for judge evaluation
            combined_context = "\n\n---\n\n".join(retrieved_contexts) if retrieved_contexts else None
            
            llm_score, judge_explanation = await self.llm_judge_scorer.score_answer(
                question=datum.question,
                model_answer=final_response,
                retrieved_context=combined_context,
                num_searches=num_searches,  # Pass search count to judge
            )
        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            llm_score = 0.0
            judge_explanation = f"Judge evaluation error: {str(e)}"
        
        quality_pass = llm_score >= self.quality_threshold
        
        return {
            "question": datum.question,
            "question_id": datum.question_id,
            "num_searches": num_searches,
            "model_response": final_response,
            "llm_judge_score": llm_score,
            "judge_explanation": judge_explanation[:500] if judge_explanation else None,
            "quality_pass": quality_pass,
            "turns": turn_count,
            "retrieved_contexts": retrieved_contexts,  # List of all retrieved contexts (each up to 2000 chars)
            "initial_model_input": initial_model_input,
        }
    
    async def evaluate_dataset(self, test_data: List[ResearchyQADatum]) -> Dict:
        """
        Evaluate the entire test dataset.
        
        Returns aggregate metrics and per-example results.
        """
        logger.info(f"Evaluating {len(test_data)} test examples...")
        
        # Evaluate all examples
        tasks = [self.evaluate_single(datum) for datum in test_data]
        results = []
        
        for task in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
            result = await task
            results.append(result)
        
        # Compute aggregate metrics
        llm_scores = [r["llm_judge_score"] for r in results]
        quality_pass_count = sum(r["quality_pass"] for r in results)
        turn_counts = [r["turns"] for r in results]
        search_counts = [r["num_searches"] for r in results]
        error_count = sum(1 for r in results if "error" in r)
        zero_search_count = sum(1 for r in results if r["num_searches"] == 0)
        
        metrics = {
            "total_examples": len(results),
            "mean_llm_judge_score": sum(llm_scores) / len(llm_scores) if llm_scores else 0.0,
            "median_llm_judge_score": sorted(llm_scores)[len(llm_scores) // 2] if llm_scores else 0.0,
            "quality_pass_rate": quality_pass_count / len(results),
            "quality_pass_count": quality_pass_count,
            "error_count": error_count,
            "success_rate": (len(results) - error_count) / len(results),
            "mean_turns": sum(turn_counts) / len(turn_counts),
            "mean_searches": sum(search_counts) / len(search_counts) if search_counts else 0.0,
            "median_searches": sorted(search_counts)[len(search_counts) // 2] if search_counts else 0.0,
            "zero_search_count": zero_search_count,
            "zero_search_rate": zero_search_count / len(results),
            "max_llm_score": max(llm_scores) if llm_scores else 0.0,
            "min_llm_score": min(llm_scores) if llm_scores else 0.0,
        }
        
        return {
            "metrics": metrics,
            "results": results,
        }


async def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ResearchyQA model with LLM judge")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Tinker checkpoint path (e.g., tinker://xxx/sampler_weights/final). If not specified, uses base model.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model name",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation data file (JSONL with questions and answers)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval_results",
        help="Directory to save results",
    )
    # Judge parameters
    parser.add_argument(
        "--judge-model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Judge model name",
    )
    parser.add_argument(
        "--judge-checkpoint",
        type=str,
        default=None,
        help="Tinker checkpoint path for judge model (optional, uses base model if not specified)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.6,
        help="LLM judge score threshold for 'quality pass'",
    )
    # Chroma parameters
    parser.add_argument(
        "--chroma-host",
        type=str,
        default="localhost",
        help="Chroma server host",
    )
    parser.add_argument(
        "--chroma-port",
        type=int,
        default=8000,
        help="Chroma server port",
    )
    parser.add_argument(
        "--chroma-collection",
        type=str,
        default="researchyqa_corpus",
        help="Chroma collection name",
    )
    parser.add_argument(
        "--renderer-name",
        type=str,
        default=None,
        help="Renderer name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens per generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )
    parser.add_argument(
        "--judge-rubric",
        type=str,
        default=None,
        help="Custom rubric for LLM judge evaluation (optional)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt for the task (optional)",
    )
    parser.add_argument(
        "--doc-max-length",
        type=int,
        default=None,
        help="Maximum length (in characters) for each retrieved document. If not specified, uses full documents.",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 ResearchyQA Model Evaluation (LLM Judge)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation data
    if args.eval_data is None:
        # Try default eval file
        eval_path_1 = Path("CS329x_Final/ResearchyQA_inference/ResearchyQA_eval_100.jsonl")
        eval_path_2 = Path("ResearchyQA/ResearchyQA_eval_100.jsonl")
        
        if eval_path_1.exists():
            eval_path = eval_path_1
        elif eval_path_2.exists():
            eval_path = eval_path_2
        else:
            raise FileNotFoundError(
                "No eval data specified. Use --eval-data <path>"
            )
    else:
        eval_path = Path(args.eval_data)
    
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval data file not found: {eval_path}")
    
    print(f"\n📂 Loading all eval data from: {eval_path}")
    # Load all evaluation data (no splitting)
    test_data = load_researchyqa_data(eval_path, seed=0)
    
    print(f"   Loaded {len(test_data)} test examples")
    
    # Get renderer name
    renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(args.model_name)
    print(f"\n🔧 Using renderer: {renderer_name}")
    
    # Initialize tokenizer and renderer for policy model
    tokenizer = get_tokenizer(args.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    
    # Initialize judge
    print(f"\n⚖️  Setting up LLM judge: {args.judge_model}")
    if args.judge_checkpoint:
        print(f"   Using judge checkpoint: {args.judge_checkpoint}")
    else:
        print(f"   Using base judge model")
    
    judge_tokenizer = get_tokenizer(args.judge_model)
    judge_renderer_name = model_info.get_recommended_renderer_name(args.judge_model)
    judge_renderer = renderers.get_renderer(judge_renderer_name, tokenizer=judge_tokenizer)
    
    service_client = tinker.ServiceClient()
    if args.judge_checkpoint:
        judge_sampling_client = service_client.create_sampling_client(
            model_path=args.judge_checkpoint,
            base_model=args.judge_model
        )
    else:
        judge_sampling_client = service_client.create_sampling_client(base_model=args.judge_model)
    
    # Use provided rubric or default rubric (same as training)
    if args.judge_rubric:
        rubric = args.judge_rubric
        print(f"   Using custom rubric ({len(rubric)} chars)")
    else:
        print(f"   Using default rubric")
        rubric = """Consider the following aspects when evaluating the answer:

1. **Relevance**: Does the answer directly address the question asked?
2. **Use of Retrieved Context**: Does the answer appropriately incorporate information from the retrieved documents?
3. **Completeness**: Are important aspects of the question covered based on the available context?
4. **Multiple Perspectives**: For questions that may have different viewpoints or be controversial, does the answer acknowledge and reflect multiple aspects or perspectives when appropriate?
5. **Coherence**: Is the information presented in a logical and consistent manner?
6. **Clarity**: Is the answer well-structured and easy to understand?

Prioritize relevance, effective use of retrieved context, and balanced presentation of multiple perspectives for controversial topics."""
    
    llm_judge_scorer = LLMJudgeScorer(
        sampling_client=judge_sampling_client,
        renderer=judge_renderer,
        rubric=rubric,
    )
    
    # Use provided system prompt or default
    if args.system_prompt:
        fewshot_prefix = [{"role": "system", "content": args.system_prompt}]
        print(f"   Using custom system prompt ({len(args.system_prompt)} chars)")
    else:
        fewshot_prefix = SearchEnv.standard_fewshot_prefix()
        print(f"   Using default system prompt")
    
    # Initialize Chroma client
    print(f"\n🔍 Connecting to Chroma DB...")
    chroma_config = ChromaToolClientConfig(
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        chroma_collection_name=args.chroma_collection,
        retrieval_config=RetrievalConfig(
            n_results=3,  # Reduced from 3 to 2 to fit context window
            embedding_config=EmbeddingConfig(
                model_name="gemini-embedding-001",
                embedding_dim=768,
            ),
        ),
    )
    chroma_client = await ChromaToolClient.create(chroma_config)
    print(f"   Connected to collection: {args.chroma_collection}")
    
    # Load sampling client for policy model
    print(f"\n🤖 Loading model...")
    if args.checkpoint:
        print(f"   Using checkpoint: {args.checkpoint}")
        sampling_client = service_client.create_sampling_client(
            model_path=args.checkpoint,
            base_model=args.model_name,
        )
    else:
        print(f"   Using base model (no checkpoint)")
        sampling_client = service_client.create_sampling_client(
            base_model=args.model_name,
        )
    print(f"   Model loaded successfully")
    
    # Create evaluator
    evaluator = ResearchyQALLMJudgeEvaluator(
        sampling_client=sampling_client,
        chroma_tool_client=chroma_client,
        llm_judge_scorer=llm_judge_scorer,
        renderer=renderer,
        fewshot_prefix=fewshot_prefix,
        quality_threshold=args.quality_threshold,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        doc_max_length=args.doc_max_length,
    )
    
    # Run evaluation
    print(f"\n🚀 Running evaluation...")
    print(f"   Temperature: {args.temperature}")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Quality threshold: {args.quality_threshold}")
    if args.doc_max_length:
        print(f"   Document max length: {args.doc_max_length} chars (truncation enabled)")
    else:
        print(f"   Document max length: unlimited (using full documents)")
    print()
    
    eval_results = await evaluator.evaluate_dataset(test_data)
    
    # Display metrics
    print("\n" + "=" * 80)
    print("📊 Evaluation Results")
    print("=" * 80)
    metrics = eval_results["metrics"]
    print(f"Total Examples:          {metrics['total_examples']}")
    print(f"Success Rate:            {metrics['success_rate']:.2%} ({metrics['total_examples'] - metrics['error_count']}/{metrics['total_examples']})")
    if metrics['error_count'] > 0:
        print(f"Errors:                  {metrics['error_count']}")
    print(f"Mean LLM Judge Score:    {metrics['mean_llm_judge_score']:.4f}")
    print(f"Median LLM Judge Score:  {metrics['median_llm_judge_score']:.4f}")
    print(f"LLM Score Range:         [{metrics['min_llm_score']:.4f}, {metrics['max_llm_score']:.4f}]")
    print(f"Quality Pass Rate:       {metrics['quality_pass_rate']:.2%} ({metrics['quality_pass_count']}/{metrics['total_examples']})")
    print(f"Mean Turns:              {metrics['mean_turns']:.2f}")
    print(f"\n🔍 Search Tool Usage:")
    print(f"Mean Searches:           {metrics['mean_searches']:.2f}")
    print(f"Median Searches:         {metrics['median_searches']:.0f}")
    print(f"Zero Search Count:       {metrics['zero_search_count']} ({metrics['zero_search_rate']:.2%})")
    print("=" * 80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    metrics_file = output_dir / f"metrics_llm_judge_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Saved metrics to: {metrics_file}")
    
    # Save detailed results
    results_file = output_dir / f"predictions_llm_judge_{timestamp}.jsonl"
    with open(results_file, "w") as f:
        for result in eval_results["results"]:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"✅ Saved predictions to: {results_file}")
    
    # Show sample predictions
    print("\n" + "=" * 80)
    print("📋 Sample Predictions (first 3)")
    print("=" * 80)
    for i, result in enumerate(eval_results["results"][:3], 1):
        print(f"\n{i}. Question: {result['question'][:100]}...")
        print(f"   Model Response: {(result['model_response'] or 'None')[:200]}...")
        print(f"   LLM Judge Score: {result['llm_judge_score']:.4f} | Quality Pass: {result['quality_pass']} | Turns: {result['turns']} | Searches: {result['num_searches']}")
        if result.get('judge_explanation'):
            print(f"   Judge Explanation: {result['judge_explanation'][:200]}...")
    
    print("\n" + "=" * 80)
    print("✅ Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

