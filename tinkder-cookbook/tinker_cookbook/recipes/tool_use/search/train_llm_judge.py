"""
Training script for LLM-as-a-Judge Reward

This script trains a model on datasets using:
- Your local Chroma DB index
- LLM judge to evaluate response quality (instead of BLEU score)
- Customizable rubric for evaluation criteria

Usage:
    python -m tinker_cookbook.recipes.tool_use.search.train_llm_judge \
        model_name=Qwen/Qwen3-4B-Instruct-2507 \
        judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507 \
        log_path=/tmp/llm_judge_run
"""

import asyncio
from datetime import datetime
from pathlib import Path

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.tool_use.search.llm_judge_env import (
    ResearchyQALLMJudgeDatasetBuilder,
    set_global_trajectory_logger,
    TrajectoryLogger,
)
from tinker_cookbook.recipes.tool_use.search.tools import (
    ChromaToolClientConfig,
    EmbeddingConfig,
    RetrievalConfig,
)
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    # Model parameters
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None
    
    # Judge model parameters
    judge_model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    judge_model_path: str | None = None  # None for base model
    judge_rubric: str
    
    # System prompt for the task
    system_prompt: str | None = None  # None for default
    
    # Training parameters
    learning_rate: float = 4e-5
    batch_size: int = 4  # Smaller batch for small dataset
    seed: int = 2
    max_tokens: int = 1024
    eval_every: int = 0
    
    # Dataset parameters
    group_size: int = 4
    max_trajectory_tokens: int = 8 * 1024
    quality_threshold: float = 0.6  # LLM judge score threshold for "good"
    
    # Chroma configuration
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "researchyqa_corpus"
    n_results: int = 3
    embedding_model_name: str = "gemini-embedding-001"
    embedding_dim: int = 768
    
    # ResearchyQA specific
    questions_path: str | None = None  # Auto-detect if None
    
    # Streaming configuration
    stream_minibatch: bool = False
    num_minibatches: int = 4
    
    # Logging parameters
    log_path: str | None = None
    wandb_project: str = "researchyqa-llm-judge"
    wandb_name: str | None = None
    wandb_api_key: str | None = None
    
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    """Main training function."""
    
    # Set W&B API key if provided
    if cli_config.wandb_api_key:
        import os
        os.environ["WANDB_API_KEY"] = cli_config.wandb_api_key
        print(f"✅ W&B API key set")
    
    # Build chroma tool config
    chroma_tool_config = ChromaToolClientConfig(
        chroma_host=cli_config.chroma_host,
        chroma_port=cli_config.chroma_port,
        chroma_collection_name=cli_config.chroma_collection_name,
        retrieval_config=RetrievalConfig(
            n_results=cli_config.n_results,
            embedding_config=EmbeddingConfig(
                model_name=cli_config.embedding_model_name,
                embedding_dim=cli_config.embedding_dim,
            ),
        ),
    )
    
    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    
    # Auto-detect questions path if not provided
    if cli_config.questions_path is None:
        # Try CS329x_Final first, then ResearchyQA
        project_root = Path(__file__).parent.parent.parent.parent.parent
        questions_path = project_root / "CS329x_Final" / "ResearchyQA" / "ResearchyQA_questions_with_answers_100.jsonl"

        if questions_path.exists():
            questions_path = str(questions_path)
        else:
            raise FileNotFoundError(
                f"Questions file not found. Checked:\n"
                f"  - {questions_path}\n"
                f"Please specify the path with: questions_path=/path/to/questions.jsonl"
            )
    else:
        questions_path = cli_config.questions_path
    
    # Verify questions file exists
    if not Path(questions_path).exists():
        raise FileNotFoundError(
            f"Questions file not found: {questions_path}\n"
        )
    
    print(f"Using questions from: {questions_path}")
    print(f"Using judge model: {cli_config.judge_model_name}")
    if cli_config.judge_model_path:
        print(f"Using judge checkpoint: {cli_config.judge_model_path}")
    else:
        print(f"Using base judge model (no fine-tuning)")
    
    # Build dataset builder with LLM judge
    builder = ResearchyQALLMJudgeDatasetBuilder(
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=cli_config.model_name,
        chroma_tool_config=chroma_tool_config,
        judge_model_name=cli_config.judge_model_name,
        judge_model_path=cli_config.judge_model_path,
        judge_rubric=cli_config.judge_rubric,
        system_prompt=cli_config.system_prompt,
        seed=cli_config.seed,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        quality_threshold=cli_config.quality_threshold,
        questions_path=questions_path,
    )
    
    # Configure streaming minibatch
    if cli_config.stream_minibatch:
        stream_minibatch_config = train.StreamMinibatchConfig(
            groups_per_batch=cli_config.batch_size,
            num_minibatches=cli_config.num_minibatches,
        )
        bs_str = f"bs{cli_config.batch_size}_stream"
    else:
        stream_minibatch_config = None
        bs_str = f"bs{cli_config.batch_size}"
    
    # Build run name
    model_name_short = cli_config.model_name.lower().replace("/", "-")
    judge_name_short = cli_config.judge_model_name.split("/")[-1].lower()
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"researchyqa_llmjudge_{model_name_short}_{bs_str}_gs{cli_config.group_size}_"
        f"judge-{judge_name_short}_thresh{cli_config.quality_threshold}_seed{cli_config.seed}_"
        f"traj{cli_config.max_trajectory_tokens // 1024}k_lr{cli_config.learning_rate}_"
        f"rank{cli_config.lora_rank}_{date_and_time}"
    )
    
    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        # Store in project directory by default
        project_root = Path(__file__).parent.parent.parent.parent.parent
        output_dir = project_root / "outputs" / "researchyqa_llm_judge"
        log_path = str(output_dir / run_name)
    
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name
    
    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    
    # Initialize trajectory logger for debugging
    traj_log_path = Path(log_path) / "trajectory_logs"
    traj_logger = TrajectoryLogger(traj_log_path, enabled=True)
    set_global_trajectory_logger(traj_logger)
    print(f"✅ Trajectory logging enabled: {traj_log_path}")
    
    # Build training config
    config = train.Config(
        model_name=cli_config.model_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        stream_minibatch_config=stream_minibatch_config,
    )
    
    print("=" * 80)
    print("🚀 Starting ResearchyQA Training with LLM-as-a-Judge Reward")
    print("=" * 80)
    print(f"Policy Model: {cli_config.model_name}")
    print(f"Judge Model: {cli_config.judge_model_name}")
    print(f"Dataset: ResearchyQA (from {questions_path})")
    print(f"Chroma Collection: {cli_config.chroma_collection_name}")
    print(f"Quality Threshold: {cli_config.quality_threshold}")
    print(f"Batch Size: {cli_config.batch_size}")
    print(f"Group Size: {cli_config.group_size}")
    print(f"Learning Rate: {cli_config.learning_rate}")
    print(f"Log Path: {log_path}")
    print("=" * 80)
    print("\n📋 Evaluation Rubric:")
    print(cli_config.judge_rubric)
    print("=" * 80)
    
    # Run training
    await train.main(config)
    
    # Print trajectory summary
    print("\n" + "=" * 80)
    print("📊 Trajectory Logging Summary")
    print("=" * 80)
    summary = traj_logger.get_summary()
    if summary:
        print(f"Total Episodes Logged: {summary.get('total_episodes', 0)}")
        print(f"Mean Searches per Episode: {summary.get('mean_searches', 0.0):.2f}")
        print(f"Mean Reward: {summary.get('mean_reward', 0.0):.4f}")
    print(f"Trajectory logs saved to: {traj_log_path}")
    print("=" * 80)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))

