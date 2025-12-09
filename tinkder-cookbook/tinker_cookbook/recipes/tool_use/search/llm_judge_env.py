"""
ResearchyQA Environment with LLM-as-a-Judge Reward

This environment uses an LLM to evaluate the quality of generated responses
based on a rubric, instead of using BLEU score.
"""
import json
import logging
import random
from functools import partial
from pathlib import Path
from typing import Literal, Sequence

import chz
import tinker
from tinker import types
from tinker_cookbook import renderers, model_info
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.recipes.tool_use.search.search_env import SearchEnv
from tinker_cookbook.recipes.tool_use.search.tools import ChromaToolClient, ChromaToolClientConfig
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.tool_use.search.trajectory_logger import (
    get_global_trajectory_logger,
    set_global_trajectory_logger,
    TrajectoryLogger,
)

logger = logging.getLogger(__name__)


class LLMJudgeScorer:
    """
    LLM-based scorer that evaluates response quality using a rubric.
    """
    
    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        rubric: str,
    ):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.rubric = rubric
    
    def _build_judge_prompt(
        self,
        question: str,
        model_answer: str,
        retrieved_context: str | None = None,
        num_searches: int = 0,
    ) -> list[renderers.Message]:
        """Build the prompt for the LLM judge."""
        
        # Build search tracking section
        search_section = f"\n**Number of Searches Performed:** {num_searches}\n"
        
        # Build the evaluation prompt
        context_section = ""
        if retrieved_context:
            context_section = f"\n**Retrieved Context (from searches):**\n{retrieved_context}\n"
        
        prompt = f"""You are evaluating the quality of an answer to a research question.

**Question:**
{question}
{search_section}
{context_section}
**Model's Answer:**
{model_answer}

**Evaluation Rubric:**
{self.rubric}

Based on the rubric, evaluate the quality of the model's answer. 
Output a score between 0.0 and 1.0, where:
- 1.0 = Excellent answer that fully addresses the question with accurate, relevant information
- 0.7-0.9 = Good answer with minor issues
- 0.4-0.6 = Partial answer with significant gaps or inaccuracies
- 0.1-0.3 = Poor answer with major problems
- 0.0 = Completely incorrect or irrelevant

After your reasoning, output exactly one line with the score:
[SCORE] X.X

where X.X is a number between 0.0 and 1.0."""
        
        return [{"role": "user", "content": prompt}]
    
    async def score_answer(
        self,
        question: str,
        model_answer: str,
        retrieved_context: str | None = None,
        num_searches: int = 0,
    ) -> tuple[float, str]:
        """
        Score the model's answer using the LLM judge.
        
        Returns:
            score: Float between 0.0 and 1.0
            explanation: The judge's reasoning
        """
        messages = self._build_judge_prompt(
            question, model_answer, retrieved_context, num_searches
        )
        
        prompt_input = self.renderer.build_generation_prompt(messages)
        
        try:
            response = await self.sampling_client.sample_async(
                prompt_input,
                num_samples=1,
                sampling_params=types.SamplingParams(
                    temperature=0.0,  # Deterministic for consistency
                    max_tokens=1024,
                ),
            )
            
            # Decode with skip_special_tokens to remove <|im_end|>, etc.
            response_text = self.renderer.tokenizer.decode(
                response.sequences[0].tokens, 
                skip_special_tokens=True
            ).strip()
            
            # Extract score from [SCORE] X.X format
            score = 0.0
            if "[SCORE]" in response_text:
                try:
                    score_line = [line for line in response_text.split("\n") if "[SCORE]" in line][0]
                    score_str = score_line.split("[SCORE]")[1].strip()
                    # Clean up any remaining special tokens or whitespace
                    score_str = ''.join(c for c in score_str if c.isdigit() or c == '.')
                    score = float(score_str)
                    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except (IndexError, ValueError) as e:
                    logger.warning(f"Failed to parse score from response '{score_str if 'score_str' in locals() else 'N/A'}': {e}")
                    score = 0.0
            else:
                logger.warning(f"No [SCORE] found in response: {response_text[:100]}")
            
            return score, response_text
            
        except Exception as e:
            logger.error(f"Error calling LLM judge: {e}")
            return 0.0, f"Error: {str(e)}"


class ResearchyQALLMJudgeEnv(SearchEnv):
    """
    Environment for ResearchyQA that uses LLM-as-a-judge for reward.
    
    Instead of BLEU score, this environment uses an LLM to evaluate
    the quality of responses based on a customizable rubric.
    """
    
    def __init__(
        self,
        problem: str,
        chroma_tool_client: ChromaToolClient,
        renderer: renderers.Renderer,
        llm_judge_scorer: LLMJudgeScorer,
        convo_prefix: list[renderers.Message] | None = None,
        max_trajectory_tokens: int = 32 * 1024,
        timeout: float = 1.0,
        max_num_calls: int = 4,
        quality_threshold: float = 0.6,  # Threshold for "good" answer
    ):
        # Parent class requires answer parameter, but we don't use it for LLM judging
        super().__init__(
            problem=problem,
            answer=[],  # Empty list - not used for LLM judge evaluation
            chroma_tool_client=chroma_tool_client,
            renderer=renderer,
            convo_prefix=convo_prefix,
            max_trajectory_tokens=max_trajectory_tokens,
            timeout=timeout,
            max_num_calls=max_num_calls,
        )
        self.llm_judge_scorer = llm_judge_scorer
        self.quality_threshold = quality_threshold
        self.retrieved_context = None  # Store context from search calls
        self.step_count = 0  # Track steps for logging
    
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Override to add trajectory logging."""
        observation, stop_condition = await super().initial_observation()
        
        # Log episode start
        traj_logger = get_global_trajectory_logger()
        if traj_logger:
            try:
                # Get tokens and text from observation
                if hasattr(observation, 'chunks') and len(observation.chunks) > 0:
                    tokens = observation.chunks[0].tokens
                    prompt_text = self.renderer.tokenizer.decode(tokens, skip_special_tokens=False)
                else:
                    tokens = []
                    prompt_text = str(observation)
                
                traj_logger.start_episode(
                    problem=self.problem,
                    initial_prompt_tokens=tokens,
                    initial_prompt_text=prompt_text,
                )
            except Exception as e:
                logger.warning(f"Failed to log episode start: {e}")
        
        return observation, stop_condition
    
    async def call_search_tool(self, tool_call: dict) -> list[renderers.Message]:
        """Override to capture retrieved context."""
        messages = await super().call_search_tool(tool_call)
        
        # Store the retrieved context for the judge
        if messages and "content" in messages[0]:
            self.retrieved_context = messages[0]["content"]
        
        return messages
    
    async def compute_answer_reward(self, sample_str: str) -> tuple[float, dict]:
        """
        Compute reward using LLM judge.
        
        Returns:
            reward: Score from 0.0 to 1.0
            metrics: Dictionary with score and quality check
        """
        model_answer = self._extract_answer(sample_str)
        
        if model_answer is None:
            return 0.0, {
                "llm_judge_score": 0.0,
                "quality_pass": 0.0,
                "num_searches": self.current_num_calls,
            }
        
        # CRITICAL ENFORCEMENT: If no searches were performed, return 0 immediately
        if self.current_num_calls == 0:
            return 0.0, {
                "llm_judge_score": 0.0,
                "quality_pass": 0.0,
                "num_searches": 0,
                "no_search_penalty": 1.0,
            }
        
        # Get LLM judge score WITH search count information
        score, explanation = await self.llm_judge_scorer.score_answer(
            question=self.problem,
            model_answer=model_answer,
            retrieved_context=self.retrieved_context,
            num_searches=self.current_num_calls,
        )
        
        # Binary quality check based on threshold
        quality_pass = float(score >= self.quality_threshold)
        
        # Note: We don't include judge_explanation in metrics because:
        # 1. It's a string, not numeric, so dict_mean() will fail
        # 2. Explanations are logged elsewhere in the training loop
        metrics = {
            "llm_judge_score": score,
            "quality_pass": quality_pass,
            "num_searches": self.current_num_calls,
        }
        
        return score, metrics
    
    def check_answer(self, sample_str: str) -> bool:
        """Check if answer passes quality threshold (for compatibility)."""
        # This is synchronous, so we can't use async here
        # We'll just use a simple check for now
        model_answer = self._extract_answer(sample_str)
        return model_answer is not None
    
    async def step(self, action: str) -> StepResult:
        """
        Step function with LLM judge-based reward.
        
        Overrides the parent step() to use LLM judge instead of exact match.
        """
        self.step_count += 1
        
        # Decode action for logging
        try:
            action_text = self.renderer.tokenizer.decode(action, skip_special_tokens=False)
            logger.debug(f"Step {self.step_count} action (raw): {action_text[:200]}")
        except Exception as e:
            logger.debug(f"Could not decode action: {e}")
            action_text = str(action)[:200]
        
        message, parse_success = self.renderer.parse_response(action)
        
        # Log parsed message details
        logger.debug(f"Parsed message keys: {message.keys()}")
        logger.debug(f"Parse success: {parse_success}")
        has_tool_call = "tool_calls" in message
        tool_call_name = None
        
        if has_tool_call:
            logger.debug(f"Tool calls detected: {message['tool_calls']}")
            if len(message["tool_calls"]) > 0:
                tool_call_name = message["tool_calls"][0].get("name")
        if "content" in message:
            logger.debug(f"Content: {message['content'][:100]}")
        
        self.past_messages.append(message)
        
        if "tool_calls" in message:
            failure_result = StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
            )
            if message["tool_calls"][0]["name"] == "search":
                self.current_num_calls += 1
                if self.current_num_calls > self.max_num_calls:
                    return failure_result
                
                try:
                    tool_return_message = await self.call_search_tool(message["tool_calls"][0])
                    self.past_messages.extend(tool_return_message)
                except Exception as e:
                    logger.error(f"Error calling search tool: {repr(e)}")
                    return failure_result
                
                next_observation = self.renderer.build_generation_prompt(self.past_messages)
                if next_observation.length > self.max_trajectory_tokens:
                    return failure_result
                
                result = StepResult(
                    reward=0.0,
                    episode_done=False,
                    next_observation=self.renderer.build_generation_prompt(self.past_messages),
                    next_stop_condition=self.stop_condition,
                )
                
                # Log this step
                self._log_step(action, action_text, message, parse_success, has_tool_call, tool_call_name, result)
                
                return result
            else:
                # Log this step
                self._log_step(action, action_text, message, parse_success, has_tool_call, tool_call_name, failure_result)
                return failure_result
        else:
            # Final answer - compute LLM judge-based reward
            correct_format = float(parse_success) and float(self.check_format(message["content"]))
            
            # CRITICAL ENFORCEMENT: Strong penalty if no searches were performed
            if self.current_num_calls == 0:
                result = StepResult(
                    reward=-1.0,  # Strong negative reward
                    episode_done=True,
                    next_observation=tinker.ModelInput.empty(),
                    next_stop_condition=self.stop_condition,
                    metrics={
                        "format": correct_format,
                        "correct": 0.0,
                        "llm_judge_score": 0.0,
                        "quality_pass": 0.0,
                        "num_searches": 0,
                        "no_search_penalty": 1.0,
                    },
                )
                
                # Log this step
                self._log_step(action, action_text, message, parse_success, has_tool_call, tool_call_name, result)
                
                return result
            
            # Compute LLM judge score reward
            llm_reward, llm_metrics = await self.compute_answer_reward(message["content"])
            
            # Total reward: format penalty + LLM judge reward
            total_reward = self.format_coef * (correct_format - 1) + llm_reward
            
            result = StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "format": correct_format,
                    "correct": llm_metrics["quality_pass"],  # Binary quality check
                    **llm_metrics,  # Include all LLM judge metrics
                },
            )
            
            # Log this step
            self._log_step(action, action_text, message, parse_success, has_tool_call, tool_call_name, result)
            
            return result
    
    def _log_step(
        self,
        action_tokens,
        action_text: str,
        parsed_message: dict,
        parse_success: bool,
        has_tool_call: bool,
        tool_call_name: str | None,
        result: StepResult,
    ):
        """Helper to log step details to trajectory logger."""
        traj_logger = get_global_trajectory_logger()
        if not traj_logger:
            return
        
        try:
            traj_logger.log_step(
                step_num=self.step_count,
                action_tokens=action_tokens if isinstance(action_tokens, list) else list(action_tokens),
                action_text=action_text,
                parsed_message=parsed_message,
                parse_success=parse_success,
                has_tool_call=has_tool_call,
                tool_call_name=tool_call_name,
                reward=result.reward,
                episode_done=result.episode_done,
                metrics=result.metrics if hasattr(result, 'metrics') else None,
            )
            
            # End episode logging if done
            if result.episode_done:
                traj_logger.end_episode()
        except Exception as e:
            logger.warning(f"Failed to log step: {e}")


class ResearchyQADatum:
    """Data format for ResearchyQA questions."""
    def __init__(self, question: str, question_id: str):
        self.question = question
        self.question_id = question_id


def load_researchyqa_data(
    questions_path: Path,
    seed: int = 0,
) -> list[ResearchyQADatum]:
    """
    Load ResearchyQA dataset from JSONL file.
    
    Args:
        questions_path: Path to ResearchyQA questions file
        seed: Random seed for shuffling
        
    Returns:
        List of ResearchyQADatum objects
    """
    data = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(ResearchyQADatum(
                    question=item["question"],
                    question_id=item["id"]
                ))
    
    # Shuffle data with seed
    random.Random(seed).shuffle(data)
    return data


class ResearchyQALLMJudgeDataset(RLDataset):
    """Dataset for ResearchyQA with LLM judge reward."""
    
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        chroma_tool_client: ChromaToolClient,
        llm_judge_scorer: LLMJudgeScorer,
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
        max_trajectory_tokens: int = 32 * 1024,
        quality_threshold: float = 0.6,
        questions_path: Path | None = None,
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_trajectory_tokens = max_trajectory_tokens
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.chroma_tool_client = chroma_tool_client
        self.llm_judge_scorer = llm_judge_scorer
        self.seed = seed
        self.quality_threshold = quality_threshold
        
        # Default path
        if questions_path is None:
            questions_path = Path(__file__).parent.parent.parent.parent.parent / "ResearchyQA" / "ResearchyQA_questions_100.jsonl"
        
        self.ds = load_researchyqa_data(questions_path, seed=seed)
        logger.info(f"Loaded {len(self.ds)} questions for training")
    
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            self._make_env_group_builder(datum, self.group_size)
            for datum in self.ds[index * self.batch_size : (index + 1) * self.batch_size]
        ]
    
    def __len__(self) -> int:
        return len(self.ds) // self.batch_size
    
    def _make_env_group_builder(self, datum: ResearchyQADatum, group_size: int) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                ResearchyQALLMJudgeEnv,
                datum.question,
                self.chroma_tool_client,
                self.renderer,
                self.llm_judge_scorer,
                convo_prefix=self.convo_prefix,
                max_trajectory_tokens=self.max_trajectory_tokens,
                quality_threshold=self.quality_threshold,
            ),
            num_envs=group_size,
        )


@chz.chz
class ResearchyQALLMJudgeDatasetBuilder(RLDatasetBuilder):
    """Builder for ResearchyQA dataset with LLM judge reward."""
    
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    chroma_tool_config: ChromaToolClientConfig
    
    # LLM Judge configuration
    judge_model_name: str
    judge_model_path: str | None = None  # None for base model
    judge_rubric: str
    
    # System prompt
    system_prompt: str | None = None  # None for default
    
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    max_eval_size: int = 1024
    max_trajectory_tokens: int = 32 * 1024
    quality_threshold: float = 0.6  # Threshold for "good" answer
    questions_path: str | None = None  # Path to questions file
    
    async def __call__(self) -> tuple[ResearchyQALLMJudgeDataset, None]:
        if self.convo_prefix == "standard":
            if self.system_prompt:
                # Use custom system prompt
                convo_prefix = [{"role": "system", "content": self.system_prompt}]
            else:
                # Use default system prompt
                convo_prefix = SearchEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        
        # Policy tokenizer and renderer
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        # Judge tokenizer and renderer
        judge_tokenizer = get_tokenizer(self.judge_model_name)
        judge_renderer_name = model_info.get_recommended_renderer_name(self.judge_model_name)
        judge_renderer = renderers.get_renderer(
            judge_renderer_name,
            tokenizer=judge_tokenizer
        )
        
        # Create judge sampling client
        service_client = tinker.ServiceClient()
        if self.judge_model_path:
            judge_sampling_client = service_client.create_sampling_client(
                model_path=self.judge_model_path
            )
        else:
            judge_sampling_client = service_client.create_sampling_client(
                base_model=self.judge_model_name
            )
        
        # Create LLM judge scorer
        llm_judge_scorer = LLMJudgeScorer(
            sampling_client=judge_sampling_client,
            renderer=judge_renderer,
            rubric=self.judge_rubric,
        )
        
        # Create chroma tool client
        chroma_tool_client = await ChromaToolClient.create(self.chroma_tool_config)
        
        questions_path = Path(self.questions_path) if self.questions_path else None
        
        train_dataset = ResearchyQALLMJudgeDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            chroma_tool_client=chroma_tool_client,
            llm_judge_scorer=llm_judge_scorer,
            convo_prefix=convo_prefix,
            seed=self.seed,
            max_trajectory_tokens=self.max_trajectory_tokens,
            quality_threshold=self.quality_threshold,
            questions_path=questions_path,
        )
        
        return (train_dataset, None)

