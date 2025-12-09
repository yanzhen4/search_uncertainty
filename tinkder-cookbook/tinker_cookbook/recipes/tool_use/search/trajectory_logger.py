"""
Trajectory logger for debugging search tool training.

Logs detailed information about model behavior during training:
- Initial prompts
- Model responses
- Whether tool calls were detected
- Rewards and metrics
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class TrajectoryLogger:
    """
    Logs training trajectories to JSONL for debugging.
    """
    
    def __init__(self, log_path: str | Path, enabled: bool = True):
        """
        Args:
            log_path: Directory to save trajectory logs
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        if not enabled:
            return
        
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_path / f"trajectories_{timestamp}.jsonl"
        
        logger.info(f"Trajectory logging enabled: {self.log_file}")
        
        # Track current episode - initialize with empty structure
        self.current_episode: Dict[str, Any] = {
            "steps": [],
            "num_searches": 0,
            "total_reward": 0.0,
        }
        self.episode_count = 0
    
    def start_episode(
        self,
        problem: str,
        initial_prompt_tokens: List[int],
        initial_prompt_text: str,
    ):
        """Start logging a new episode."""
        if not self.enabled:
            return
        
        self.episode_count += 1
        self.current_episode = {
            "episode_id": self.episode_count,
            "problem": problem,
            "initial_prompt_tokens": initial_prompt_tokens,
            "initial_prompt_text": initial_prompt_text,
            "steps": [],
            "num_searches": 0,
            "total_reward": 0.0,
            "final_metrics": {},
        }
    
    def log_step(
        self,
        step_num: int,
        action_tokens: List[int],
        action_text: str,
        parsed_message: Dict[str, Any],
        parse_success: bool,
        has_tool_call: bool,
        tool_call_name: str | None,
        reward: float,
        episode_done: bool,
        metrics: Dict[str, Any] | None = None,
    ):
        """Log a single step in the episode."""
        if not self.enabled:
            return
        
        # Safety check: ensure episode is started and has required structure
        if not self.current_episode:
            logger.warning("log_step called before start_episode, initializing minimal episode")
            self.current_episode = {
                "episode_id": self.episode_count + 1,
                "problem": "unknown",
                "steps": [],
                "num_searches": 0,
                "total_reward": 0.0,
            }
        
        if "steps" not in self.current_episode:
            logger.warning("current_episode missing 'steps' key, initializing")
            self.current_episode["steps"] = []
        
        if "num_searches" not in self.current_episode:
            self.current_episode["num_searches"] = 0
        
        if "total_reward" not in self.current_episode:
            self.current_episode["total_reward"] = 0.0
        
        step_info = {
            "step": step_num,
            "action_text": action_text,
            "action_tokens_length": len(action_tokens),
            "parsed_message_keys": list(parsed_message.keys()),
            "parse_success": parse_success,
            "has_tool_call": has_tool_call,
            "tool_call_name": tool_call_name,
            "reward": reward,
            "episode_done": episode_done,
            "metrics": metrics or {},
        }
        
        # Include content if available
        if "content" in parsed_message:
            step_info["content"] = parsed_message["content"][:500]  # Truncate
        
        # Include tool call details if available
        if has_tool_call and "tool_calls" in parsed_message:
            try:
                tool_call = parsed_message["tool_calls"][0]
                step_info["tool_call_details"] = {
                    "name": tool_call.get("name"),
                    "arguments": str(tool_call.get("arguments", {}))[:200],
                }
            except Exception as e:
                step_info["tool_call_error"] = str(e)
        
        self.current_episode["steps"].append(step_info)
        
        # Update episode-level stats
        if has_tool_call and tool_call_name == "search":
            self.current_episode["num_searches"] += 1
        
        self.current_episode["total_reward"] += reward
        
        if metrics:
            self.current_episode["final_metrics"] = metrics
    
    def end_episode(self):
        """Finish logging the current episode and save to file."""
        if not self.enabled:
            return
        
        if not self.current_episode:
            return
        
        # Only write if episode has required fields
        if "episode_id" not in self.current_episode:
            # Reset to safe state
            self.current_episode = {
                "steps": [],
                "num_searches": 0,
                "total_reward": 0.0,
            }
            return
        
        # Write to JSONL
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.current_episode, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write trajectory log: {e}")
        
        # Reset to safe state for next episode
        self.current_episode = {
            "steps": [],
            "num_searches": 0,
            "total_reward": 0.0,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from logged trajectories."""
        if not self.enabled or not self.log_file.exists():
            return {}
        
        try:
            episodes = []
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        episodes.append(json.loads(line))
            
            if not episodes:
                return {}
            
            # Compute statistics
            num_searches = [ep["num_searches"] for ep in episodes]
            total_rewards = [ep["total_reward"] for ep in episodes]
            
            zero_search_episodes = sum(1 for n in num_searches if n == 0)
            
            return {
                "total_episodes": len(episodes),
                "mean_searches": sum(num_searches) / len(num_searches),
                "zero_search_count": zero_search_episodes,
                "zero_search_rate": zero_search_episodes / len(episodes),
                "mean_reward": sum(total_rewards) / len(total_rewards),
            }
        except Exception as e:
            logger.error(f"Failed to compute summary: {e}")
            return {}


# Global logger instance (can be set by training script)
_global_trajectory_logger: TrajectoryLogger | None = None


def set_global_trajectory_logger(logger: TrajectoryLogger):
    """Set the global trajectory logger."""
    global _global_trajectory_logger
    _global_trajectory_logger = logger


def get_global_trajectory_logger() -> TrajectoryLogger | None:
    """Get the global trajectory logger."""
    return _global_trajectory_logger

