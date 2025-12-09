import json
import pandas as pd
from dotenv import load_dotenv

from judge_utils import (
    run_llm_answer_behavior_on_df,
    run_llm_reasoning_behavior_on_df,
)

load_dotenv()

INPUT_PATH = "parsed_results.json"
OUTPUT_PATH = "parsed_agent_outputs_with_behavior_judge.json"
MODEL_NAME = "gemini/gemini-2.0-flash"  # 你可以按需要换

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    df_raw = pd.DataFrame(data)

    # 1. answer-only judge
    df_answer = pd.DataFrame()
    df_answer["question"] = df_raw["question"]
    df_answer["answer"] = df_raw["parsed_agent_answer"]

    ans_beh, ans_rel = run_llm_answer_behavior_on_df(
        df_answer,
        model=MODEL_NAME,
        num_threads=8,
    )

    # 2. reasoning-only judge
    df_reason = pd.DataFrame()
    df_reason["question"] = df_raw["question"]
    df_reason["reasoning_chain"] = df_raw["reasoning_chain"]

    rea_beh, rea_rel = run_llm_reasoning_behavior_on_df(
        df_reason,
        model=MODEL_NAME,
        num_threads=8,
    )

    # 3. 写回原始 df
    df_raw["answer_source_behavior"] = ans_beh
    df_raw["answer_relevance_score"] = ans_rel
    df_raw["reasoning_source_behavior"] = rea_beh
    df_raw["reasoning_relevance_score"] = rea_rel

    df_raw.to_json(
        OUTPUT_PATH,
        orient="records",
        indent=2,
        force_ascii=False,
    )

    print(
        df_raw[
            [
                "question",
                "parsed_agent_answer",
                "answer_source_behavior",
                "answer_relevance_score",
                "reasoning_source_behavior",
                "reasoning_relevance_score",
            ]
        ].head()
    )

if __name__ == "__main__":
    main()

