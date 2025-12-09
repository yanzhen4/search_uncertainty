# ====== NEW: question-behavior judges (answer-only & reasoning-only) ======
import json
import re
from typing import List, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from litellm import completion


def llm(messages: List[dict], model: str = "gemini/gemini-2.0-flash") -> str:
    """
    Call LLM API using litellm.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: Model name (e.g., "gemini/gemini-2.0-flash")
    
    Returns:
        Response text from the LLM
    """
    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=0.0,  # Deterministic for evaluation
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""


# ---------- 1. 只看 ANSWER 的 question-related behavior + 相关性 ----------

ANSWER_BEHAVIOR_CRITERIA = """
You are evaluating how a search-based agent behaves with respect to an AMBIGUOUS user question,
based ONLY on its FINAL ANSWER text and the QUESTION (you do NOT see the reasoning chain or documents).
You will infer the QUESTION-RELATED BEHAVIOR from the wording of the final answer itself.

You must output TWO things:

1. question_behavior (categorical)
   Classify how the agent handles potential ambiguity in the question.
   Choose EXACTLY ONE of the following three categories:

   - "clarification_seeking":
       The answer explicitly asks follow-up questions or requests clarification
       to resolve ambiguity in the user question.
       It may highlight multiple possible meanings and ask the user which one they mean,
       instead of committing to a single interpretation.

   - "assumption_making":
       The answer CHOOSES ONE plausible interpretation of the ambiguous question
       and proceeds with that interpretation, without seriously exploring or answering
       alternative interpretations.
       It may or may not briefly mention ambiguity, but the actual answer covers only
       a single interpretation.

   - "multi_interpretation_reasoning":
       The answer actively considers MULTIPLE plausible interpretations of the question
       and responds to more than one of them.
       For example, it might say "If you mean X, then..., but if you mean Y, then...",
       or otherwise provide answers for several interpretations.

2. relevance_score (integer 1–5)
   Score how relevant and appropriate the FINAL ANSWER is to the QUESTION.

   - 5: Directly addresses the user question (including its ambiguity) and provides
        a clear, helpful answer or behavior (clarification, assumptions, or multiple cases).
   - 4: Mostly addresses the question, with minor omissions or small issues but still clearly
        relevant and mostly appropriate.
   - 3: Partially addresses the question, noticeable missing pieces or vagueness,
        but some relevant content.
   - 2: Weakly related to the question, major gaps or unclear, or mixes in irrelevant content.
   - 1: Irrelevant, wrong, or does not answer the question at all.

You MUST return your final decision in STRICT JSON format:

{
  "question_behavior": "<one of: clarification_seeking | assumption_making | multi_interpretation_reasoning>",
  "relevance_score": <an integer from 1 to 5>
}
"""


def llm_answer_behavior_judge(
    question: str,
    answer: str,
    model: str = "gemini/gemini-2.0-flash",
):
    """
    只基于 QUESTION 和 FINAL ANSWER 来判断 question-related behavior + 相关性。

    返回:
    {
      "question_behavior": "clarification_seeking" | "assumption_making" | "multi_interpretation_reasoning",
      "relevance_score": 1-5
    }
    """
    user_prompt = f"""
You will evaluate the agent's FINAL ANSWER.

{ANSWER_BEHAVIOR_CRITERIA}

[QUESTION]
{question}

[FINAL ANSWER]
{answer}

Now output ONLY a JSON object in the exact format specified above.
"""

    messages = [
        {
            "role": "system",
            "content": "You are a strict evaluator. Always follow the required JSON output format.",
        },
        {"role": "user", "content": user_prompt},
    ]

    raw = llm(messages, model=model)

    # ---- 尝试从回复中提取 JSON ----
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    try:
        result = json.loads(json_str)
    except Exception:
        # fallback：防止整个 pipeline 崩
        result = {
            "question_behavior": "assumption_making",
            "relevance_score": 3,
        }

    # 规范化字段
    beh = str(result.get("question_behavior", "assumption_making")).strip().lower()
    if beh not in [
        "clarification_seeking",
        "assumption_making",
        "multi_interpretation_reasoning",
    ]:
        beh = "assumption_making"

    try:
        score = int(result.get("relevance_score", 3))
    except Exception:
        score = 3
    score = max(1, min(5, score))

    return {
        "question_behavior": beh,
        "relevance_score": score,
    }


# ---------- 2. 只看 REASONING 的 question-related behavior + 相关性 ----------

REASONING_BEHAVIOR_CRITERIA = """
You are evaluating how a search-based agent behaves with respect to an AMBIGUOUS user question,
based ONLY on its REASONING CHAIN and the QUESTION (you do NOT see the final answer).
The reasoning chain may include searches, retrieved documents, and intermediate thoughts.

You must output TWO things:

1. question_behavior (categorical)
   Classify how the reasoning handles potential ambiguity in the question.
   Choose EXACTLY ONE of the following three categories:

   - "clarification_seeking":
       The reasoning proposes or plans to ASK THE USER for clarification
       (e.g., to ask a follow-up question), or explicitly decides that clarification
       is needed before proceeding.

   - "assumption_making":
       The reasoning chooses ONE specific interpretation of the ambiguous question
       and proceeds with that interpretation, without seriously pursuing or exploring
       alternative interpretations.

   - "multi_interpretation_reasoning":
       The reasoning actively considers MULTIPLE plausible interpretations,
       explores or evaluates several of them, or branches the reasoning depending on
       different possible meanings of the question.

2. relevance_score (integer 1–5)
   Score how relevant and appropriate the REASONING CHAIN is to the QUESTION
   (i.e., does it focus on steps and considerations that help address the user question and its ambiguity?).

   - 5: Strongly focused on the question and its ambiguity; uses reasoning steps effectively and coherently.
   - 4: Mostly relevant to the question, with minor digressions or small gaps.
   - 3: Partially relevant, some useful steps but also clear noise or missing key reasoning.
   - 2: Weakly related to the question, large parts are irrelevant or confused.
   - 1: Largely irrelevant or incoherent with respect to the question.

You MUST return your final decision in STRICT JSON format:

{
  "question_behavior": "<one of: clarification_seeking | assumption_making | multi_interpretation_reasoning>",
  "relevance_score": <an integer from 1 to 5>
}
"""


def llm_reasoning_behavior_judge(
    question: str,
    reasoning_chain: str,
    model: str = "gemini/gemini-2.0-flash",
):
    """
    只基于 QUESTION 和 REASONING CHAIN 来判断 question-related behavior + 相关性。

    返回:
    {
      "question_behavior": "clarification_seeking" | "assumption_making" | "multi_interpretation_reasoning",
      "relevance_score": 1-5
    }
    """
    user_prompt = f"""
You will evaluate the agent's REASONING CHAIN.

{REASONING_BEHAVIOR_CRITERIA}

[QUESTION]
{question}

[REASONING CHAIN]
{reasoning_chain}

Now output ONLY a JSON object in the exact format specified above.
"""

    messages = [
        {
            "role": "system",
            "content": "You are a strict evaluator. Always follow the required JSON output format.",
        },
        {"role": "user", "content": user_prompt},
    ]

    raw = llm(messages, model=model)

    # ---- 尝试从回复中提取 JSON ----
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    try:
        result = json.loads(json_str)
    except Exception:
        # fallback：防止整个 pipeline 崩
        result = {
            "question_behavior": "assumption_making",
            "relevance_score": 3,
        }

    # 规范化字段
    beh = str(result.get("question_behavior", "assumption_making")).strip().lower()
    if beh not in [
        "clarification_seeking",
        "assumption_making",
        "multi_interpretation_reasoning",
    ]:
        beh = "assumption_making"

    try:
        score = int(result.get("relevance_score", 3))
    except Exception:
        score = 3
    score = max(1, min(5, score))

    return {
        "question_behavior": beh,
        "relevance_score": score,
    }


# ---------- 3. DataFrame 批量版本 ----------

def run_llm_answer_behavior_on_df(
    df: pd.DataFrame,
    model: str = "gemini/gemini-2.0-flash",
    num_threads: int = 8,
) -> Tuple[List[str], List[int]]:
    """
    对 DataFrame 批量跑 answer-only question-related behavior 评估。

    要求 df 有列: 'question', 'answer'  (通常 answer = parsed_agent_answer)

    返回:
        behaviors: List[str]  (question_behavior)
        relevance_scores: List[int]
    """
    required_cols = ["question", "answer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must have column '{col}'")
    
    def process_row(idx, row):
        result = llm_answer_behavior_judge(
            question=str(row["question"]),
            answer=str(row["answer"]),
            model=model,
        )
        return idx, result["question_behavior"], result["relevance_score"]
    
    behaviors = [None] * len(df)
    scores = [None] * len(df)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(process_row, idx, row): idx
            for idx, (_, row) in enumerate(df.iterrows())
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Answer-only judging"):
            try:
                idx, beh, score = future.result()
                behaviors[idx] = beh
                scores[idx] = score
            except Exception as e:
                idx = futures[future]
                print(f"Error processing row {idx}: {e}")
                behaviors[idx] = "assumption_making"
                scores[idx] = 3
    
    return behaviors, scores


def run_llm_reasoning_behavior_on_df(
    df: pd.DataFrame,
    model: str = "gemini/gemini-2.0-flash",
    num_threads: int = 8,
) -> Tuple[List[str], List[int]]:
    """
    对 DataFrame 批量跑 reasoning-only question-related behavior 评估。

    要求 df 有列: 'question', 'reasoning_chain'

    返回:
        behaviors: List[str]  (question_behavior)
        relevance_scores: List[int]
    """
    required_cols = ["question", "reasoning_chain"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must have column '{col}'")
    
    def process_row(idx, row):
        result = llm_reasoning_behavior_judge(
            question=str(row["question"]),
            reasoning_chain=str(row["reasoning_chain"]),
            model=model,
        )
        return idx, result["question_behavior"], result["relevance_score"]
    
    behaviors = [None] * len(df)
    scores = [None] * len(df)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(process_row, idx, row): idx
            for idx, (_, row) in enumerate(df.iterrows())
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reasoning-only judging"):
            try:
                idx, beh, score = future.result()
                behaviors[idx] = beh
                scores[idx] = score
            except Exception as e:
                idx = futures[future]
                print(f"Error processing row {idx}: {e}")
                behaviors[idx] = "assumption_making"
                scores[idx] = 3
    
    return behaviors, scores
