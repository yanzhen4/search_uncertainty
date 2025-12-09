# ====== NEW: source-behavior judges (answer-only & reasoning-only) ======
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


# ---------- 1. 只看 ANSWER 的行为 + 相关性 ----------
ANSWER_BEHAVIOR_CRITERIA = """
You are evaluating how a search-based agent behaves with respect to conflicting or multiple sources of information, 
based ONLY on its FINAL ANSWER text and the QUESTION (you do NOT see the reasoning chain or documents).
You will infer the behavior style from the wording of the final answer itself.

You must output THREE things:

1. source_behavior (categorical)
   Choose EXACTLY ONE of the following three categories, inferred from the style of the answer:

   - "non_committal":
       The answer avoids taking a clear stance, is very vague, or refuses to commit
       even though the question invites a concrete answer. It may say things like
       "it depends", "there are many opinions" without clearly explaining or committing.

   - "biased_selection":
       The answer clearly favors one perspective or outcome and presents it as the only view,
       without acknowledging that there may be alternative viewpoints or uncertainty.
       It sounds very one-sided or over-confident given the question.

   - "multi_perspective_reasoning":
       The answer actively acknowledges that there can be multiple viewpoints, scenarios, or interpretations.
       It may mention different sides or conditions (e.g., "On one hand..., on the other hand..."),
       or clearly state that different sources/people might disagree.

2. relevance_score (integer 1–5)
   Score how relevant and appropriate the FINAL ANSWER is to the QUESTION.

   - 5: Directly answers the question, clearly on-topic and informative.
   - 4: Mostly answers the question, minor omissions or small issues but still clearly relevant and mostly correct.
   - 3: Partially answers the question, noticeable missing pieces or vagueness, but some relevant content.
   - 2: Weakly related to the question, major gaps or unclear, or mixes in irrelevant content.
   - 1: Irrelevant, wrong, or does not answer the question at all.

3. question_type (categorical)
   Determine the type of question being asked. Choose EXACTLY ONE of the following:

   - "yes_no": The question expects a yes/no answer (e.g., “Is…?”, “Do…?”, “Should…?”).
   - "factoid": The question asks for a specific factual detail (e.g., “What is…?”, “When…?”, “Where…?”, “Who…?”).
   - "explanation_why": The question asks for a reason or cause (e.g., “Why…?”).
   - "process_how": The question asks for procedures, methods, mechanisms, or steps (e.g., “How do I…?”, “How does… work?”).
   - "opinion": The question asks for personal judgment or subjective evaluation.
   - "instruction": The question instructs the model to perform a task (write/translate/summarize/generate etc.).
   - "comparison": The question asks to compare or contrast items.
   - "other": The question does not fit into the above categories.

You MUST return your final decision in STRICT JSON format:

{
  "source_behavior": "<one of: non_committal | biased_selection | multi_perspective_reasoning>",
  "relevance_score": <an integer from 1 to 5>,
  "question_type": "<one of: yes_no | factoid | explanation_why | process_how | opinion | instruction | comparison | other>"
}
"""



def llm_answer_behavior_judge(
    question: str,
    answer: str,
    model: str = "gemini/gemini-2.0-flash",
):
    """
    只基于 QUESTION 和 FINAL ANSWER 来判断行为 + 相关性。

    返回:
    {
      "source_behavior": "non_committal" | "biased_selection" | "multi_perspective_reasoning",
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
            "source_behavior": "non_committal",
            "relevance_score": 3,
        }

    # 规范化字段
    beh = str(result.get("source_behavior", "non_committal")).strip().lower()
    if beh not in [
        "non_committal",
        "biased_selection",
        "multi_perspective_reasoning",
    ]:
        beh = "non_committal"

    try:
        score = int(result.get("relevance_score", 3))
    except Exception:
        score = 3
    score = max(1, min(5, score))

    return {
        "source_behavior": beh,
        "relevance_score": score,
    }


# ---------- 2. 只看 REASONING 的行为 + 相关性 ----------
REASONING_BEHAVIOR_CRITERIA = """
You are evaluating how a search-based agent behaves with respect to conflicting or multiple sources of information,
based ONLY on its REASONING CHAIN and the QUESTION.
The reasoning chain may include searches, retrieved documents, and intermediate thoughts.

You must output TWO things:

1. source_behavior (categorical)
   Choose EXACTLY ONE of the following three categories, based on how the reasoning uses sources:

   - "non_committal":
       The reasoning repeatedly avoids committing to a conclusion, stays very vague,
       or keeps deferring judgment even when evidence is available.
       It does not clearly resolve or weigh the evidence toward any direction.

   - "biased_selection":
       The reasoning selectively focuses on one document or one side of the evidence,
       while ignoring or downplaying other conflicting documents or viewpoints.
       It may cherry-pick convenient evidence and disregard the rest.

   - "multi_perspective_reasoning":
       The reasoning actively acknowledges and engages with multiple documents or viewpoints,
       especially when they conflict. It compares them, explains their differences,
       and tries to synthesize or weigh them to reach a more nuanced conclusion.

2. relevance_score (integer 1–5)
   Score how relevant and appropriate the REASONING CHAIN is to the QUESTION
   (i.e., does it focus on information and steps that help answer the question?).

   - 5: Strongly focused on the question, uses retrieved information effectively and coherently.
   - 4: Mostly relevant to the question, with minor digressions or small gaps.
   - 3: Partially relevant, some useful steps but also clear noise or missing key reasoning.
   - 2: Weakly related to the question, large parts are irrelevant or confused.
   - 1: Largely irrelevant or incoherent with respect to the question.

You MUST return your final decision in STRICT JSON format:

{
  "source_behavior": "<one of: non_committal | biased_selection | multi_perspective_reasoning>",
  "relevance_score": <an integer from 1 to 5>
}
"""


def llm_reasoning_behavior_judge(
    question: str,
    reasoning_chain: str,
    model: str = "gemini/gemini-2.0-flash",
):
    """
    只基于 QUESTION 和 REASONING CHAIN 来判断行为 + 相关性。

    返回:
    {
      "source_behavior": "non_committal" | "biased_selection" | "multi_perspective_reasoning",
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
            "source_behavior": "non_committal",
            "relevance_score": 3,
        }

    # 规范化字段
    beh = str(result.get("source_behavior", "non_committal")).strip().lower()
    if beh not in [
        "non_committal",
        "biased_selection",
        "multi_perspective_reasoning",
    ]:
        beh = "non_committal"

    try:
        score = int(result.get("relevance_score", 3))
    except Exception:
        score = 3
    score = max(1, min(5, score))

    return {
        "source_behavior": beh,
        "relevance_score": score,
    }


# ---------- 3.（可选）DataFrame 批量版本 ----------

def run_llm_answer_behavior_on_df(
    df: pd.DataFrame,
    model: str = "gemini/gemini-2.0-flash",
    num_threads: int = 8,
) -> Tuple[List[str], List[int]]:
    """
    对 DataFrame 批量跑 answer-only 行为评估。

    要求 df 有列: 'question', 'answer'  (通常 answer = parsed_agent_answer)

    返回:
        behaviors: List[str]
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
        return idx, result["source_behavior"], result["relevance_score"]
    
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
                behaviors[idx] = "non_committal"
                scores[idx] = 3
    
    return behaviors, scores


def run_llm_reasoning_behavior_on_df(
    df: pd.DataFrame,
    model: str = "gemini/gemini-2.0-flash",
    num_threads: int = 8,
) -> Tuple[List[str], List[int]]:
    """
    对 DataFrame 批量跑 reasoning-only 行为评估。

    要求 df 有列: 'question', 'reasoning_chain'

    返回:
        behaviors: List[str]
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
        return idx, result["source_behavior"], result["relevance_score"]
    
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
                behaviors[idx] = "non_committal"
                scores[idx] = 3
    
    return behaviors, scores
