"""
多数据集统计（默认 ResearchyQA，可扩展 Debate_QA 等）：
1) original 数据，按 agent 对比：
   - answer 分类饼图（每个 agent 一张）
   - answer relevance 1-5 分柱状图，三 agent 并排
2) original 数据，answer vs reasoning 汇总：
   - answer / reasoning 各一张分类饼图
   - relevance 1-5 分柱状图，answer / reasoning 并排
3) original 数据，job type（question_type）：
   - 三个 agent 的行为分类按 question_type 取均值，分组柱状图
4) search-r1 original vs optimized（answer）：
   - 分类饼图 original / optimized
   - relevance 1-5 分柱状图，original / optimized 并排
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).parent / "Eval_results"
# 可配置多个数据集；question_file 可为 None
DATASETS = [
    # {
    #     "name": "ResearchyQA",             # 目录名
    #     "file_prefix": "Researchy_QA",     # 输入文件前缀
    #     "output_prefix": "ResearchyQA",    # 输出文件前缀
    #     "question_file": "Researchy_questions.txt",
    #},
    {
        "name": "DebateQA",
        "file_prefix": "Debate_QA",
        "output_prefix": "DebateQA",
        "question_file": "Debate_questions.txt",  # 如有题型文件可填写
    },
    # {
    #     "name": "QACC",
    #     "file_prefix": "QACC",
    #     "output_prefix": "QACC",
    #     "question_file": None,  # 如有题型文件可填写
    # },
    # {
    #     "name": "AmbigQA",
    #     "file_prefix": "AmbigQA",
    #     "output_prefix": "AmbigQA",
    #     "question_file": None,  # 如有题型文件可填写
    # },
]

BEHAVIOR_CLASSES = ["non_committal", "biased_selection", "multi_perspective_reasoning"]
#BEHAVIOR_CLASSES = ["assumption_making", "clarification_seeking", "multi_interpretation_reasoning"]
SCORE_RANGE = list(range(1, 6))


def load_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_question_types(qfile: Optional[Path]) -> List[str]:
    if not qfile or not qfile.exists():
        return []
    try:
        with qfile.open("r", encoding="utf-8") as f:
            qdata = json.load(f)
        qdata_sorted = sorted(qdata, key=lambda x: x.get("id", 0))
        return [q.get("question_type") for q in qdata_sorted]
    except Exception as e:
        print(f"⚠️  读取 question types 失败 {qfile}: {e}")
        return []


def get_agent_name(path: Path, file_prefix: str) -> str:
    """从文件名解析 agent 名称：<file_prefix>_<agent>..."""
    stem = path.stem
    prefix = f"{file_prefix}_"
    if stem.startswith(prefix):
        remainder = stem[len(prefix) :]
        return remainder.split("_", 1)[0]
    return stem.split("_", 1)[0]


def plot_pie(counts: pd.Series, title: str, out_path: Path):
    counts = counts.reindex(BEHAVIOR_CLASSES, fill_value=0)
    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_group_bar(df_counts: pd.DataFrame, title: str, out_path: Path, xlabel: str = ""):
    ax = df_counts.plot(kind="bar", figsize=(10, 5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def agg_behavior_counts(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].value_counts().reindex(BEHAVIOR_CLASSES, fill_value=0)


def agg_score_counts(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].value_counts().reindex(SCORE_RANGE, fill_value=0)


def agent_files_original(original_dir: Path, file_prefix: str) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    if original_dir.exists():
        for path in original_dir.glob("*.json"):
            if "other_prompts" in path.name:
                continue  # other_prompts 视为 optimized，不用于原始 agent 对比
            agent = get_agent_name(path, file_prefix)
            files[agent] = path
    return files


def get_searchr1_opt_path(original_dir: Path, optimized_dir: Path, file_prefix: str) -> Optional[Path]:
    candidates = [
        optimized_dir / f"{file_prefix}_search-r1_other_prompts_parsed_agent_outputs_with_behavior_judge.json",
        original_dir / f"{file_prefix}_search-r1_other_prompts_parsed_agent_outputs_with_behavior_judge.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def detect_behavior_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Try to detect behavior column names for answer / reasoning."""
    cand_answer = [
        "answer_source_behavior",
        "answer_question_behavior",
        "answer_behavior",
    ]
    cand_reasoning = [
        "reasoning_source_behavior",
        "reasoning_question_behavior",
        "reasoning_behavior",
    ]
    ans_col = next((c for c in cand_answer if c in df.columns), None)
    rea_col = next((c for c in cand_reasoning if c in df.columns), None)
    return {"answer": ans_col, "reasoning": rea_col}


def agent_comparison(agent_dfs: Dict[str, pd.DataFrame], out_root: Path, out_prefix: str, answer_col: str):
    for agent, df in agent_dfs.items():
        counts = agg_behavior_counts(df, answer_col)
        plot_pie(counts, f"{agent} answer behavior", out_root / f"{out_prefix}_{agent}_answer_behavior_pie.png")

    score_df = pd.DataFrame(
        {agent: agg_score_counts(df, "answer_relevance_score") for agent, df in agent_dfs.items()}
    )
    score_df.index.name = "score"
    plot_group_bar(
        score_df,
        "answer relevance score by agent (original)",
        out_root / f"{out_prefix}_answer_relevance_agents_bar.png",
        xlabel="score",
    )


def answer_reasoning_comparison(agent_dfs: Dict[str, pd.DataFrame], out_root: Path, out_prefix: str, answer_col: str, reasoning_col: str):
    if not agent_dfs:
        return
    all_df = pd.concat(agent_dfs.values(), ignore_index=True)

    plot_pie(
        agg_behavior_counts(all_df, answer_col),
        "answer behavior (all agents original)",
        out_root / f"{out_prefix}_all_answer_behavior_pie.png",
    )
    plot_pie(
        agg_behavior_counts(all_df, reasoning_col),
        "reasoning behavior (all agents original)",
        out_root / f"{out_prefix}_all_reasoning_behavior_pie.png",
    )

    score_df = pd.DataFrame(
        {
            "answer": agg_score_counts(all_df, "answer_relevance_score"),
            "reasoning": agg_score_counts(all_df, "reasoning_relevance_score"),
        }
    )
    score_df.index.name = "score"
    plot_group_bar(
        score_df,
        "relevance score (answer vs reasoning, original)",
        out_root / f"{out_prefix}_answer_vs_reasoning_relevance_bar.png",
        xlabel="score",
    )


def jobtype_behavior_avg(agent_dfs: Dict[str, pd.DataFrame], question_types: List[str], out_root: Path, out_prefix: str, answer_col: str):
    if not question_types:
        print("⚠️  无 question_type，跳过 job type 图")
        return
    if not agent_dfs:
        return

    per_agent = []
    for agent, df in agent_dfs.items():
        if len(df) != len(question_types):
            print(f"⚠️  {agent} 行数与 question_type 不一致，跳过该 agent 的 jobtype 统计")
            continue
        tmp = pd.DataFrame({"question_type": question_types, "behavior": df[answer_col].tolist()})
        pivot = (
            tmp.pivot_table(index="question_type", columns="behavior", aggfunc="size", fill_value=0)
            .reindex(columns=BEHAVIOR_CLASSES, fill_value=0)
        )
        per_agent.append(pivot)

    if not per_agent:
        return

    avg_df = sum(per_agent) / len(per_agent)
    qtype_order = list(dict.fromkeys(question_types))
    avg_df = avg_df.reindex(qtype_order)

    ax = avg_df.plot(kind="bar", figsize=(14, 6))
    plt.title("Job type behavior avg (answer, 3 agents original)")
    plt.xlabel("question_type")
    plt.ylabel("Average count across agents")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="behavior")
    plt.tight_layout()
    plt.savefig(out_root / f"{out_prefix}_jobtype_answer_behavior_avg_bar.png", bbox_inches="tight")
    plt.close()


def searchr1_original_vs_optimized(agent_dfs: Dict[str, pd.DataFrame], original_dir: Path, optimized_dir: Path, file_prefix: str, out_root: Path, out_prefix: str, answer_col: str):
    if "search-r1" not in agent_dfs:
        print("⚠️  缺少 search-r1 original 数据，跳过优化对比")
        return
    df_orig = agent_dfs["search-r1"]

    opt_path = get_searchr1_opt_path(original_dir, optimized_dir, file_prefix)
    if not opt_path:
        print("⚠️  未找到 search-r1 optimized 数据，跳过优化对比")
        return
    df_opt = load_json(opt_path)

    plot_pie(
        agg_behavior_counts(df_orig, answer_col),
        "search-r1 original answer behavior",
        out_root / f"{out_prefix}_search-r1_answer_behavior_original_pie.png",
    )
    plot_pie(
        agg_behavior_counts(df_opt, answer_col),
        "search-r1 optimized answer behavior",
        out_root / f"{out_prefix}_search-r1_answer_behavior_optimized_pie.png",
    )

    score_df = pd.DataFrame(
        {
            "original": agg_score_counts(df_orig, "answer_relevance_score"),
            "optimized": agg_score_counts(df_opt, "answer_relevance_score"),
        }
    )
    score_df.index.name = "score"
    plot_group_bar(
        score_df,
        "search-r1 answer relevance (original vs optimized)",
        out_root / f"{out_prefix}_search-r1_answer_relevance_compare_bar.png",
        xlabel="score",
    )


def run_dataset(cfg: Dict):
    name = cfg["name"]
    file_prefix = cfg["file_prefix"]
    out_prefix = cfg.get("output_prefix", file_prefix.replace("_", ""))
    root = BASE_DIR / name
    original_dir = root / "original"
    optimized_dir = root / "optimized"
    out_root = root

    if not root.exists():
        print(f"⚠️  Skip {name}: root not found {root}")
        return

    qfile = root / cfg["question_file"] if cfg.get("question_file") else None
    question_types = load_question_types(qfile) if qfile else []

    agent_paths = agent_files_original(original_dir, file_prefix)
    if not agent_paths:
        print(f"⚠️  {name}: no original agent files found.")
        return

    agent_dfs = {agent: load_json(path) for agent, path in agent_paths.items()}
    # 检测列名（基于首个 df）
    sample_df = next(iter(agent_dfs.values()))
    cols = detect_behavior_cols(sample_df)
    answer_col, reasoning_col = cols.get("answer"), cols.get("reasoning")
    if not answer_col or not reasoning_col:
        print(f"⚠️  {name}: 未找到行为列，answer={answer_col}, reasoning={reasoning_col}，跳过")
        return

    agent_comparison(agent_dfs, out_root, out_prefix, answer_col)
    answer_reasoning_comparison(agent_dfs, out_root, out_prefix, answer_col, reasoning_col)
    if question_types:
        jobtype_behavior_avg(agent_dfs, question_types, out_root, out_prefix, answer_col)
    else:
        print(f"ℹ️  {name}: no question types, skip jobtype plot.")
    searchr1_original_vs_optimized(agent_dfs, original_dir, optimized_dir, file_prefix, out_root, out_prefix, answer_col)

    print(f"{name} done.")


def main():
    for cfg in DATASETS:
        run_dataset(cfg)
    print("All datasets done.")


if __name__ == "__main__":
    main()

