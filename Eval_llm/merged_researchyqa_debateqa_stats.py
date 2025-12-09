"""
合并 ResearchyQA 和 DebateQA 的统计（使用相同的统计标准）：
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

# 要合并的数据集配置
DATASET_CONFIGS = [
    {
        "name": "ResearchyQA",
        "file_prefix": "Researchy_QA",
        "question_file": "Researchy_questions.txt",
    },
    {
        "name": "DebateQA",
        "file_prefix": "Debate_QA",
        "question_file": "Debate_questions.txt",
    },
]

OUTPUT_PREFIX = "Merged_ResearchyQA_DebateQA"

BEHAVIOR_CLASSES = ["non_committal", "biased_selection", "multi_perspective_reasoning"]
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


def merge_agent_files(dataset_configs: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    从多个数据集加载并合并相同 agent 的文件
    返回: {agent_name: merged_dataframe}
    """
    all_agent_files: Dict[str, List[Path]] = {}
    
    for cfg in dataset_configs:
        name = cfg["name"]
        file_prefix = cfg["file_prefix"]
        root = BASE_DIR / name
        original_dir = root / "original"
        
        if not root.exists():
            print(f"⚠️  Skip {name}: root not found {root}")
            continue
        
        agent_paths = agent_files_original(original_dir, file_prefix)
        for agent, path in agent_paths.items():
            if agent not in all_agent_files:
                all_agent_files[agent] = []
            all_agent_files[agent].append(path)
    
    # 合并每个 agent 的数据
    merged_agents: Dict[str, pd.DataFrame] = {}
    for agent, paths in all_agent_files.items():
        dfs = [load_json(path) for path in paths]
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_agents[agent] = merged_df
        print(f"✓  {agent}: 合并了 {len(paths)} 个文件，共 {len(merged_df)} 条数据")
    
    return merged_agents


def merge_question_types(dataset_configs: List[Dict]) -> List[str]:
    """合并多个数据集的 question types"""
    all_types = []
    for cfg in dataset_configs:
        name = cfg["name"]
        qfile = BASE_DIR / name / cfg.get("question_file", "")
        types = load_question_types(qfile) if cfg.get("question_file") else []
        all_types.extend(types)
        if types:
            print(f"✓  {name}: 加载了 {len(types)} 个 question types")
    return all_types


def agent_comparison(agent_dfs: Dict[str, pd.DataFrame], out_root: Path, out_prefix: str, answer_col: str):
    for agent, df in agent_dfs.items():
        counts = agg_behavior_counts(df, answer_col)
        plot_pie(counts, f"{agent} answer behavior (merged)", out_root / f"{out_prefix}_{agent}_answer_behavior_pie.png")

    score_df = pd.DataFrame(
        {agent: agg_score_counts(df, "answer_relevance_score") for agent, df in agent_dfs.items()}
    )
    score_df.index.name = "score"
    plot_group_bar(
        score_df,
        "answer relevance score by agent (original, merged)",
        out_root / f"{out_prefix}_answer_relevance_agents_bar.png",
        xlabel="score",
    )


def answer_reasoning_comparison(agent_dfs: Dict[str, pd.DataFrame], out_root: Path, out_prefix: str, answer_col: str, reasoning_col: str):
    if not agent_dfs:
        return
    all_df = pd.concat(agent_dfs.values(), ignore_index=True)

    plot_pie(
        agg_behavior_counts(all_df, answer_col),
        "answer behavior (all agents original, merged)",
        out_root / f"{out_prefix}_all_answer_behavior_pie.png",
    )
    plot_pie(
        agg_behavior_counts(all_df, reasoning_col),
        "reasoning behavior (all agents original, merged)",
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
        "relevance score (answer vs reasoning, original, merged)",
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
            print(f"⚠️  {agent} 行数 ({len(df)}) 与 question_type 数量 ({len(question_types)}) 不一致，跳过该 agent 的 jobtype 统计")
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
    plt.title("Job type behavior avg (answer, 3 agents original, merged)")
    plt.xlabel("question_type")
    plt.ylabel("Average count across agents")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="behavior")
    plt.tight_layout()
    plt.savefig(out_root / f"{out_prefix}_jobtype_answer_behavior_avg_bar.png", bbox_inches="tight")
    plt.close()


def searchr1_original_vs_optimized(agent_dfs: Dict[str, pd.DataFrame], dataset_configs: List[Dict], out_root: Path, out_prefix: str, answer_col: str):
    if "search-r1" not in agent_dfs:
        print("⚠️  缺少 search-r1 original 数据，跳过优化对比")
        return
    df_orig = agent_dfs["search-r1"]

    # 尝试从所有数据集中找到 optimized 文件并合并
    opt_dfs = []
    for cfg in dataset_configs:
        name = cfg["name"]
        file_prefix = cfg["file_prefix"]
        root = BASE_DIR / name
        original_dir = root / "original"
        optimized_dir = root / "optimized"
        
        opt_path = get_searchr1_opt_path(original_dir, optimized_dir, file_prefix)
        if opt_path:
            opt_df = load_json(opt_path)
            opt_dfs.append(opt_df)
            print(f"✓  找到 {name} 的 search-r1 optimized 数据")
    
    if not opt_dfs:
        print("⚠️  未找到任何 search-r1 optimized 数据，跳过优化对比")
        return
    
    df_opt = pd.concat(opt_dfs, ignore_index=True)

    plot_pie(
        agg_behavior_counts(df_orig, answer_col),
        "search-r1 original answer behavior (merged)",
        out_root / f"{out_prefix}_search-r1_answer_behavior_original_pie.png",
    )
    plot_pie(
        agg_behavior_counts(df_opt, answer_col),
        "search-r1 optimized answer behavior (merged)",
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
        "search-r1 answer relevance (original vs optimized, merged)",
        out_root / f"{out_prefix}_search-r1_answer_relevance_compare_bar.png",
        xlabel="score",
    )


def main():
    print("=" * 60)
    print("合并 ResearchyQA 和 DebateQA 统计")
    print("=" * 60)
    
    # 创建输出目录（使用第一个数据集的目录作为输出目录）
    out_root = BASE_DIR / DATASET_CONFIGS[0]["name"]
    if not out_root.exists():
        out_root.mkdir(parents=True, exist_ok=True)
    
    # 合并 agent 文件
    print("\n[1/4] 加载并合并 agent 文件...")
    agent_dfs = merge_agent_files(DATASET_CONFIGS)
    if not agent_dfs:
        print("⚠️  未找到任何 agent 文件，退出")
        return
    
    # 检测列名（基于首个 df）
    sample_df = next(iter(agent_dfs.values()))
    cols = detect_behavior_cols(sample_df)
    answer_col, reasoning_col = cols.get("answer"), cols.get("reasoning")
    if not answer_col or not reasoning_col:
        print(f"⚠️  未找到行为列，answer={answer_col}, reasoning={reasoning_col}，退出")
        return
    
    print(f"✓  检测到列名: answer={answer_col}, reasoning={reasoning_col}")
    
    # 合并 question types
    print("\n[2/4] 加载并合并 question types...")
    question_types = merge_question_types(DATASET_CONFIGS)
    
    # 生成统计图表
    print("\n[3/4] 生成统计图表...")
    agent_comparison(agent_dfs, out_root, OUTPUT_PREFIX, answer_col)
    answer_reasoning_comparison(agent_dfs, out_root, OUTPUT_PREFIX, answer_col, reasoning_col)
    if question_types:
        jobtype_behavior_avg(agent_dfs, question_types, out_root, OUTPUT_PREFIX, answer_col)
    else:
        print("ℹ️  no question types, skip jobtype plot.")
    
    print("\n[4/4] 生成 search-r1 优化对比...")
    searchr1_original_vs_optimized(agent_dfs, DATASET_CONFIGS, out_root, OUTPUT_PREFIX, answer_col)
    
    print("\n" + "=" * 60)
    print("✓ 合并统计完成！")
    print(f"输出目录: {out_root}")
    print(f"输出前缀: {OUTPUT_PREFIX}")
    print("=" * 60)


if __name__ == "__main__":
    main()
