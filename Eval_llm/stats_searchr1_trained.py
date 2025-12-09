"""
统计 DebateQA search-r1-trained 数据的行为分类分布和相关性分数分布
生成：
1. answer_source_behavior 分类饼状图
2. reasoning_source_behavior 分类饼状图
3. answer_relevance_score 柱状图
4. reasoning_relevance_score 柱状图
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 输入文件路径
INPUT_FILE = Path(r"C:\Users\silin\Desktop\cs329x\project\Eval_llm\Eval_results\DebateQA\search-r1-trained\parsed_agent_outputs_with_behavior_judge.json")

# 输出目录（与输入文件同一目录）
OUTPUT_DIR = INPUT_FILE.parent

BEHAVIOR_CLASSES = ["non_committal", "biased_selection", "multi_perspective_reasoning"]
SCORE_RANGE = list(range(1, 6))


def load_data(file_path: Path) -> pd.DataFrame:
    """加载JSON数据文件"""
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def plot_pie(counts: pd.Series, title: str, out_path: Path):
    """绘制饼状图"""
    counts = counts.reindex(BEHAVIOR_CLASSES, fill_value=0)
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ 已保存: {out_path}")


def plot_bar(counts: pd.Series, title: str, out_path: Path, xlabel: str = "Score", ylabel: str = "Count"):
    """绘制柱状图"""
    counts = counts.reindex(SCORE_RANGE, fill_value=0)
    plt.figure(figsize=(8, 6))
    ax = counts.plot(kind="bar", color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=0)
    
    # 在柱子上显示数值
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, str(int(v)), ha='center', va='bottom', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ 已保存: {out_path}")


def main():
    print("=" * 60)
    print("统计 DebateQA search-r1-trained 数据")
    print("=" * 60)
    
    # 检查文件是否存在
    if not INPUT_FILE.exists():
        print(f"❌ 文件不存在: {INPUT_FILE}")
        return
    
    # 加载数据
    print(f"\n[1/3] 加载数据: {INPUT_FILE}")
    df = load_data(INPUT_FILE)
    print(f"✓ 共加载 {len(df)} 条数据")
    
    # 检查必要的列是否存在
    required_cols = ["answer_source_behavior", "answer_relevance_score", 
                     "reasoning_source_behavior", "reasoning_relevance_score"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少必要的列: {missing_cols}")
        print(f"可用列: {list(df.columns)}")
        return
    
    # 统计行为分类分布
    print("\n[2/3] 统计行为分类分布...")
    answer_behavior_counts = df["answer_source_behavior"].value_counts()
    reasoning_behavior_counts = df["reasoning_source_behavior"].value_counts()
    
    print("\nAnswer Source Behavior 分布:")
    for behavior, count in answer_behavior_counts.items():
        pct = count / len(df) * 100
        print(f"  {behavior}: {count} ({pct:.1f}%)")
    
    print("\nReasoning Source Behavior 分布:")
    for behavior, count in reasoning_behavior_counts.items():
        pct = count / len(df) * 100
        print(f"  {behavior}: {count} ({pct:.1f}%)")
    
    # 统计相关性分数分布
    print("\n[3/3] 统计相关性分数分布...")
    answer_score_counts = df["answer_relevance_score"].value_counts()
    reasoning_score_counts = df["reasoning_relevance_score"].value_counts()
    
    print("\nAnswer Relevance Score 分布:")
    for score in sorted(answer_score_counts.index):
        count = answer_score_counts[score]
        pct = count / len(df) * 100
        print(f"  {score}: {count} ({pct:.1f}%)")
    
    print("\nReasoning Relevance Score 分布:")
    for score in sorted(reasoning_score_counts.index):
        count = reasoning_score_counts[score]
        pct = count / len(df) * 100
        print(f"  {score}: {count} ({pct:.1f}%)")
    
    # 生成图表
    print("\n生成图表...")
    
    # 1. Answer behavior 饼状图
    plot_pie(
        answer_behavior_counts,
        "Answer Source Behavior Distribution",
        OUTPUT_DIR / "searchr1_trained_answer_behavior_pie.png"
    )
    
    # 2. Reasoning behavior 饼状图
    plot_pie(
        reasoning_behavior_counts,
        "Reasoning Source Behavior Distribution",
        OUTPUT_DIR / "searchr1_trained_reasoning_behavior_pie.png"
    )
    
    # 3. Answer relevance 柱状图
    plot_bar(
        answer_score_counts,
        "Answer Relevance Score Distribution",
        OUTPUT_DIR / "searchr1_trained_answer_relevance_bar.png",
        xlabel="Relevance Score",
        ylabel="Count"
    )
    
    # 4. Reasoning relevance 柱状图
    plot_bar(
        reasoning_score_counts,
        "Reasoning Relevance Score Distribution",
        OUTPUT_DIR / "searchr1_trained_reasoning_relevance_bar.png",
        xlabel="Relevance Score",
        ylabel="Count"
    )
    
    print("\n" + "=" * 60)
    print("✓ 统计完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
