"""
自动化运行 AmbigQA 的 question-related behavior judge 脚本
处理 AmbigQA 目录下所有的 parsed_results.json 文件
"""

import json
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from judge_utils_ambig import (
    run_llm_answer_behavior_on_df,
    run_llm_reasoning_behavior_on_df,
)

load_dotenv()

MODEL_NAME = "gemini/gemini-2.5-flash"
NUM_THREADS = 8

# 要处理的根目录（现在是 AmbigQA）
DATASET_DIRS = ["AmbigQA"]


def generate_output_filename(input_path: str, script_dir: Path) -> str:
    """
    根据输入路径生成输出文件名
    格式: {QA类型}_{子文件夹名}_{other_prompts_}parsed_agent_outputs_with_qbehavior_judge.json
    
    Args:
        input_path: 输入的 parsed_results.json 文件路径
        script_dir: 脚本所在目录
    
    Returns:
        输出文件路径
    """
    input_path_obj = Path(input_path)
    parts = input_path_obj.parts
    
    qa_type = None
    subfolder = None
    is_other_prompts = False
    
    for i, part in enumerate(parts):
        if part in DATASET_DIRS:
            qa_type = part
            if i + 1 < len(parts):
                subfolder = parts[i + 1]
            if "other_prompts" in parts:
                is_other_prompts = True
            break
    
    if qa_type is None or subfolder is None:
        # 如果无法解析，使用默认名称
        return str(script_dir / "parsed_agent_outputs_with_qbehavior_judge.json")
    
    filename_parts = [qa_type, subfolder]
    if is_other_prompts:
        filename_parts.append("other_prompts")
    # 标记一下这是 question-behavior 版本
    filename_parts.append("parsed_agent_outputs_with_qbehavior_judge.json")
    
    filename = "_".join(filename_parts)
    output_path = script_dir / filename
    
    return str(output_path)


def process_single_file(input_path: str, output_path: str = None, script_dir: Path = None):
    """
    处理单个 parsed_results.json 文件（AmbigQA）
    
    Args:
        input_path: 输入的 parsed_results.json 文件路径
        output_path: 输出文件路径，如果为 None 则自动生成
        script_dir: 脚本所在目录，用于生成输出路径
    """
    if script_dir is None:
        script_dir = Path(__file__).parent
    
    if output_path is None:
        output_path = generate_output_filename(input_path, script_dir)
    
    print(f"\n{'='*80}")
    print(f"处理文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"{'='*80}")
    
    try:
        # 读取数据
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            print(f"警告: {input_path} 是空文件，跳过")
            return
        
        df_raw = pd.DataFrame(data)
        print(f"数据行数: {len(df_raw)}")
        
        # 1. answer-only question-related behavior judge
        print("\n开始 answer-only question_behavior judge...")
        df_answer = pd.DataFrame()
        df_answer["question"] = df_raw["question"]
        df_answer["answer"] = df_raw["parsed_agent_answer"]
        
        ans_beh, ans_rel = run_llm_answer_behavior_on_df(
            df_answer,
            model=MODEL_NAME,
            num_threads=NUM_THREADS,
        )
        
        # 2. reasoning-only question-related behavior judge
        print("\n开始 reasoning-only question_behavior judge...")
        df_reason = pd.DataFrame()
        df_reason["question"] = df_raw["question"]
        df_reason["reasoning_chain"] = df_raw["reasoning_chain"]
        
        rea_beh, rea_rel = run_llm_reasoning_behavior_on_df(
            df_reason,
            model=MODEL_NAME,
            num_threads=NUM_THREADS,
        )
        
        # 3. 写回原始 df（列名改成 question_behavior 语义）
        df_raw["answer_question_behavior"] = ans_beh
        df_raw["answer_relevance_score"] = ans_rel
        df_raw["reasoning_question_behavior"] = rea_beh
        df_raw["reasoning_relevance_score"] = rea_rel
        
        # 保存结果
        df_raw.to_json(
            output_path,
            orient="records",
            indent=2,
            force_ascii=False,
        )
        
        print(f"\n✅ 成功保存到: {output_path}")
        print(f"\n结果预览:")
        print(
            df_raw[
                [
                    "question",
                    "parsed_agent_answer",
                    "answer_question_behavior",
                    "answer_relevance_score",
                    "reasoning_question_behavior",
                    "reasoning_relevance_score",
                ]
            ].head()
        )
        
    except Exception as e:
        print(f"❌ 处理文件 {input_path} 时出错: {e}")
        import traceback
        traceback.print_exc()


def find_all_parsed_results(base_dir: str):
    """
    找到所有 parsed_results.json 文件
    
    Args:
        base_dir: 基础目录路径
    
    Returns:
        parsed_results.json 文件路径列表
    """
    results = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"警告: 目录 {base_dir} 不存在")
        return results
    
    for parsed_file in base_path.rglob("parsed_results.json"):
        results.append(str(parsed_file))
    
    return sorted(results)


def main():
    """主函数：处理 AmbigQA 中的所有 parsed_results.json"""
    print("="*80)
    print("开始批量处理 AmbigQA question-related behavior judge")
    print("="*80)
    
    script_dir = Path(__file__).parent
    all_files = []
    
    for dataset_dir in DATASET_DIRS:
        dataset_path = script_dir / dataset_dir
        if dataset_path.exists():
            files = find_all_parsed_results(str(dataset_path))
            all_files.extend(files)
            print(f"\n在 {dataset_dir} 中找到 {len(files)} 个 parsed_results.json 文件")
        else:
            print(f"警告: 目录 {dataset_dir} 不存在")
    
    if not all_files:
        print("❌ 没有找到任何 parsed_results.json 文件")
        return
    
    print(f"\n总共找到 {len(all_files)} 个文件需要处理:")
    for i, file_path in enumerate(all_files, 1):
        print(f"  {i}. {file_path}")
    
    # 确认是否继续
    response = input(f"\n是否继续处理这 {len(all_files)} 个文件? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 逐个处理
    for i, input_path in enumerate(all_files, 1):
        print(f"\n\n[{i}/{len(all_files)}] 处理中...")
        process_single_file(input_path, script_dir=script_dir)
    
    print("\n" + "="*80)
    print("✅ 所有文件处理完成!")
    print("="*80)


if __name__ == "__main__":
    main()
