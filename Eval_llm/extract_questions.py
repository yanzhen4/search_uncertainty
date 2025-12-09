"""
从JSON文件中提取所有question字段
"""
import json
from pathlib import Path


def extract_questions(input_file: str, output_file: str = None):
    """
    从JSON文件中提取所有question字段
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的文本文件路径，如果为None则自动生成
    """
    # 读取JSON文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 提取所有question
    questions = []
    for item in data:
        if "question" in item:
            questions.append(item["question"])
    
    # 生成输出文件名
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_questions.txt"
    
    # 保存到文件
    with open(output_file, "w", encoding="utf-8") as f:
        for i, question in enumerate(questions, 1):
            f.write(f"{i}. {question}\n")
    
    print(f"✅ 成功提取 {len(questions)} 个问题")
    print(f"📄 保存到: {output_file}")
    
    # 显示前几个问题作为预览
    print(f"\n前5个问题预览:")
    for i, question in enumerate(questions[:5], 1):
        print(f"  {i}. {question}")
    
    return questions


def main():
    """主函数"""
    import sys
    
    # 默认文件路径
    default_file = r"C:\Users\silin\Desktop\cs329x\project\Eval_llm\Researchy_QA\desa\parsed_results.json"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

        input_file = default_file
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = None
    
    print("="*80)
    print("提取问题脚本")
    print("="*80)
    print(f"输入文件: {input_file}")
    
    try:
        extract_questions(input_file, output_file)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {input_file}")
    except json.JSONDecodeError as e:
        print(f"❌ 错误: JSON解析失败 - {e}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

