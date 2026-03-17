import pandas as pd
import re
import os

def find_sentences_with_context(text, keyword="装修", context_sentences=3):
    """
    在文本中查找包含关键词的句子，并返回其上下文
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # 分句（按句号、问号、感叹号、换行符分割）
    sentences = re.split(r'[。！？!?\n]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    results = []
    for i, sentence in enumerate(sentences):
        if keyword in sentence:
            # 确定上下文范围
            start = max(0, i - context_sentences)
            end = min(len(sentences), i + context_sentences + 1)
            
            # 获取上下文
            context = sentences[start:end]
            context_text = "。".join(context) + "。"
            
            results.append({
                '原句子': sentence,
                '上下文': context_text,
                '句子位置': i,
                '上下文起始位置': start,
                '上下文结束位置': end - 1
            })
    
    return results

def process_csv_file(input_file, output_file=None, keyword="装修", text_column=None):
    """
    处理CSV文件，查找包含关键词的句子及其上下文
    """
    print(f"正在读取文件: {input_file}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        print("使用UTF-8编码读取")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, encoding='gbk')
            print("使用GBK编码读取")
        except UnicodeDecodeError:
            df = pd.read_csv(input_file, encoding='utf-8-sig')
            print("使用UTF-8-SIG编码读取")
    
    print(f"文件包含 {len(df)} 行数据")
    print(f"列名: {list(df.columns)}")
    
    # 显示前几行数据预览
    print("\n数据预览（前3行）:")
    print(df.head(3))
    
    # 自动检测文本列
    if text_column is None:
        text_columns = []
        for col in df.columns:
            # 检查列是否包含文本数据
            sample = df[col].dropna()
            if len(sample) > 0 and isinstance(sample.iloc[0], str) and len(str(sample.iloc[0])) > 20:
                text_columns.append(col)
        
        if len(text_columns) == 0:
            # 如果没有找到明显的文本列，使用第一个字符串类型的列
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_columns.append(col)
                    break
        
        if text_columns:
            text_column = text_columns[0]
            print(f"\n自动选择文本列: {text_column}")
        else:
            text_column = df.columns[0]
            print(f"\n未找到合适的文本列，使用第一列: {text_column}")
    
    # 处理每一行
    all_results = []
    
    print(f"\n开始搜索关键词 '{keyword}'...")
    
    for idx, row in df.iterrows():
        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        results = find_sentences_with_context(text, keyword)
        
        for result in results:
            result_data = {
                '原始行号': idx,
                '关键词': keyword,
                f'{text_column}_预览': text[:100] + '...' if len(text) > 100 else text,
                '匹配句子': result['原句子'],
                '上下文': result['上下文'],
            }
            
            # 添加原始CSV的其他列
            for col in df.columns:
                if col != text_column:
                    result_data[f'原始_{col}'] = row[col]
            
            all_results.append(result_data)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) > 0:
        print(f"\n找到 {len(results_df)} 个包含'{keyword}'的句子（包含上下文）")
        
        # 显示匹配结果的预览
        print("\n匹配结果预览:")
        for i in range(min(3, len(results_df))):
            print(f"\n--- 匹配 {i+1} ---")
            print(f"匹配句子: {results_df.iloc[i]['匹配句子']}")
            print(f"上下文: {results_df.iloc[i]['上下文'][:150]}...")
    else:
        print(f"\n未找到包含'{keyword}'的句子")
    
    # 保存结果
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_包含{keyword}_上下文.csv"
    
    # 保存为CSV
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_file}")
    
    return results_df

if __name__ == "__main__":
    import sys
    
    # 设置输入文件路径
    input_file = r"asap-master\data\test.csv"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在!")
        print(f"当前工作目录: {os.getcwd()}")
        sys.exit(1)
    
    # 处理文件
    results = process_csv_file(
        input_file=input_file,
        keyword="装修",
        output_file="test_包含装修_上下文.csv"
    )