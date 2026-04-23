import torch
import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm

# ==========================================
# 0. 环境与模型导入设置
# ==========================================
# 添加你的模型代码所在目录到系统路径
sys.path.append('train_learnable/v3_all1')
try:
    from train_encoder import AnglE
except ImportError:
    print("❌ 导入失败，请确保在正确的项目根目录下运行此代码！")
    sys.exit(1)

def mine_extreme_cases(csv_path, model_path):
    # ⚠️ 强烈建议：这里加载的应该是你消融实验中“未加 AmpLoss (w/o AmpLoss)”的模型
    print(f"📦 正在加载模型提取真实特征: {model_path}")
    try:
        model = AnglE.from_pretrained(model_path, train_mode=False)
        model.cuda() # 如果显存不够或没有GPU，可以改为 model.cpu()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print(f"📄 正在读取数据: {csv_path}")
    try:
        # 读取 CSV (支持包含中文路径和表头的情况)
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # ==========================================
    # 核心修复 1：明确指定“文本”列的名称
    # ==========================================
    text_column = '匹配句子'  # 👈 精准锁定包含中文评论的列

    if text_column not in df.columns:
        print(f"❌ 找不到列名 '{text_column}'，请检查你的 CSV 表头！")
        return

    print(f"🔍 开始计算 {len(df)} 条句子的复数特征和模长...")

    results = []
    batch_size = 32 # 如果 GPU 显存爆了，把它改成 16 或 8

    # 提取特征并计算模长
    for i in tqdm(range(0, len(df), batch_size)):
        # 提取当前批次的文本，确保转为字符串
        batch_texts = df[text_column].iloc[i:i+batch_size].astype(str).tolist()

        with torch.no_grad():
            # 获取模型特征输出
            vecs = model.encode(batch_texts, to_numpy=False).cpu()
            # 按照你模型的逻辑切分实部和虚部
            re, im = torch.chunk(vecs, 2, dim=1)
            # 计算精确的 L2 模长
            amps = torch.sqrt(torch.sum(re**2 + im**2, dim=1)).numpy()

        for text, amp in zip(batch_texts, amps):
            results.append({
                'text': text,
                'length': len(text),
                'amplitude': amp
            })

    # 将计算出的模长与原始数据合并
    results_df = pd.DataFrame(results)
    for col in df.columns:
        if col != text_column:
            results_df[col] = df[col]

    # 导出完整数据备查
    output_csv = csv_path.replace('.csv', '_with_amplitudes.csv')
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 全部计算完成，附带模长的完整数据已保存至: {output_csv}")

    # ==========================================
    # 核心修复 2：精准狙击 Decoration 维度
    # ==========================================
    print("\n" + "="*60)
    print("🏆 终极挖掘结果 (Extreme Cases for Paper)")
    print("="*60)

    # 锁定你的装修维度表头
    target_aspect = '原始_Ambience#Decoration'

    if target_aspect in results_df.columns:
        print(f"\n--- 🌟 策略：精准狙击 [{target_aspect}] 维度的最大差距 ---")

        # 筛选出确实评价了“装修”，并且是正面评价(1)的句子
        # ⚠️ 注意：如果你的数据集里正面评价的标签是别的值（比如 2 或 'Positive'），请修改下面的 == 1
        decor_df = results_df[results_df[target_aspect] == 1].copy() 

        if len(decor_df) > 1:
            # 找出该类别下模长的最小值和最大值索引
            min_idx = decor_df['amplitude'].idxmin()
            max_idx = decor_df['amplitude'].idxmax()

            min_amp = decor_df.loc[min_idx, 'amplitude']
            max_amp = decor_df.loc[max_idx, 'amplitude']
            diff = max_amp - min_amp

            print(f"🔽 最小模长 (客观陈述) | Norm = {min_amp:.4f} | 长度 = {decor_df.loc[min_idx, 'length']}")
            print(f"   💬 文本: {decor_df.loc[min_idx, 'text']}")

            print(f"\n🔼 最大模长 (强烈主观) | Norm = {max_amp:.4f} | 长度 = {decor_df.loc[max_idx, 'length']}")
            print(f"   💬 文本: {decor_df.loc[max_idx, 'text']}")

            print(f"\n📈 完美对比 Gap (Delta Norm): {diff:.4f}")

            if diff > 0.5:
                print("🎉 恭喜！Gap 超过 0.5，这组数据可以完美支撑你的论点，直接拿去画图吧！")
            else:
                print("💡 提示：当前的 Gap 比较小，说明该模型可能已经加入了模长约束。请确认加载的是【未加 AmpLoss】的基线权重以展现最强对比！")
        else:
            print("❌ 当前数据中关于装修的好评数量不足，无法对比。")
    else:
        print(f"❌ 在你的 CSV 中找不到目标列 '{target_aspect}'。")

if __name__ == '__main__':
    # ⚠️ 记得确保加载的是 w/o AmpLoss 的模型权重
    MODEL_PATH = "checkpoints_learnable/v3_all1"
    CSV_PATH = r"plot\amp\test_data_mining\test_sentence.csv"

    mine_extreme_cases(CSV_PATH, MODEL_PATH)