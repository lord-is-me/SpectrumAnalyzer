"""
跑完所有backbone后执行，汇总对比表格
用法: python summarize_results.py
"""
import pandas as pd
from pathlib import Path

RESULT_ROOT = Path("results")
BACKBONES   = ["vgg16", "resnet50", "resnet101", "vit_b"]

rows = []
for bb in BACKBONES:
    p = RESULT_ROOT / bb / "test_result.csv"
    if p.exists():
        rows.append(pd.read_csv(p).iloc[0].to_dict())
    else:
        rows.append({"backbone": bb, "test_f1": "-", "test_auc": "-", "best_val_f1": "-", "epochs_ran": "-"})

df = pd.DataFrame(rows)[["backbone", "test_f1", "test_auc", "best_val_f1", "epochs_ran"]]
df.columns = ["Backbone", "Test Macro-F1", "Test Macro-AUC", "Best Val-F1", "Epochs"]

print("\n===== Backbone 对比结果 =====")
print(df.to_string(index=False))
df.to_csv(RESULT_ROOT / "comparison.csv", index=False)
print(f"\n已保存到 results/comparison.csv")
