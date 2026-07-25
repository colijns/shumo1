import pandas as pd
print("=== λ敏感性总表 ===")
df1 = pd.read_excel(r'Q2/output_penalty/λ敏感性总表.xlsx')
print(df1.to_string())
print("\n=== 方案对比总表 ===")
df2 = pd.read_excel(r'Q2/output_penalty/方案对比总表.xlsx')
print(df2.to_string())
