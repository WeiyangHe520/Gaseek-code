import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 配置参数
src_dir = r"D:\何维阳\20250927\20250927\data\csv"
null_threshold = 0.9  # 缺失值比例阈值

def main():
    # 1. 确定有效的列（缺失值比例低于阈值）
    print("步骤1: 筛选有效列...")
    
    # 获取第一个CSV文件来确定列数
    csv_files = [f for f in os.listdir(src_dir) if f.endswith(".csv")]
    if not csv_files:
        print("未找到CSV文件")
        return
    
    # 读取第一个文件获取列数
    first_file = os.path.join(src_dir, csv_files[0])
    df_sample = pd.read_csv(first_file, encoding='gbk')
    n_columns = len(df_sample.columns)
    
    mask = np.ones(n_columns, dtype=np.int32)
    
    # 统计所有文件的缺失值情况
    for f in csv_files:
        file_path = os.path.join(src_dir, f)
        df = pd.read_csv(file_path, encoding='gbk')
        
        # 确保列数一致
        if len(df.columns) != n_columns:
            print(f"警告: 文件 {f} 的列数({len(df.columns)})与样本文件不一致({n_columns})")
            continue
        
        cnt = df.isnull().sum().values
        null_ratio = cnt / df.shape[0]
        mask[null_ratio > null_threshold] = 0
    
    print(f"原始列数: {n_columns}, 保留列数: {np.sum(mask)}")
    
    # 2. 使用MICE进行缺失值插补
    print("\n步骤2: 使用MICE进行缺失值插补...")
    
    # 初始化MICE插补器
    imp = IterativeImputer(
        max_iter=20,
        random_state=0,
        initial_strategy='median',
        skip_complete=True
    )
    
    processed_count = 0
    for f in csv_files:
        try:
            file_path = os.path.join(src_dir, f)
            df = pd.read_csv(file_path, encoding='gbk')
            
            # 确保列数一致
            if len(df.columns) != n_columns:
                print(f"跳过文件 {f}: 列数不一致")
                continue
            
            # 选择有效列
            col_names = df.columns[mask == 1]
            data = df.values[:, mask == 1]
            
            # 检查是否有数据需要插补
            if np.isnan(data).any():
                print(f"处理文件: {f} - 存在缺失值")
                # 进行缺失值插补
                data_imputed = imp.fit_transform(data)
            else:
                print(f"处理文件: {f} - 无缺失值")
                data_imputed = data
            
            # 保存结果
            output_filename = f.replace(".csv", "_mice.csv")
            output_path = os.path.join(src_dir, output_filename)
            
            df_imputed = pd.DataFrame(data=data_imputed, columns=col_names)
            df_imputed.to_csv(output_path, index=False, encoding='gbk')
            
            processed_count += 1
            print(f"  → 已保存: {output_filename}")
            
        except Exception as e:
            print(f"处理文件 {f} 时出错: {str(e)}")
            continue
    
    print(f"\n处理完成！共处理 {processed_count} 个文件")

if __name__ == "__main__":
    main()