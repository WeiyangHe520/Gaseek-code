import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """数据处理类，包含多种缺失值处理方法"""
    
    def __init__(self, null_threshold=0.25, imputation_method='median'):
        """
        初始化参数
        
        Parameters:
        -----------
        null_threshold : float, default=0.25
            缺失值比例阈值，超过此比例的列将被删除
        imputation_method : str, default='median'
            缺失值填充方法，可选：'median', 'mean', 'zero', 'forward', 'backward'
        """
        self.null_threshold = null_threshold
        self.imputation_method = imputation_method
        self.mask = None
        self.column_names = None
        
    def simple_imputation(self, data):
        """
        简单的缺失值填充
        
        Parameters:
        -----------
        data : numpy.ndarray
            输入数据
            
        Returns:
        --------
        numpy.ndarray : 填充后的数据
        """
        if self.imputation_method == 'median':
            fill_values = np.nanmedian(data, axis=0)
        elif self.imputation_method == 'mean':
            fill_values = np.nanmean(data, axis=0)
        elif self.imputation_method == 'zero':
            fill_values = 0
        elif self.imputation_method == 'forward':
            return pd.DataFrame(data).fillna(method='ffill').values
        elif self.imputation_method == 'backward':
            return pd.DataFrame(data).fillna(method='bfill').values
        else:
            fill_values = np.nanmedian(data, axis=0)
        
        # 创建数据副本
        data_imputed = data.copy()
        
        # 对每一列进行填充
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            nan_mask = np.isnan(col_data)
            if np.any(nan_mask):
                if isinstance(fill_values, np.ndarray):
                    fill_value = fill_values[col_idx]
                else:
                    fill_value = fill_values
                data_imputed[nan_mask, col_idx] = fill_value
        
        return data_imputed
    
    def detect_and_remove_high_null_columns(self, src_dir):
        """
        检测并标记高缺失值列
        
        Parameters:
        -----------
        src_dir : str
            源数据目录
            
        Returns:
        --------
        tuple : (列名列表, 列掩码)
        """
        print("正在分析数据质量...")
        
        csv_files = [f for f in os.listdir(src_dir) if f.endswith(".csv")]
        if not csv_files:
            raise ValueError(f"在目录 {src_dir} 中未找到CSV文件")
        
        # 读取第一个文件获取列信息
        first_file = os.path.join(src_dir, csv_files[0])
        df_sample = pd.read_csv(first_file, encoding='gbk')
        n_columns = len(df_sample.columns)
        self.column_names = df_sample.columns.tolist()
        
        # 初始化掩码（1表示保留，0表示删除）
        mask = np.ones(n_columns, dtype=bool)
        
        # 统计缺失值
        null_counts = np.zeros(n_columns)
        total_rows = 0
        
        for idx, f in enumerate(csv_files[:10]):  # 只分析前10个文件以提高速度
            try:
                file_path = os.path.join(src_dir, f)
                df = pd.read_csv(file_path, encoding='gbk')
                
                if len(df.columns) != n_columns:
                    print(f"警告: 文件 {f} 列数不一致，跳过")
                    continue
                
                null_counts += df.isnull().sum().values
                total_rows += df.shape[0]
                
            except Exception as e:
                print(f"读取文件 {f} 时出错: {e}")
                continue
        
        # 计算缺失值比例
        if total_rows > 0:
            null_ratios = null_counts / total_rows
            mask = null_ratios <= self.null_threshold
            
            # 打印统计信息
            print(f"\n数据质量分析结果:")
            print(f"- 分析文件数: {min(10, len(csv_files))}")
            print(f"- 总行数: {total_rows}")
            print(f"- 原始列数: {n_columns}")
            print(f"- 保留列数: {np.sum(mask)}")
            print(f"- 删除列数: {np.sum(~mask)}")
            
            # 显示被删除的列
            removed_cols = [self.column_names[i] for i in range(n_columns) if not mask[i]]
            if removed_cols:
                print(f"- 删除的列: {removed_cols}")
        
        self.mask = mask
        return self.column_names, mask
    
    def process_file(self, file_path):
        """
        处理单个文件
        
        Parameters:
        -----------
        file_path : str
            文件路径
            
        Returns:
        --------
        pandas.DataFrame : 处理后的数据框
        """
        # 读取数据
        df = pd.read_csv(file_path, encoding='gbk')
        
        # 应用列筛选
        if self.mask is not None:
            selected_columns = [col for i, col in enumerate(df.columns) 
                              if i < len(self.mask) and self.mask[i]]
            df = df[selected_columns]
        
        # 转换为数值类型
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 处理缺失值
        data = df.values
        nan_before = np.isnan(data).sum()
        
        if nan_before > 0:
            data_imputed = self.simple_imputation(data)
            nan_after = np.isnan(data_imputed).sum()
            
            if nan_after > 0:
                # 如果仍有缺失值，使用0填充
                data_imputed = np.nan_to_num(data_imputed, nan=0.0)
            
            df_imputed = pd.DataFrame(data_imputed, columns=df.columns)
        else:
            df_imputed = df
        
        return df_imputed
    
    def process_directory(self, src_dir, output_suffix="_processed"):
        """
        处理整个目录
        
        Parameters:
        -----------
        src_dir : str
            源数据目录
        output_suffix : str
            输出文件后缀
            
        Returns:
        --------
        int : 成功处理的文件数量
        """
        print("=" * 60)
        print("数据清洗和缺失值填充程序")
        print("=" * 60)
        
        # 1. 检测高缺失值列
        try:
            column_names, mask = self.detect_and_remove_high_null_columns(src_dir)
        except Exception as e:
            print(f"数据分析失败: {e}")
            return 0
        
        # 2. 处理所有文件
        csv_files = [f for f in os.listdir(src_dir) if f.endswith(".csv")]
        processed_count = 0
        
        print(f"\n开始处理 {len(csv_files)} 个文件...")
        print("-" * 40)
        
        for f in csv_files:
            try:
                file_path = os.path.join(src_dir, f)
                
                # 处理文件
                df_processed = self.process_file(file_path)
                
                # 保存结果
                output_filename = f.replace(".csv", f"{output_suffix}.csv")
                output_path = os.path.join(src_dir, output_filename)
                df_processed.to_csv(output_path, index=False, encoding='gbk')
                
                processed_count += 1
                print(f"✅ {f} -> {output_filename}")
                
            except Exception as e:
                print(f"❌ 处理文件 {f} 失败: {e}")
                continue
        
        # 3. 打印汇总信息
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        print(f"源目录: {src_dir}")
        print(f"成功处理: {processed_count}/{len(csv_files)} 个文件")
        print(f"填充方法: {self.imputation_method}")
        print(f"缺失值阈值: {self.null_threshold}")
        print(f"输出文件后缀: {output_suffix}")
        print(f"保留列数: {np.sum(mask)}/{len(mask)}")
        
        # 保存处理日志
        log_file = os.path.join(src_dir, "processing_log.txt")
        with open(log_file, 'w', encoding='gbk') as f:
            f.write(f"处理时间: {pd.Timestamp.now()}\n")
            f.write(f"源目录: {src_dir}\n")
            f.write(f"成功处理: {processed_count}/{len(csv_files)} 个文件\n")
            f.write(f"填充方法: {self.imputation_method}\n")
            f.write(f"缺失值阈值: {self.null_threshold}\n")
            f.write(f"保留列数: {np.sum(mask)}/{len(mask)}\n")
            if np.sum(~mask) > 0:
                f.write(f"删除的列: {[column_names[i] for i in range(len(mask)) if not mask[i]]}\n")
        
        print(f"\n处理日志已保存: {log_file}")
        
        return processed_count


# 使用示例
if __name__ == "__main__":
    # 配置参数
    SRC_DIR = r"D:\何维阳\20250927\20250927\data\csv"
    
    # 创建处理器实例
    processor = DataProcessor(
        null_threshold=0.25,  # 缺失值比例阈值
        imputation_method='median'  # 填充方法
    )
    
    # 处理目录中的所有文件
    processor.process_directory(SRC_DIR, output_suffix="_cleaned")