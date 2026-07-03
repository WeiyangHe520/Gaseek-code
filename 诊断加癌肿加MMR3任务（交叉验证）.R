# ============================================================
# 多任务深度学习模型：诊断 + MMR预测 + 癌肿分型（带5折交叉验证）
# ============================================================

rm(list = ls())
Sys.setenv(RETICULATE_PYTHON = "D:/anaconda/python.exe")
Sys.setenv(RETICULATE_AUTOMATIC_UVM = "false")

library(reticulate)
library(pROC)
library(caret)
library(ggplot2)

# 验证 Python 环境
py_available(initialize = TRUE)
use_python("D:/anaconda/python.exe", required = TRUE)

py_run_string("
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')
print('PyTorch版本:', torch.__version__)
")

# ============================================================
# 1. 加载原始数据
# ============================================================
load(file = ".left_data.rdata")

cat("=== 数据分布 ===\n")
cat("训练集分组分布:\n")
print(table(train_data$group))
cat("\n测试集分组分布:\n")
print(table(test_data$group))

# 查看 group2 列的唯一值（癌肿类型）
cat("\ngroup2 列的唯一值（训练集）:\n")
print(unique(train_data$group2))

# 查看 MMR 列的唯一值
cat("\nMMR.state 列的唯一值（训练集）:\n")
print(unique(train_data$MMR.state))

# 定义特征列
feature_cols <- c("gender",
                  "age",
                  "BMI",
                  "Smoking_history",
                  "drinking_history",
                  "Family_history_of_cancer",
                  "WBC",
                  "LYMPH_percentage",
                  "MONO_percentage",
                  "EO_percentage",
                  "BASO_percentage",
                  "HGB",
                  "MCV",
                  "MCHC",
                  "PLT",
                  "DBIL",
                  "IBIL",
                  "ALB",
                  "GLB",
                  "ALT",
                  "AST",
                  "GGT",
                  "ALP")

# ============================================================
# 2. 创建多任务标签函数（诊断 + MMR + 癌肿分型）
# ============================================================
create_multitask_labels <- function(data) {
  n <- nrow(data)
  
  # 任务1: 诊断标签 (所有样本)
  diag_label <- ifelse(data$group == "cancer", 1, 0)
  
  # 任务2: MMR标签 (仅cancer且有MMR信息的样本: PMMR=0, dMMR=1)
  mmr_label <- rep(NA, n)
  cancer_idx <- which(data$group == "cancer")
  
  for (idx in cancer_idx) {
    mmr_val <- data$MMR.state[idx]
    if (is.null(mmr_val) || length(mmr_val) == 0) next
    if (is.na(mmr_val)) next
    mmr_char <- as.character(mmr_val)
    if (mmr_char %in% c("", "NA", "cancer_unknown", "cancer_NA", "NULL", "null")) next
    if (mmr_char == "PMMR") {
      mmr_label[idx] <- 0
    } else if (mmr_char == "dMMR") {
      mmr_label[idx] <- 1
    }
  }
  
  # 任务3: 癌肿分型标签 (仅cancer样本: colorectal cancer=0, gastric cancer=1)
  cancer_type_label <- rep(NA, n)
  for (idx in cancer_idx) {
    type_val <- data$group2[idx]
    if (is.null(type_val) || length(type_val) == 0) next
    if (is.na(type_val)) next
    type_char <- as.character(type_val)
    if (type_char %in% c("", "NA", "NULL", "null")) next
    if (type_char == "colorectal cancer") {
      cancer_type_label[idx] <- 0
    } else if (type_char == "gastric cancer") {
      cancer_type_label[idx] <- 1
    }
  }
  
  return(data.frame(
    diag = diag_label,
    mmr = mmr_label,
    cancer_type = cancer_type_label
  ))
}

# ============================================================
# 3. 创建标签
# ============================================================
train_labels <- create_multitask_labels(train_data)
test_labels <- create_multitask_labels(test_data)

cat("\n=== 标签分布 ===\n")
cat("训练集:\n")
cat("  诊断 - 癌症:", sum(train_labels$diag == 1, na.rm = TRUE), 
    ", 对照:", sum(train_labels$diag == 0, na.rm = TRUE), "\n")
cat("  MMR - PMMR:", sum(train_labels$mmr == 0, na.rm = TRUE), 
    ", dMMR:", sum(train_labels$mmr == 1, na.rm = TRUE), "\n")
cat("  MMR - 未知(不参与训练):", sum(is.na(train_labels$mmr) & train_labels$diag == 1), "\n")
cat("  癌肿分型 - 结直肠癌:", sum(train_labels$cancer_type == 0, na.rm = TRUE), 
    ", 胃癌:", sum(train_labels$cancer_type == 1, na.rm = TRUE), "\n")
cat("  癌肿分型 - 未知(不参与训练):", sum(is.na(train_labels$cancer_type) & train_labels$diag == 1), "\n")

cat("\n测试集:\n")
cat("  诊断 - 癌症:", sum(test_labels$diag == 1, na.rm = TRUE), 
    ", 对照:", sum(test_labels$diag == 0, na.rm = TRUE), "\n")
cat("  MMR - PMMR:", sum(test_labels$mmr == 0, na.rm = TRUE), 
    ", dMMR:", sum(test_labels$mmr == 1, na.rm = TRUE), "\n")
cat("  MMR - 未知(不参与评估):", sum(is.na(test_labels$mmr) & test_labels$diag == 1), "\n")
cat("  癌肿分型 - 结直肠癌:", sum(test_labels$cancer_type == 0, na.rm = TRUE), 
    ", 胃癌:", sum(test_labels$cancer_type == 1, na.rm = TRUE), "\n")
cat("  癌肿分型 - 未知(不参与评估):", sum(is.na(test_labels$cancer_type) & test_labels$diag == 1), "\n")

# ============================================================
# 4. 提取特征矩阵
# ============================================================
X_train <- as.matrix(train_data[, feature_cols])
X_test <- as.matrix(test_data[, feature_cols])

# 处理缺失值
X_train[is.na(X_train)] <- 0
X_test[is.na(X_test)] <- 0

cat("\n特征矩阵维度:\n")
cat("训练集:", dim(X_train), "\n")
cat("测试集:", dim(X_test), "\n")

# ============================================================
# 5. 定义 PyTorch 模型（Python部分）- 3任务版本
# ============================================================
py_run_string("
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super(MultiTaskModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.shared_backbone = nn.Sequential(*layers)
        
        # 诊断头
        self.diag_head = nn.Sequential(
            nn.Linear(prev_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )
        # MMR头
        self.mmr_head = nn.Sequential(
            nn.Linear(prev_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )
        # 癌肿分型头
        self.cancer_type_head = nn.Sequential(
            nn.Linear(prev_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )
    
    def forward(self, x):
        shared = self.shared_backbone(x)
        out_diag = self.diag_head(shared)
        out_mmr = self.mmr_head(shared)
        out_type = self.cancer_type_head(shared)
        return out_diag, out_mmr, out_type

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_tasks=3):
        super(UncertaintyWeightedLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        precision = torch.exp(-self.log_vars)
        return torch.sum(precision * losses + self.log_vars)

def create_dataloader(X, y_diag, y_mmr, y_type, batch_size=128, shuffle=True):
    X_t = torch.FloatTensor(X)
    y_diag_t = torch.FloatTensor(y_diag).view(-1, 1)
    y_mmr_t = torch.FloatTensor(y_mmr).view(-1, 1)
    y_type_t = torch.FloatTensor(y_type).view(-1, 1)
    dataset = TensorDataset(X_t, y_diag_t, y_mmr_t, y_type_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
")

# ============================================================
# 6. 准备数据（转换为Python格式）
# ============================================================
y_diag_train <- train_labels$diag
y_diag_test <- test_labels$diag
y_mmr_train <- train_labels$mmr
y_mmr_test <- test_labels$mmr
y_type_train <- train_labels$cancer_type
y_type_test <- test_labels$cancer_type

# 将R数据传递给Python
py$X_train_np <- X_train
py$X_test_np <- X_test
py$y_diag_train_np <- y_diag_train
py$y_diag_test_np <- y_diag_test
py$y_mmr_train_np <- y_mmr_train
py$y_mmr_test_np <- y_mmr_test
py$y_type_train_np <- y_type_train
py$y_type_test_np <- y_type_test

# 在Python中转换为numpy数组
py_run_string("
# 将R矩阵转换为numpy数组
X_train_np = np.array(X_train_np, dtype=np.float32)
X_test_np = np.array(X_test_np, dtype=np.float32)
y_diag_train_np = np.array(y_diag_train_np, dtype=np.int64)
y_diag_test_np = np.array(y_diag_test_np, dtype=np.int64)
y_mmr_train_np = np.array(y_mmr_train_np, dtype=np.float64)
y_mmr_test_np = np.array(y_mmr_test_np, dtype=np.float64)
y_type_train_np = np.array(y_type_train_np, dtype=np.float64)
y_type_test_np = np.array(y_type_test_np, dtype=np.float64)

# 将NaN转换为-1以便在numpy中处理
y_mmr_train_np = np.nan_to_num(y_mmr_train_np, nan=-1)
y_mmr_test_np = np.nan_to_num(y_mmr_test_np, nan=-1)
y_type_train_np = np.nan_to_num(y_type_train_np, nan=-1)
y_type_test_np = np.nan_to_num(y_type_test_np, nan=-1)

print(f'训练集形状: {X_train_np.shape}')
print(f'测试集形状: {X_test_np.shape}')
print(f'训练集诊断标签形状: {y_diag_train_np.shape}')
print(f'训练集MMR标签分布: PMMR={np.sum(y_mmr_train_np==0)}, dMMR={np.sum(y_mmr_train_np==1)}, 未知={np.sum(y_mmr_train_np==-1)}')
print(f'训练集癌肿分型标签分布: 结直肠癌={np.sum(y_type_train_np==0)}, 胃癌={np.sum(y_type_train_np==1)}, 未知={np.sum(y_type_train_np==-1)}')
")

# ============================================================
# 7. 定义单折训练函数（用于交叉验证）- 3任务版本
# ============================================================
py_run_string("
def train_single_fold_multi_3tasks(X_train_fold, y_diag_fold, y_mmr_fold, y_type_fold,
                                   X_val_fold, y_diag_val, y_mmr_val, y_type_val,
                                   input_dim=23, epochs=100, lr=0.001, batch_size=128):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据加载器
    train_loader = create_dataloader(X_train_fold, y_diag_fold, y_mmr_fold, y_type_fold, batch_size, True)
    val_loader = create_dataloader(X_val_fold, y_diag_val, y_mmr_val, y_type_val, batch_size, False)
    
    model = MultiTaskModel(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout=0.3).to(device)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    unc_loss = UncertaintyWeightedLoss(num_tasks=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    best_model_state = None
    fold_history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_d, y_m, y_t in train_loader:
            X_batch, y_d, y_m, y_t = X_batch.to(device), y_d.to(device), y_m.to(device), y_t.to(device)
            out_d, out_m, out_t = model(X_batch)
            
            # 诊断损失（所有样本）
            loss_d = bce_loss(out_d, y_d).mean()
            
            # MMR损失（仅有效样本）
            valid_m = (y_m != -1)
            if valid_m.any():
                loss_m = bce_loss(out_m[valid_m], y_m[valid_m]).mean()
            else:
                loss_m = torch.tensor(0.0).to(device)
            
            # 癌肿分型损失（仅有效样本）
            valid_t = (y_t != -1)
            if valid_t.any():
                loss_t = bce_loss(out_t[valid_t], y_t[valid_t]).mean()
            else:
                loss_t = torch.tensor(0.0).to(device)
            
            total_loss = unc_loss(torch.stack([loss_d, loss_m, loss_t]))
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_d, y_m, y_t in val_loader:
                X_batch, y_d, y_m, y_t = X_batch.to(device), y_d.to(device), y_m.to(device), y_t.to(device)
                out_d, out_m, out_t = model(X_batch)
                
                loss_d = bce_loss(out_d, y_d).mean()
                valid_m = (y_m != -1)
                if valid_m.any():
                    loss_m = bce_loss(out_m[valid_m], y_m[valid_m]).mean()
                else:
                    loss_m = torch.tensor(0.0).to(device)
                valid_t = (y_t != -1)
                if valid_t.any():
                    loss_t = bce_loss(out_t[valid_t], y_t[valid_t]).mean()
                else:
                    loss_t = torch.tensor(0.0).to(device)
                
                total_loss = unc_loss(torch.stack([loss_d, loss_m, loss_t]))
                val_loss += total_loss.item()
        
        val_loss /= len(val_loader)
        fold_history['train_loss'].append(train_loss)
        fold_history['val_loss'].append(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    return model, fold_history, best_model_state
")

# ============================================================
# 8. 5折交叉验证函数 - 3任务版本
# ============================================================
py_run_string("
def perform_cv_multitask_3tasks(X, y_diag, y_mmr, y_type, nfolds=5, epochs=100, lr=0.001, batch_size=128):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 使用诊断标签进行分层抽样（确保每折癌症/对照比例一致）
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=3456)
    
    cv_results = []
    cv_predictions_diag = []
    cv_predictions_mmr = []
    cv_predictions_type = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_diag), 1):
        print('\\n--- 交叉验证第 {} 折 ---'.format(fold))
        
        train_idx = train_idx.astype(int)
        val_idx = val_idx.astype(int)
        
        X_train_fold = X[train_idx]
        y_diag_train_fold = y_diag[train_idx]
        y_mmr_train_fold = y_mmr[train_idx]
        y_type_train_fold = y_type[train_idx]
        X_val_fold = X[val_idx]
        y_diag_val_fold = y_diag[val_idx]
        y_mmr_val_fold = y_mmr[val_idx]
        y_type_val_fold = y_type[val_idx]
        
        print('  训练集样本数: {}'.format(len(train_idx)))
        print('  验证集样本数: {}'.format(len(val_idx)))
        print('  验证集癌症数: {}, 对照数: {}'.format(np.sum(y_diag_val_fold), len(y_diag_val_fold) - np.sum(y_diag_val_fold)))
        print('  验证集MMR - PMMR: {}, dMMR: {}'.format(np.sum(y_mmr_val_fold==0), np.sum(y_mmr_val_fold==1)))
        print('  验证集癌肿分型 - 结直肠癌: {}, 胃癌: {}'.format(np.sum(y_type_val_fold==0), np.sum(y_type_val_fold==1)))
        
        # 训练模型
        model, fold_history, best_state = train_single_fold_multi_3tasks(
            X_train_fold, y_diag_train_fold, y_mmr_train_fold, y_type_train_fold,
            X_val_fold, y_diag_val_fold, y_mmr_val_fold, y_type_val_fold,
            input_dim=X.shape[1], epochs=epochs, lr=lr, batch_size=batch_size
        )
        
        # 使用最佳模型进行验证集预测
        model.load_state_dict(best_state)
        model = model.to(device)
        model.eval()
        
        X_val_t = torch.FloatTensor(X_val_fold).to(device)
        with torch.no_grad():
            out_d, out_m, out_t = model(X_val_t)
            pred_diag = torch.sigmoid(out_d).cpu().numpy().flatten()
            pred_mmr = torch.sigmoid(out_m).cpu().numpy().flatten()
            pred_type = torch.sigmoid(out_t).cpu().numpy().flatten()
        
        cv_predictions_diag.append(pred_diag)
        cv_predictions_mmr.append(pred_mmr)
        cv_predictions_type.append(pred_type)
        
        # 计算验证集指标 - 诊断任务
        try:
            auc_diag = roc_auc_score(y_diag_val_fold, pred_diag)
        except:
            auc_diag = 0.5
        
        # 计算最佳阈值 (Youden指数)
        fpr, tpr, thresholds = roc_curve(y_diag_val_fold, pred_diag)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        # 分类
        pred_class = (pred_diag > best_thresh).astype(int)
        
        # 计算指标
        cm = confusion_matrix(y_diag_val_fold, pred_class)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        acc = accuracy_score(y_diag_val_fold, pred_class)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        ydi = sens + spec - 1
        f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0
        brier = brier_score_loss(y_diag_val_fold, pred_diag)
        
        # 计算MMR任务AUC（仅有效样本）
        valid_mmr = (y_mmr_val_fold != -1)
        if np.sum(valid_mmr) > 0 and len(np.unique(y_mmr_val_fold[valid_mmr])) > 1:
            auc_mmr = roc_auc_score(y_mmr_val_fold[valid_mmr], pred_mmr[valid_mmr])
        else:
            auc_mmr = np.nan
        
        # 计算癌肿分型任务AUC（仅有效样本）
        valid_type = (y_type_val_fold != -1)
        if np.sum(valid_type) > 0 and len(np.unique(y_type_val_fold[valid_type])) > 1:
            auc_type = roc_auc_score(y_type_val_fold[valid_type], pred_type[valid_type])
        else:
            auc_type = np.nan
        
        # 存储结果
        fold_result = {
            'Fold': fold,
            'AUC_Diag': auc_diag,
            'AUC_MMR': auc_mmr,
            'AUC_Type': auc_type,
            'Brier_Score': brier,
            'ACC': acc,
            'SENS': sens,
            'SPEC': spec,
            'PPV': ppv,
            'NPV': npv,
            'YDI': ydi,
            'F1': f1,
            'Best_Threshold': best_thresh,
            'N_Val': len(val_idx),
            'N_MMR_Valid': np.sum(valid_mmr),
            'N_Type_Valid': np.sum(valid_type)
        }
        cv_results.append(fold_result)
        
        print('  诊断 AUC = {:.3f}'.format(auc_diag))
        if not np.isnan(auc_mmr):
            print('  MMR AUC = {:.3f}'.format(auc_mmr))
        else:
            print('  MMR AUC = N/A')
        if not np.isnan(auc_type):
            print('  癌肿分型 AUC = {:.3f}'.format(auc_type))
        else:
            print('  癌肿分型 AUC = N/A')
        print('  ACC = {:.3f}'.format(acc))
        print('  SENS = {:.3f}'.format(sens))
        print('  SPEC = {:.3f}'.format(spec))
    
    # 汇总结果
    import pandas as pd
    cv_df = pd.DataFrame(cv_results)
    
    # 诊断任务汇总
    diag_summary = {
        'Metric': ['AUC', 'Brier_Score', 'ACC', 'SENS', 'SPEC', 'PPV', 'NPV', 'YDI', 'F1'],
        'Mean': [cv_df['AUC_Diag'].mean(), cv_df['Brier_Score'].mean(), cv_df['ACC'].mean(),
                 cv_df['SENS'].mean(), cv_df['SPEC'].mean(), cv_df['PPV'].mean(),
                 cv_df['NPV'].mean(), cv_df['YDI'].mean(), cv_df['F1'].mean()],
        'SD': [cv_df['AUC_Diag'].std(), cv_df['Brier_Score'].std(), cv_df['ACC'].std(),
               cv_df['SENS'].std(), cv_df['SPEC'].std(), cv_df['PPV'].std(),
               cv_df['NPV'].std(), cv_df['YDI'].std(), cv_df['F1'].std()]
    }
    diag_summary_df = pd.DataFrame(diag_summary)
    
    # MMR任务汇总
    mmr_auc_mean = cv_df['AUC_MMR'].mean()
    mmr_auc_std = cv_df['AUC_MMR'].std()
    mmr_valid_n = cv_df['N_MMR_Valid'].sum()
    
    # 癌肿分型任务汇总
    type_auc_mean = cv_df['AUC_Type'].mean()
    type_auc_std = cv_df['AUC_Type'].std()
    type_valid_n = cv_df['N_Type_Valid'].sum()
    
    return cv_df, diag_summary_df, cv_predictions_diag, cv_predictions_mmr, cv_predictions_type, mmr_auc_mean, mmr_auc_std, mmr_valid_n, type_auc_mean, type_auc_std, type_valid_n
")

# ============================================================
# 9. 执行5折交叉验证
# ============================================================
cat("\n========================================\n")
cat("开始5折交叉验证（诊断 + MMR + 癌肿分型）\n")
cat("========================================\n")

py_run_string("
cv_df, diag_summary_df, cv_predictions_diag, cv_predictions_mmr, cv_predictions_type, mmr_auc_mean, mmr_auc_std, mmr_valid_n, type_auc_mean, type_auc_std, type_valid_n = perform_cv_multitask_3tasks(
    X_train_np, y_diag_train_np, y_mmr_train_np, y_type_train_np,
    nfolds=5, epochs=100, lr=0.001, batch_size=128
)
")

# 将交叉验证结果传递到R
cv_results_df <- py$cv_df
cv_summary_df <- py$diag_summary_df
mmr_auc_mean <- py$mmr_auc_mean
mmr_auc_std <- py$mmr_auc_std
mmr_valid_n <- py$mmr_valid_n
type_auc_mean <- py$type_auc_mean
type_auc_std <- py$type_auc_std
type_valid_n <- py$type_valid_n

# 保存交叉验证结果
write.csv(cv_results_df, "./multitask_cv_fold_results_3tasks.csv", row.names = FALSE)
write.csv(cv_summary_df, "./multitask_cv_summary_3tasks.csv", row.names = FALSE)

cat("\n========================================\n")
cat("交叉验证结果汇总 - 诊断任务 (5折平均)\n")
cat("========================================\n")
print(cv_summary_df)

cat("\n=== MMR任务交叉验证结果 ===\n")
cat(sprintf("  AUC (5折平均) = %.3f (SD: %.3f)\n", mmr_auc_mean, mmr_auc_std))
cat(sprintf("  有效MMR样本总数 = %d\n", mmr_valid_n))

cat("\n=== 癌肿分型任务交叉验证结果 ===\n")
cat(sprintf("  AUC (5折平均) = %.3f (SD: %.3f)\n", type_auc_mean, type_auc_std))
cat(sprintf("  有效癌肿分型样本总数 = %d\n", type_valid_n))

# ============================================================
# 10. 训练最终模型（使用全部训练数据）
# ============================================================
cat("\n========================================\n")
cat("训练最终模型（使用全部训练数据）\n")
cat("========================================\n")

py_run_string("
def train_final_model_multi_3tasks(X_train, y_diag_train, y_mmr_train, y_type_train,
                                   input_dim=23, epochs=100, lr=0.001, batch_size=128):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'训练配置: epochs={epochs}, lr={lr}, batch_size={batch_size}')
    
    train_loader = create_dataloader(X_train, y_diag_train, y_mmr_train, y_type_train, batch_size, True)
    
    model = MultiTaskModel(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout=0.3).to(device)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    unc_loss = UncertaintyWeightedLoss(num_tasks=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {'train_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_d, y_m, y_t in train_loader:
            X_batch, y_d, y_m, y_t = X_batch.to(device), y_d.to(device), y_m.to(device), y_t.to(device)
            out_d, out_m, out_t = model(X_batch)
            
            loss_d = bce_loss(out_d, y_d).mean()
            valid_m = (y_m != -1)
            if valid_m.any():
                loss_m = bce_loss(out_m[valid_m], y_m[valid_m]).mean()
            else:
                loss_m = torch.tensor(0.0).to(device)
            valid_t = (y_t != -1)
            if valid_t.any():
                loss_t = bce_loss(out_t[valid_t], y_t[valid_t]).mean()
            else:
                loss_t = torch.tensor(0.0).to(device)
            
            total_loss = unc_loss(torch.stack([loss_d, loss_m, loss_t]))
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}')
    
    return model, history

# 训练最终模型
model, history = train_final_model_multi_3tasks(
    X_train_np, y_diag_train_np, y_mmr_train_np, y_type_train_np,
    input_dim=23, epochs=100, lr=0.001, batch_size=128
)
")

# ============================================================
# 11. 训练集和测试集预测
# ============================================================
py_run_string("
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 训练集预测
X_train_t = torch.FloatTensor(X_train_np).to(device)
with torch.no_grad():
    out_d_train, out_m_train, out_t_train = model(X_train_t)
    train_pred_diag = torch.sigmoid(out_d_train).cpu().numpy().flatten()
    train_pred_mmr = torch.sigmoid(out_m_train).cpu().numpy().flatten()
    train_pred_type = torch.sigmoid(out_t_train).cpu().numpy().flatten()

# 测试集预测
X_test_t = torch.FloatTensor(X_test_np).to(device)
with torch.no_grad():
    out_d_test, out_m_test, out_t_test = model(X_test_t)
    test_pred_diag = torch.sigmoid(out_d_test).cpu().numpy().flatten()
    test_pred_mmr = torch.sigmoid(out_m_test).cpu().numpy().flatten()
    test_pred_type = torch.sigmoid(out_t_test).cpu().numpy().flatten()
")

# 传递所有预测结果到R
train_pred_diag <- py$train_pred_diag
train_pred_mmr <- py$train_pred_mmr
train_pred_type <- py$train_pred_type
test_pred_diag <- py$test_pred_diag
test_pred_mmr <- py$test_pred_mmr
test_pred_type <- py$test_pred_type

# ============================================================
# 12. 完整评估函数（与glmnet格式一致）
# ============================================================
calculate_full_metrics <- function(true_labels, pred_probs, dataset_name) {
  
  valid_idx <- !is.na(true_labels)
  if (sum(valid_idx) == 0) {
    cat(dataset_name, ": 无有效样本\n")
    return(NULL)
  }
  
  true_valid <- true_labels[valid_idx]
  pred_valid <- pred_probs[valid_idx]
  
  if (length(unique(true_valid)) < 2) {
    cat(dataset_name, ": 只有一个类别，无法计算AUC\n")
    return(NULL)
  }
  
  roc_obj <- roc(true_valid, pred_valid, ci = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  auc_ci <- ci.auc(roc_obj, conf.level = 0.95)
  
  coords <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"))
  best_thresh <- coords[1, "threshold"]
  
  pred_class <- ifelse(pred_valid > best_thresh, 