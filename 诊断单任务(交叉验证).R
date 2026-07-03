# ============================================================
# 单任务深度学习模型：仅诊断（带5折交叉验证）- 最终修复版
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
# 2. 创建诊断标签函数
# ============================================================
create_diag_labels <- function(data) {
  diag_label <- ifelse(data$group == "cancer", 1, 0)
  return(data.frame(diag = diag_label))
}

# ============================================================
# 3. 创建标签
# ============================================================
train_labels <- create_diag_labels(train_data)
test_labels <- create_diag_labels(test_data)

cat("\n=== 标签分布 ===\n")
cat("训练集:\n")
cat("  诊断 - 癌症:", sum(train_labels$diag == 1, na.rm = TRUE), 
    ", 对照:", sum(train_labels$diag == 0, na.rm = TRUE), "\n")

cat("\n测试集:\n")
cat("  诊断 - 癌症:", sum(test_labels$diag == 1, na.rm = TRUE), 
    ", 对照:", sum(test_labels$diag == 0, na.rm = TRUE), "\n")

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
# 5. 定义 PyTorch 模型
# ============================================================
py_run_string("
class SingleTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super(SingleTaskModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        
        # 诊断头
        self.diag_head = nn.Sequential(
            nn.Linear(prev_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )
    
    def forward(self, x):
        shared = self.backbone(x)
        out_diag = self.diag_head(shared)
        return out_diag

def create_dataloader(X, y_diag, batch_size=128, shuffle=True):
    X_t = torch.FloatTensor(X)
    y_diag_t = torch.FloatTensor(y_diag).reshape(-1, 1)
    dataset = TensorDataset(X_t, y_diag_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
")

# ============================================================
# 6. 准备数据 - 直接在Python中处理
# ============================================================
y_diag_train <- train_labels$diag
y_diag_test <- test_labels$diag

# 将R数据传递给Python
py$X_train_r <- X_train
py$X_test_r <- X_test
py$y_diag_train_r <- y_diag_train
py$y_diag_test_r <- y_diag_test

# 在Python中转换为numpy数组
py_run_string("
# 将R矩阵转换为numpy数组
X_train_np = np.array(X_train_r, dtype=np.float32)
X_test_np = np.array(X_test_r, dtype=np.float32)
y_diag_train_np = np.array(y_diag_train_r, dtype=np.int64)
y_diag_test_np = np.array(y_diag_test_r, dtype=np.int64)

print(f'训练集形状: {X_train_np.shape}')
print(f'测试集形状: {X_test_np.shape}')
print(f'训练集标签形状: {y_diag_train_np.shape}')
print(f'测试集标签形状: {y_diag_test_np.shape}')
")

# ============================================================
# 7. 训练函数
# ============================================================
py_run_string("
def train_single_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                      input_dim=23, epochs=100, lr=0.001, batch_size=128):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader = create_dataloader(X_train_fold, y_train_fold, batch_size, True)
    val_loader = create_dataloader(X_val_fold, y_val_fold, batch_size, False)
    
    model = SingleTaskModel(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout=0.3).to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    fold_history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_d in train_loader:
            X_batch, y_d = X_batch.to(device), y_d.to(device)
            out_d = model(X_batch)
            
            loss_d = bce_loss(out_d, y_d)
            
            optimizer.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss_d.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_d in val_loader:
                X_batch, y_d = X_batch.to(device), y_d.to(device)
                out_d = model(X_batch)
                loss_d = bce_loss(out_d, y_d)
                val_loss += loss_d.item()
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
# 8. 5折交叉验证函数 - 直接在Python中处理索引
# ============================================================
py_run_string("
def perform_cv(X, y_diag, nfolds=5, epochs=100, lr=0.001, batch_size=128):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=3456)
    
    cv_results = []
    cv_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_diag), 1):
        print(f'\\n--- 交叉验证第 {fold}/{nfolds} 折 ---')
        
        # 使用整数索引获取数据
        train_idx = train_idx.astype(int)
        val_idx = val_idx.astype(int)
        
        X_train_fold = X[train_idx]
        y_train_fold = y_diag[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y_diag[val_idx]
        
        print(f'  训练集样本数: {len(train_idx)}')
        print(f'  验证集样本数: {len(val_idx)}')
        print(f'  验证集癌症数: {np.sum(y_val_fold)}, 对照数: {len(y_val_fold) - np.sum(y_val_fold)}')
        
        # 训练模型
        model, fold_history, best_state = train_single_fold(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold,
            input_dim=X.shape[1], epochs=epochs, lr=lr, batch_size=batch_size
        )
        
        # 使用最佳模型进行验证集预测
        model.load_state_dict(best_state)
        model = model.to(device)
        model.eval()
        
        X_val_t = torch.FloatTensor(X_val_fold).to(device)
        with torch.no_grad():
            out_d = model(X_val_t)
            pred_prob = torch.sigmoid(out_d).cpu().numpy().flatten()
        
        cv_predictions.append(pred_prob)
        
        # 计算验证集指标
        try:
            auc_val = roc_auc_score(y_val_fold, pred_prob)
        except:
            auc_val = 0.5
        
        # 计算最佳阈值 (Youden指数)
        fpr, tpr, thresholds = roc_curve(y_val_fold, pred_prob)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        # 分类
        pred_class = (pred_prob > best_thresh).astype(int)
        
        # 计算指标
        cm = confusion_matrix(y_val_fold, pred_class)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        acc = accuracy_score(y_val_fold, pred_class)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        ydi = sens + spec - 1
        f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0
        brier = brier_score_loss(y_val_fold, pred_prob)
        
        # 存储结果
        fold_result = {
            'Fold': fold,
            'AUC': auc_val,
            'Brier_Score': brier,
            'ACC': acc,
            'SENS': sens,
            'SPEC': spec,
            'PPV': ppv,
            'NPV': npv,
            'YDI': ydi,
            'F1': f1,
            'Best_Threshold': best_thresh,
            'N_Val': len(val_idx)
        }
        cv_results.append(fold_result)
        
        print(f'  AUC = {auc_val:.3f}')
        print(f'  ACC = {acc:.3f}')
        print(f'  SENS = {sens:.3f}')
        print(f'  SPEC = {spec:.3f}')
    
    # 汇总结果
    import pandas as pd
    cv_df = pd.DataFrame(cv_results)
    
    summary = {
        'Metric': ['AUC', 'Brier_Score', 'ACC', 'SENS', 'SPEC', 'PPV', 'NPV', 'YDI', 'F1'],
        'Mean': [cv_df['AUC'].mean(), cv_df['Brier_Score'].mean(), cv_df['ACC'].mean(),
                 cv_df['SENS'].mean(), cv_df['SPEC'].mean(), cv_df['PPV'].mean(),
                 cv_df['NPV'].mean(), cv_df['YDI'].mean(), cv_df['F1'].mean()],
        'SD': [cv_df['AUC'].std(), cv_df['Brier_Score'].std(), cv_df['ACC'].std(),
               cv_df['SENS'].std(), cv_df['SPEC'].std(), cv_df['PPV'].std(),
               cv_df['NPV'].std(), cv_df['YDI'].std(), cv_df['F1'].std()]
    }
    summary_df = pd.DataFrame(summary)
    
    return cv_df, summary_df, cv_predictions
")

# ============================================================
# 9. 执行5折交叉验证
# ============================================================
cat("\n========================================\n")
cat("开始5折交叉验证\n")
cat("========================================\n")

py_run_string("
cv_df, summary_df, cv_predictions = perform_cv(
    X_train_np, y_diag_train_np,
    nfolds=5, epochs=100, lr=0.001, batch_size=128
)
")

# 将交叉验证结果传递到R
cv_results_df <- py$cv_df
cv_summary_df <- py$summary_df

# 保存交叉验证结果
write.csv(cv_results_df, "./singletask_cv_fold_results.csv", row.names = FALSE)
write.csv(cv_summary_df, "./singletask_cv_summary.csv", row.names = FALSE)

cat("\n========================================\n")
cat("交叉验证结果汇总 (5折平均)\n")
cat("========================================\n")
print(cv_summary_df)

# ============================================================
# 10. 训练最终模型
# ============================================================
cat("\n========================================\n")
cat("训练最终模型（使用全部训练数据）\n")
cat("========================================\n")

py_run_string("
def train_final_model(X_train, y_diag_train, input_dim=23, epochs=100, lr=0.001, batch_size=128):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'训练配置: epochs={epochs}, lr={lr}, batch_size={batch_size}')
    
    train_loader = create_dataloader(X_train, y_diag_train, batch_size, True)
    
    model = SingleTaskModel(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout=0.3).to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {'train_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_d in train_loader:
            X_batch, y_d = X_batch.to(device), y_d.to(device)
            out_d = model(X_batch)
            
            loss_d = bce_loss(out_d, y_d)
            
            optimizer.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss_d.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}')
    
    return model, history

# 训练最终模型
model, history = train_final_model(
    X_train_np, y_diag_train_np,
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
    out_d_train = model(X_train_t)
    train_pred_diag = torch.sigmoid(out_d_train).cpu().numpy().flatten()

# 测试集预测
X_test_t = torch.FloatTensor(X_test_np).to(device)
with torch.no_grad():
    out_d_test = model(X_test_t)
    test_pred_diag = torch.sigmoid(out_d_test).cpu().numpy().flatten()
")

# 传递预测结果到R
train_pred_diag <- py$train_pred_diag
test_pred_diag <- py$test_pred_diag

# ============================================================
# 12. 完整评估函数
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
  
  pred_class <- ifelse(pred_valid > best_thresh, 1, 0)
  cm <- confusionMatrix(as.factor(pred_class), as.factor(true_valid), positive = "1")
  
  brier_score <- mean((pred_valid - true_valid)^2)
  
  sens <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  ydi <- sens + spec - 1
  
  ppv <- cm$byClass["Pos Pred Value"]
  f1 <- 2 * (ppv * sens) / (ppv + sens)
  
  result <- data.frame(
    Dataset = dataset_name,
    AUC = round(auc_val, 3),
    AUC_95CI = sprintf("%.3f (%.3f-%.3f)", auc_val, auc_ci[1], auc_ci[3]),
    AUC_lower = round(auc_ci[1], 3),
    AUC_upper = round(auc_ci[3], 3),
    Brier_Score = round(brier_score, 4),
    ACC = round(cm$overall["Accuracy"], 3),
    SENS = round(sens, 3),
    SPEC = round(spec, 3),
    PPV = round(ppv, 3),
    NPV = round(cm$byClass["Neg Pred Value"], 3),
    YDI = round(ydi, 3),
    F1 = round(f1, 3)
  )
  
  return(result)
}

# ============================================================
# 13. 计算训练集和测试集指标
# ============================================================
cat("\n========================================\n")
cat("最终模型评估结果\n")
cat("========================================\n")

train_metrics <- calculate_full_metrics(train_labels$diag, train_pred_diag, "Training")
test_metrics <- calculate_full_metrics(test_labels$diag, test_pred_diag, "Testing")

results_df <- rbind(train_metrics, test_metrics)
cat("\n=== 诊断任务结果汇总 ===\n")
print(results_df)

# 保存结果
write.csv(results_df, "./singletask_diag_metrics.csv", row.names = FALSE)
cat("\n结果已保存到: ./singletask_diag_metrics.csv\n")

# ============================================================
# 14. 详细性能报告
# ============================================================
cat("\n========================================\n")
cat("详细性能报告\n")
cat("========================================\n")

cat("\n=== 5折交叉验证平均值 ===\n")
for (i in 1:nrow(cv_summary_df)) {
  cat(sprintf("  %s = %.3f (SD: %.3f)\n", 
              cv_summary_df$Metric[i],
              cv_summary_df$Mean[i],
              cv_summary_df$SD[i]))
}

cat("\n=== 最终模型性能 ===\n")

cat("\n训练集:\n")
cat(sprintf("  AUC = %.3f (95%% CI: %.3f-%.3f)\n", 
            train_metrics$AUC, train_metrics$AUC_lower, train_metrics$AUC_upper))
cat(sprintf("  ACC = %.3f\n", train_metrics$ACC))
cat(sprintf("  SENS = %.3f\n", train_metrics$SENS))
cat(sprintf("  SPEC = %.3f\n", train_metrics$SPEC))
cat(sprintf("  PPV = %.3f\n", train_metrics$PPV))
cat(sprintf("  NPV = %.3f\n", train_metrics$NPV))
cat(sprintf("  F1 = %.3f\n", train_metrics$F1))
cat(sprintf("  Brier = %.4f\n", train_metrics$Brier_Score))

cat("\n测试集:\n")
cat(sprintf("  AUC = %.3f (95%% CI: %.3f-%.3f)\n", 
            test_metrics$AUC, test_metrics$AUC_lower, test_metrics$AUC_upper))
cat(sprintf("  ACC = %.3f\n", test_metrics$ACC))
cat(sprintf("  SENS = %.3f\n", test_metrics$SENS))
cat(sprintf("  SPEC = %.3f\n", test_metrics$SPEC))
cat(sprintf("  PPV = %.3f\n", test_metrics$PPV))
cat(sprintf("  NPV = %.3f\n", test_metrics$NPV))
cat(sprintf("  F1 = %.3f\n", test_metrics$F1))
cat(sprintf("  Brier = %.4f\n", test_metrics$Brier_Score))

# ============================================================
# 15. 绘制训练曲线
# ============================================================
train_loss_vec <- tryCatch(unlist(py$history['train_loss']), error = function(e) NULL)

if (!is.null(train_loss_vec) && length(train_loss_vec) > 0) {
  loss_df <- data.frame(Epoch = 1:length(train_loss_vec), Train_Loss = train_loss_vec)
  
  p <- ggplot(loss_df, aes(x = Epoch, y = Train_Loss)) +
    geom_line(color = "blue", linewidth = 1) +
    labs(title = "训练过程 - 损失曲线（单任务）", y = "Loss", x = "Epoch") +
    theme_minimal()
  
  ggsave("./singletask_loss_curve.png", p, width = 8, height = 6)
  cat("\n损失曲线已保存: ./singletask_loss_curve.png\n")
}

# ============================================================
# 16. 绘制ROC曲线
# ============================================================
valid_diag <- !is.na(test_labels$diag)
if (sum(valid_diag) > 0 && length(unique(test_labels$diag[valid_diag])) > 1) {
  png("./singletask_diag_roc.png", width = 6, height = 6, units = "in", res = 300)
  roc_diag <- roc(test_labels$diag[valid_diag], test_pred_diag[valid_diag])
  plot(roc_diag, main = paste0("单任务诊断 ROC\nAUC = ", round(roc_diag$auc, 3)), 
       col = "blue", lwd = 2)
  dev.off()
  cat("诊断ROC曲线已保存: ./singletask_diag_roc.png\n")
}

# ============================================================
# 17. 保存模型
# ============================================================


# ============================================================
# 18. 与多任务模型对比（如果存在）
# ============================================================
cat("\n========================================\n")
cat("与多任务模型对比\n")
cat("========================================\n")

if (file.exists("./multitask_diag_metrics_4tasks.csv")) {
  multitask_results <- read.csv("./multitask_diag_metrics_4tasks.csv")
  
  cat("\n单任务模型（仅诊断）:\n")
  cat(sprintf("  训练集 AUC: %.3f\n", train_metrics$AUC))
  cat(sprintf("  测试集 AUC: %.3f\n", test_metrics$AUC))
  cat(sprintf("  测试集 SENS: %.3f\n", test_metrics$SENS))
  cat(sprintf("  测试集 SPEC: %.3f\n", test_metrics$SPEC))
  cat(sprintf("  测试集 F1: %.3f\n", test_metrics$F1))
  
  cat("\n多任务模型（诊断+MMR+癌肿分型+分期）:\n")
  cat(sprintf("  训练集 AUC: %.3f\n", multitask_results$AUC[1]))
  cat(sprintf("  测试集 AUC: %.3f\n", multitask_results$AUC[2]))
  
  cat("\n=== 对比结论 ===\n")
  diff_auc <- test_metrics$AUC - multitask_results$AUC[2]
  if (diff_auc > 0) {
    cat(sprintf("单任务模型 AUC 比多任务模型高 %.3f\n", diff_auc))
    cat("结论：其他任务（MMR、癌肿分型、分期）对诊断任务有负面影响\n")
  } else if (diff_auc < 0) {
    cat(sprintf("多任务模型 AUC 比单任务模型高 %.3f\n", abs(diff_auc)))
    cat("结论：其他任务（MMR、癌肿分型、分期）对诊断任务有正面加成作用\n")
  } else {
    cat("两个模型 AUC 相当\n")
    cat("结论：其他任务对诊断任务无显著影响\n")
  }
} else {
  cat("未找到多任务模型结果文件，跳过对比\n")
  cat("请先运行多任务模型生成 ./multitask_diag_metrics_4tasks.csv\n")
}

cat("\n========================================\n")
cat("单任务模型（带交叉验证）训练完成！\n")
cat("========================================\n")