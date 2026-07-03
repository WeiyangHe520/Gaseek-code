# ============================================================
# 单任务深度学习模型：仅诊断
# （使用全部训练数据，无验证集，100轮 - 最优配置）
# 用于对比多任务学习的效果
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
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
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
  n <- nrow(data)
  
  # 诊断标签 (所有样本)
  diag_label <- ifelse(data$group == "cancer", 1, 0)
  
  return(data.frame(
    diag = diag_label
  ))
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
# 5. 定义 PyTorch 模型（单任务版本）
# ============================================================
py_run_string("
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score

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
    y_diag_t = torch.FloatTensor(y_diag).view(-1, 1)
    dataset = TensorDataset(X_t, y_diag_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
")

# ============================================================
# 6. 准备数据（转换为Python格式）
# ============================================================
y_diag_train <- train_labels$diag
y_diag_test <- test_labels$diag

py$X_train <- X_train
py$X_test <- X_test
py$y_diag_train <- y_diag_train
py$y_diag_test <- y_diag_test

# ============================================================
# 7. 使用全部训练数据
# ============================================================
cat("\n=== 数据划分 ===\n")
cat("训练集样本数:", nrow(X_train), "\n")
cat("测试集样本数:", nrow(X_test), "\n")

py$X_train_final <- X_train
py$y_diag_train_final <- y_diag_train

# ============================================================
# 8. 训练函数（单任务）
# ============================================================
py_run_string("
import torch.optim as optim

def train_singletask_model(X_train, y_diag_train,
                            input_dim=23, epochs=100, lr=0.001, batch_size=128):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'训练配置: epochs={epochs}, lr={lr}, batch_size={batch_size}')
    print(f'网络结构: [128, 64, 32], dropout=0.3, weight_decay=1e-5')
    
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
")

# ============================================================
# 9. 训练模型
# ============================================================
cat("\n========================================\n")
cat("开始训练单任务模型（仅诊断）...\n")
cat("网络结构: [128, 64, 32], dropout=0.3\n")
cat("学习率: 0.001, 轮数: 100, weight_decay: 1e-5\n")
cat("========================================\n\n")

py_run_string("
model, history = train_singletask_model(
    X_train_final, y_diag_train_final,
    input_dim=23, epochs=100, lr=0.001, batch_size=128
)
")

# ============================================================
# 10. 训练集和测试集预测
# ============================================================
py_run_string("
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 训练集预测
X_train_t = torch.FloatTensor(X_train).to(device)
with torch.no_grad():
    out_d_train = model(X_train_t)
    train_pred_diag = torch.sigmoid(out_d_train).cpu().numpy().flatten()

# 测试集预测
X_test_t = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    out_d_test = model(X_test_t)
    test_pred_diag = torch.sigmoid(out_d_test).cpu().numpy().flatten()
")

# 传递预测结果到R
train_pred_diag <- py$train_pred_diag
test_pred_diag <- py$test_pred_diag

# ============================================================
# 11. 完整评估函数
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
# 12. 计算训练集和测试集指标（诊断任务）
# ============================================================
cat("\n========================================\n")
cat("单任务模型评估结果（仅诊断）\n")
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
# 13. 与多任务模型对比（如果存在）
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

# ============================================================
# 14. 绘制训练曲线
# ============================================================
train_loss_vec <- tryCatch(unlist(py$history['train_loss']), error = function(e) NULL)

if (!is.null(train_loss_vec) && length(train_loss_vec) > 0) {
  loss_df <- data.frame(Epoch = 1:length(train_loss_vec), Train_Loss = train_loss_vec)
  
  p <- ggplot(loss_df, aes(x = Epoch, y = Train_Loss)) +
    geom_line(color = "blue", linewidth = 1) +
    labs(title = "训练过程 - 损失曲线（单任务）", y = "Loss", x = "Epoch") +
    theme_minimal()
  
  ggsave("./singletask_loss_curve.png", p, width = 8, height = 6)
  cat("损失曲线已保存: ./singletask_loss_curve.png\n")
}

# ============================================================
# 15. 绘制ROC曲线
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

cat("\n========================================\n")
cat("单任务模型训练完成！\n")
cat("========================================\n")