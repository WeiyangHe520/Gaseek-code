# ============================================================
# 多任务深度学习模型：诊断 + 癌肿分型
# （使用全部训练数据，无验证集，100轮 - 最优配置）
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

# 查看 group2 列的唯一值（癌肿类型）
cat("\ngroup2 列的唯一值（训练集）:\n")
print(unique(train_data$group2))

# 定义特征列
feature_cols <- c("gender", "age", "BMI", "Smoking_history", "drinking_history", 
                  "Family_history_of_cancer", "LYMPH_percentage", "MONO_percentage", 
                  "HGB", "MCV", "MCHC", "PLT", "DBIL", "IBIL", "ALB", "GLB", "ALT")

# ============================================================
# 2. 创建多任务标签函数（诊断 + 癌肿分型）
# ============================================================
create_multitask_labels <- function(data) {
  n <- nrow(data)
  
  # 任务1: 诊断标签 (所有样本)
  diag_label <- ifelse(data$group == "cancer", 1, 0)
  
  # 任务2: 癌肿分型标签 (仅cancer样本: colorectal cancer=0, gastric cancer=1)
  cancer_type_label <- rep(NA, n)
  cancer_idx <- which(data$group == "cancer")
  
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
cat("  癌肿分型 - 结直肠癌:", sum(train_labels$cancer_type == 0, na.rm = TRUE), 
    ", 胃癌:", sum(train_labels$cancer_type == 1, na.rm = TRUE), "\n")
cat("  癌肿分型 - 未知(不参与训练):", sum(is.na(train_labels$cancer_type) & train_labels$diag == 1), "\n")

cat("\n测试集:\n")
cat("  诊断 - 癌症:", sum(test_labels$diag == 1, na.rm = TRUE), 
    ", 对照:", sum(test_labels$diag == 0, na.rm = TRUE), "\n")
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
# 5. 定义 PyTorch 模型（Python部分）- 2任务版本
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
        # 癌肿分型头
        self.cancer_type_head = nn.Sequential(
            nn.Linear(prev_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )
    
    def forward(self, x):
        shared = self.shared_backbone(x)
        out_diag = self.diag_head(shared)
        out_type = self.cancer_type_head(shared)
        return out_diag, out_type

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(UncertaintyWeightedLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        precision = torch.exp(-self.log_vars)
        return torch.sum(precision * losses + self.log_vars)

def create_dataloader(X, y_diag, y_type, batch_size=128, shuffle=True):
    X_t = torch.FloatTensor(X)
    y_diag_t = torch.FloatTensor(y_diag).view(-1, 1)
    y_type_t = torch.FloatTensor(y_type).view(-1, 1)
    dataset = TensorDataset(X_t, y_diag_t, y_type_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
")

# ============================================================
# 6. 准备数据（转换为Python格式）
# ============================================================
y_diag_train <- train_labels$diag
y_diag_test <- test_labels$diag
y_type_train <- train_labels$cancer_type
y_type_test <- test_labels$cancer_type

py$X_train <- X_train
py$X_test <- X_test
py$y_diag_train <- y_diag_train
py$y_diag_test <- y_diag_test
py$y_type_train <- y_type_train
py$y_type_test <- y_type_test

# ============================================================
# 7. 使用全部训练数据
# ============================================================
cat("\n=== 数据划分 ===\n")
cat("训练集样本数:", nrow(X_train), "\n")
cat("测试集样本数:", nrow(X_test), "\n")

py$X_train_final <- X_train
py$y_diag_train_final <- y_diag_train
py$y_type_train_final <- y_type_train

# ============================================================
# 8. 训练函数（2个任务）
# ============================================================
py_run_string("
import torch.optim as optim

def train_multitask_model(X_train, y_diag_train, y_type_train,
                           input_dim=17, epochs=100, lr=0.001, batch_size=128):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'训练配置: epochs={epochs}, lr={lr}, batch_size={batch_size}')
    print(f'网络结构: [128, 64, 32], dropout=0.3, weight_decay=1e-5')
    
    train_loader = create_dataloader(X_train, y_diag_train, y_type_train, batch_size, True)
    
    model = MultiTaskModel(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout=0.3).to(device)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    unc_loss = UncertaintyWeightedLoss(num_tasks=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {'train_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_d, y_t in train_loader:
            X_batch, y_d, y_t = X_batch.to(device), y_d.to(device), y_t.to(device)
            out_d, out_t = model(X_batch)
            
            mask_d = ~torch.isnan(y_d)
            mask_t = ~torch.isnan(y_t)
            
            loss_d = bce_loss(out_d[mask_d], y_d[mask_d]).mean() if mask_d.any() else torch.tensor(0.0).to(device)
            loss_t = bce_loss(out_t[mask_t], y_t[mask_t]).mean() if mask_t.any() else torch.tensor(0.0).to(device)
            
            total_loss = unc_loss(torch.stack([loss_d, loss_t]))
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
")

# ============================================================
# 9. 训练模型
# ============================================================
cat("\n========================================\n")
cat("开始训练多任务模型（诊断 + 癌肿分型）...\n")
cat("========================================\n\n")

py_run_string("
model, history = train_multitask_model(
    X_train_final, y_diag_train_final, y_type_train_final,
    input_dim=17, epochs=100, lr=0.001, batch_size=128
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
    out_d_train, out_t_train = model(X_train_t)
    train_pred_diag = torch.sigmoid(out_d_train).cpu().numpy().flatten()
    train_pred_type = torch.sigmoid(out_t_train).cpu().numpy().flatten()

# 测试集预测
X_test_t = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    out_d_test, out_t_test = model(X_test_t)
    test_pred_diag = torch.sigmoid(out_d_test).cpu().numpy().flatten()
    test_pred_type = torch.sigmoid(out_t_test).cpu().numpy().flatten()
")

# 传递所有预测结果到R
train_pred_diag <- py$train_pred_diag
train_pred_type <- py$train_pred_type
test_pred_diag <- py$test_pred_diag
test_pred_type <- py$test_pred_type

# ============================================================
# 11. 完整评估函数（与glmnet格式一致）
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
cat("多任务模型评估结果（诊断任务）\n")
cat("========================================\n")

train_metrics <- calculate_full_metrics(train_labels$diag, train_pred_diag, "Training")
test_metrics <- calculate_full_metrics(test_labels$diag, test_pred_diag, "Testing")

results_df <- rbind(train_metrics, test_metrics)
cat("\n=== 诊断任务结果汇总 ===\n")
print(results_df)

# 保存结果
write.csv(results_df, "./multitask_diag_metrics_2tasks.csv", row.names = FALSE)
cat("\n诊断任务结果已保存到: ./multitask_diag_metrics_2tasks.csv\n")

# ============================================================
# 13. 癌肿分型任务评估（测试集，仅癌症样本）
# ============================================================
cat("\n========================================\n")
cat("多任务模型评估结果（癌肿分型任务 - 测试集）\n")
cat("========================================\n")

cancer_idx_test <- which(test_data$group == "cancer")
if (length(cancer_idx_test) > 0) {
  type_true <- test_labels$cancer_type[cancer_idx_test]
  type_pred <- test_pred_type[cancer_idx_test]
  
  valid_type <- !is.na(type_true)
  if (sum(valid_type) > 0 && length(unique(type_true[valid_type])) > 1) {
    roc_type <- roc(type_true[valid_type], type_pred[valid_type])
    auc_type <- auc(roc_type)
    ci_type <- ci.auc(roc_type, conf.level = 0.95)
    
    cat(sprintf("癌肿分型任务: AUC = %.3f (95%% CI: %.3f-%.3f)\n", auc_type, ci_type[1], ci_type[3]))
    cat(sprintf("有效样本数: %d\n", sum(valid_type)))
    
    # 保存癌肿分型结果
    type_result <- data.frame(
      Task = "Cancer_Type",
      AUC = round(auc_type, 3),
      AUC_lower = round(ci_type[1], 3),
      AUC_upper = round(ci_type[3], 3),
      N = sum(valid_type)
    )
    write.csv(type_result, "./multitask_type_metrics_2tasks.csv", row.names = FALSE)
  } else {
    cat("癌肿分型任务: 样本不足，无法计算AUC\n")
  }
}

# ============================================================
# 14. 绘制训练曲线
# ============================================================
train_loss_vec <- tryCatch(unlist(py$history['train_loss']), error = function(e) NULL)

if (!is.null(train_loss_vec) && length(train_loss_vec) > 0) {
  loss_df <- data.frame(Epoch = 1:length(train_loss_vec), Train_Loss = train_loss_vec)
  
  p <- ggplot(loss_df, aes(x = Epoch, y = Train_Loss)) +
    geom_line(color = "blue", linewidth = 1) +
    labs(title = "训练过程 - 损失曲线（2任务）", y = "Loss", x = "Epoch") +
    theme_minimal()
  
  ggsave("./multitask_loss_curve_2tasks.png", p, width = 8, height = 6)
  cat("损失曲线已保存: ./multitask_loss_curve_2tasks.png\n")
}

# ============================================================
# 15. 绘制ROC曲线
# ============================================================
# 诊断ROC
valid_diag <- !is.na(test_labels$diag)
if (sum(valid_diag) > 0 && length(unique(test_labels$diag[valid_diag])) > 1) {
  png("./multitask_diag_roc_2tasks.png", width = 6, height = 6, units = "in", res = 300)
  roc_diag <- roc(test_labels$diag[valid_diag], test_pred_diag[valid_diag])
  plot(roc_diag, main = paste0("诊断任务 ROC (2任务)\nAUC = ", round(roc_diag$auc, 3)), 
       col = "blue", lwd = 2)
  dev.off()
  cat("诊断ROC曲线已保存: ./multitask_diag_roc_2tasks.png\n")
}

# 癌肿分型ROC
if (length(cancer_idx_test) > 0) {
  valid_type <- !is.na(test_labels$cancer_type[cancer_idx_test])
  if (sum(valid_type) > 0 && length(unique(test_labels$cancer_type[cancer_idx_test][valid_type])) > 1) {
    png("./multitask_type_roc_2tasks.png", width = 6, height = 6, units = "in", res = 300)
    roc_type <- roc(test_labels$cancer_type[cancer_idx_test][valid_type], 
                    test_pred_type[cancer_idx_test][valid_type])
    plot(roc_type, main = paste0("癌肿分型任务 ROC (2任务)\nAUC = ", round(roc_type$auc, 3)), 
         col = "green", lwd = 2)
    dev.off()
    cat("癌肿分型ROC曲线已保存: ./multitask_type_roc_2tasks.png\n")
  }
}

cat("\n========================================\n")
cat("多任务模型（诊断 + 癌肿分型）训练完成！\n")
cat("========================================\n")