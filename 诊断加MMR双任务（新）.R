# ============================================================
# 多任务深度学习模型：诊断 + MMR预测（渐进式学习优化版）
# 方案：前50轮只训练诊断，后50轮联合训练
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

# 查看 MMR 列的唯一值
cat("\nMMR.state 列的唯一值（训练集）:\n")
print(unique(train_data$MMR.state))

# 定义特征列
feature_cols <- c("gender", "age", "BMI", "Smoking_history", "drinking_history", 
                  "Family_history_of_cancer", "LYMPH_percentage", "MONO_percentage", 
                  "HGB", "MCV", "MCHC", "PLT", "DBIL", "IBIL", "ALB", "GLB", "ALT")

# ============================================================
# 2. 创建多任务标签函数（诊断 + MMR）
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
  
  return(data.frame(
    diag = diag_label,
    mmr = mmr_label
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

cat("\n测试集:\n")
cat("  诊断 - 癌症:", sum(test_labels$diag == 1, na.rm = TRUE), 
    ", 对照:", sum(test_labels$diag == 0, na.rm = TRUE), "\n")
cat("  MMR - PMMR:", sum(test_labels$mmr == 0, na.rm = TRUE), 
    ", dMMR:", sum(test_labels$mmr == 1, na.rm = TRUE), "\n")
cat("  MMR - 未知(不参与评估):", sum(is.na(test_labels$mmr) & test_labels$diag == 1), "\n")

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
# 5. 定义 PyTorch 模型（Python部分）- 2任务渐进式学习版
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
        
        # 诊断头（主任务）
        self.diag_head = nn.Sequential(
            nn.Linear(prev_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )
        # MMR头
        self.mmr_head = nn.Sequential(
            nn.Linear(prev_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )
    
    def forward(self, x):
        shared = self.shared_backbone(x)
        out_diag = self.diag_head(shared)
        out_mmr = self.mmr_head(shared)
        return out_diag, out_mmr
    
    def get_task_heads(self):
        \"\"\"返回任务头的第一层权重\"\"\"
        return [self.diag_head[0], self.mmr_head[0]]


# ============================================================
# 加权不确定性损失（可调主任务权重）
# ============================================================
class WeightedUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=2, primary_task_idx=0, primary_weight=2.0):
        super(WeightedUncertaintyLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.primary_idx = primary_task_idx
        self.primary_weight = primary_weight
    
    def forward(self, losses):
        precision = torch.exp(-self.log_vars)
        weighted_losses = precision * losses + self.log_vars
        weighted_losses[self.primary_idx] = weighted_losses[self.primary_idx] * self.primary_weight
        return weighted_losses.sum()


# ============================================================
# 任务头相似性正则化（可开关）
# ============================================================
def task_similarity_loss(task_heads, penalty_weight=0.005):
    if len(task_heads) < 2 or penalty_weight == 0:
        return torch.tensor(0.0)
    
    loss = 0.0
    head_primary = task_heads[0].weight  # 诊断头作为参考
    
    for i in range(1, len(task_heads)):
        head_i_w = task_heads[i].weight
        loss += torch.norm(head_primary - head_i_w, p='fro')
    
    loss = loss / len(task_heads) * penalty_weight
    return loss


def create_dataloader(X, y_diag, y_mmr, batch_size=128, shuffle=True):
    X_t = torch.FloatTensor(X)
    y_diag_t = torch.FloatTensor(y_diag).view(-1, 1)
    y_mmr_t = torch.FloatTensor(y_mmr).view(-1, 1)
    dataset = TensorDataset(X_t, y_diag_t, y_mmr_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
")

# ============================================================
# 6. 准备数据（转换为Python格式）
# ============================================================
y_diag_train <- train_labels$diag
y_diag_test <- test_labels$diag
y_mmr_train <- train_labels$mmr
y_mmr_test <- test_labels$mmr

py$X_train <- X_train
py$X_test <- X_test
py$y_diag_train <- y_diag_train
py$y_diag_test <- y_diag_test
py$y_mmr_train <- y_mmr_train
py$y_mmr_test <- y_mmr_test

py$X_train_final <- X_train
py$y_diag_train_final <- y_diag_train
py$y_mmr_train_final <- y_mmr_train

# ============================================================
# 7. 渐进式学习训练函数（2任务，前50轮只训练诊断）
# ============================================================
py_run_string("
import torch.optim as optim

def train_multitask_model_progressive(X_train, y_diag_train, y_mmr_train,
                                        input_dim=17, epochs=100, lr=0.001, batch_size=128,
                                        primary_weight=2.0, similarity_penalty=0.005,
                                        progressive_epochs=50):
    '''
    渐进式学习训练函数（2任务版本）
    
    Args:
        progressive_epochs: 前N轮只训练诊断任务（默认50）
        primary_weight: 主任务权重倍数
        similarity_penalty: 相似性正则化强度
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'训练配置: epochs={epochs}, lr={lr}, batch_size={batch_size}')
    print(f'网络结构: [128, 64, 32], dropout=0.3, weight_decay=1e-5')
    print(f'优化配置: primary_weight={primary_weight}, similarity_penalty={similarity_penalty}')
    print(f'渐进式学习: 前{progressive_epochs}轮只训练诊断任务，后{epochs - progressive_epochs}轮联合训练（MMR）')
    
    train_loader = create_dataloader(X_train, y_diag_train, y_mmr_train, batch_size, True)
    
    model = MultiTaskModel(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout=0.3).to(device)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    weighted_unc_loss = WeightedUncertaintyLoss(
        num_tasks=2, 
        primary_task_idx=0, 
        primary_weight=primary_weight
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {
        'train_loss': [],
        'diag_loss': [],
        'mmr_loss': [],
        'similarity_loss': [],
        'task_weights': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        diag_loss_sum = 0.0
        mmr_loss_sum = 0.0
        similarity_loss_sum = 0.0
        
        # 判断当前阶段是否使用MMR任务
        use_mmr = (epoch >= progressive_epochs)
        
        for X_batch, y_d, y_m in train_loader:
            X_batch, y_d, y_m = X_batch.to(device), y_d.to(device), y_m.to(device)
            out_d, out_m = model(X_batch)
            
            mask_d = ~torch.isnan(y_d)
            mask_m = ~torch.isnan(y_m)
            
            loss_d = bce_loss(out_d[mask_d], y_d[mask_d]).mean() if mask_d.any() else torch.tensor(0.0).to(device)
            
            if use_mmr and mask_m.any():
                loss_m = bce_loss(out_m[mask_m], y_m[mask_m]).mean()
                # 使用加权不确定性损失
                total_loss = weighted_unc_loss(torch.stack([loss_d, loss_m]))
            else:
                loss_m = torch.tensor(0.0).to(device)
                total_loss = loss_d
            
            # 相似性正则化（仅在联合训练时启用）
            sim_loss = torch.tensor(0.0).to(device)
            if use_mmr and similarity_penalty > 0:
                task_heads = model.get_task_heads()
                sim_loss = task_similarity_loss(task_heads, penalty_weight=similarity_penalty)
                total_loss = total_loss + sim_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            diag_loss_sum += loss_d.item()
            mmr_loss_sum += loss_m.item() if use_mmr else 0
            similarity_loss_sum += sim_loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        history['diag_loss'].append(diag_loss_sum / len(train_loader))
        history['mmr_loss'].append(mmr_loss_sum / len(train_loader) if use_mmr else 0)
        history['similarity_loss'].append(similarity_loss_sum / len(train_loader))
        
        with torch.no_grad():
            log_vars = weighted_unc_loss.log_vars.cpu().numpy()
            task_weights = np.exp(-log_vars)
            history['task_weights'].append(task_weights.copy())
        
        if (epoch + 1) % 20 == 0:
            if use_mmr:
                print(f'Epoch {epoch+1}/{epochs}: Total Loss={train_loss:.4f}, Diag Loss={diag_loss_sum/len(train_loader):.4f}, MMR Loss={mmr_loss_sum/len(train_loader):.4f}, Sim Loss={similarity_loss_sum/len(train_loader):.6f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}: Total Loss={train_loss:.4f}, Diag Loss={diag_loss_sum/len(train_loader):.4f} (诊断-only阶段)')
        
        # 阶段转换提示
        if epoch == progressive_epochs - 1:
            print(f'\\n>>> 阶段转换：第{progressive_epochs}轮开始，加入MMR任务 <<<\\n')
    
    with torch.no_grad():
        log_vars = weighted_unc_loss.log_vars.cpu().numpy()
        task_weights = np.exp(-log_vars)
        print(f'\\n最终学习到的任务权重: 诊断={task_weights[0]:.4f}, MMR={task_weights[1]:.4f}')
    
    return model, history
")

# ============================================================
# 8. 训练模型（前50轮只训练诊断）
# ============================================================
cat("\n")
cat("══════════════════════════════════════════════════════════════════\n")
cat("渐进式学习：前50轮只训练诊断，后50轮联合训练（MMR）\n")
cat("参数: primary_weight=2.0, similarity_penalty=0.005\n")
cat("══════════════════════════════════════════════════════════════════\n\n")

py_run_string("
model, history = train_multitask_model_progressive(
    X_train_final, y_diag_train_final, y_mmr_train_final,
    input_dim=17, epochs=100, lr=0.001, batch_size=128,
    primary_weight=2.0,        # 诊断任务权重2倍
    similarity_penalty=0.005,  # 相似性正则化
    progressive_epochs=50      # 前50轮只训练诊断
)
")

# ============================================================
# 9. 训练集和测试集预测
# ============================================================
py_run_string("
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 训练集预测
X_train_t = torch.FloatTensor(X_train).to(device)
with torch.no_grad():
    out_d_train, out_m_train = model(X_train_t)
    train_pred_diag = torch.sigmoid(out_d_train).cpu().numpy().flatten()
    train_pred_mmr = torch.sigmoid(out_m_train).cpu().numpy().flatten()

# 测试集预测
X_test_t = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    out_d_test, out_m_test = model(X_test_t)
    test_pred_diag = torch.sigmoid(out_d_test).cpu().numpy().flatten()
    test_pred_mmr = torch.sigmoid(out_m_test).cpu().numpy().flatten()
")

# 传递所有预测结果到R
train_pred_diag <- py$train_pred_diag
train_pred_mmr <- py$train_pred_mmr
test_pred_diag <- py$test_pred_diag
test_pred_mmr <- py$test_pred_mmr

# ============================================================
# 10. 完整评估函数
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
# 11. 计算训练集和测试集指标（诊断任务）
# ============================================================
cat("\n========================================\n")
cat("渐进式学习-多任务模型评估结果（诊断任务）\n")
cat("========================================\n")

train_metrics <- calculate_full_metrics(train_labels$diag, train_pred_diag, "Training")
test_metrics <- calculate_full_metrics(test_labels$diag, test_pred_diag, "Testing")

results_df <- rbind(train_metrics, test_metrics)
cat("\n=== 诊断任务结果汇总 ===\n")
print(results_df)

# 保存结果
write.csv(results_df, "./multitask_diag_metrics_diag_mmr_progressive.csv", row.names = FALSE)
cat("\n诊断任务结果已保存到: ./multitask_diag_metrics_diag_mmr_progressive.csv\n")

# ============================================================
# 12. MMR任务评估（测试集，仅癌症样本）
# ============================================================
cat("\n========================================\n")
cat("渐进式学习-多任务模型评估结果（MMR任务）\n")
cat("========================================\n")

cancer_idx_test <- which(test_data$group == "cancer")
if (length(cancer_idx_test) > 0) {
  mmr_true <- test_labels$mmr[cancer_idx_test]
  mmr_pred <- test_pred_mmr[cancer_idx_test]
  
  valid_mmr <- !is.na(mmr_true)
  if (sum(valid_mmr) > 0 && length(unique(mmr_true[valid_mmr])) > 1) {
    roc_mmr <- roc(mmr_true[valid_mmr], mmr_pred[valid_mmr])
    auc_mmr <- auc(roc_mmr)
    ci_mmr <- ci.auc(roc_mmr, conf.level = 0.95)
    
    cat(sprintf("\nMMR任务: AUC = %.3f (95%% CI: %.3f-%.3f)\n", auc_mmr, ci_mmr[1], ci_mmr[3]))
    cat(sprintf("有效样本数: %d\n", sum(valid_mmr)))
    
    mmr_result <- data.frame(
      Task = "MMR",
      AUC = round(auc_mmr, 3),
      AUC_lower = round(ci_mmr[1], 3),
      AUC_upper = round(ci_mmr[3], 3),
      N = sum(valid_mmr)
    )
    write.csv(mmr_result, "./multitask_mmr_metrics_diag_mmr_progressive.csv", row.names = FALSE)
  } else {
    cat("MMR任务: 样本不足，无法计算AUC\n")
  }
}

# ============================================================
# 13. 与单任务基线对比
# ============================================================
cat("\n========================================\n")
cat("与单任务基线对比\n")
cat("========================================\n")

# 单任务基线
baseline_acc <- 0.703
baseline_auc <- 0.766

cat(sprintf("\n单任务诊断模型:\n"))
cat(sprintf("  测试集 ACC: %.3f\n", baseline_acc))
cat(sprintf("  测试集 AUC: %.3f\n", baseline_auc))

cat(sprintf("\n渐进式学习（诊断+MMR）:\n"))
cat(sprintf("  测试集 ACC: %.3f\n", test_metrics$ACC))
cat(sprintf("  测试集 AUC: %.3f\n", test_metrics$AUC))

cat(sprintf("\n=== 提升幅度 ===\n"))
cat(sprintf("  ACC提升: %+.3f (%.1f%%)\n", 
            test_metrics$ACC - baseline_acc,
            (test_metrics$ACC - baseline_acc) / baseline_acc * 100))
cat(sprintf("  AUC提升: %+.3f (%.1f%%)\n",
            test_metrics$AUC - baseline_auc,
            (test_metrics$AUC - baseline_auc) / baseline_auc * 100))

# ============================================================
# 14. 绘制训练曲线（增强版）
# ============================================================
train_loss_vec <- tryCatch(unlist(py$history['train_loss']), error = function(e) NULL)
diag_loss_vec <- tryCatch(unlist(py$history['diag_loss']), error = function(e) NULL)
mmr_loss_vec <- tryCatch(unlist(py$history['mmr_loss']), error = function(e) NULL)
similarity_loss_vec <- tryCatch(unlist(py$history['similarity_loss']), error = function(e) NULL)
task_weights_history <- tryCatch(py$history['task_weights'], error = function(e) NULL)

if (!is.null(train_loss_vec) && length(train_loss_vec) > 0) {
  # 总损失曲线
  loss_df <- data.frame(Epoch = 1:length(train_loss_vec), Train_Loss = train_loss_vec)
  
  p1 <- ggplot(loss_df, aes(x = Epoch, y = Train_Loss)) +
    geom_line(color = "blue", linewidth = 1) +
    geom_vline(xintercept = 50, linetype = "dashed", color = "red", alpha = 0.7) +
    annotate("text", x = 52, y = max(train_loss_vec) * 0.9, 
             label = "阶段转换 (加入MMR任务)", color = "red", size = 3, hjust = 0) +
    labs(title = "训练过程 - 总损失曲线（渐进式学习）", 
         subtitle = "红色虚线：第50轮开始加入MMR任务",
         y = "Total Loss", x = "Epoch") +
    theme_minimal()
  ggsave("./multitask_loss_curve_diag_mmr_progressive.png", p1, width = 8, height = 6)
  
  # 分项损失曲线
  if (!is.null(diag_loss_vec) && !is.null(mmr_loss_vec)) {
    loss_detail_df <- data.frame(
      Epoch = 1:length(diag_loss_vec),
      Diag_Loss = diag_loss_vec,
      MMR_Loss = mmr_loss_vec
    )
    
    p2 <- ggplot(loss_detail_df, aes(x = Epoch)) +
      geom_line(aes(y = Diag_Loss, color = "Diagnosis"), linewidth = 1) +
      geom_line(aes(y = MMR_Loss, color = "MMR"), linewidth = 1) +
      geom_vline(xintercept = 50, linetype = "dashed", color = "red", alpha = 0.7) +
      labs(title = "训练过程 - 分项损失曲线", y = "Loss", x = "Epoch") +
      scale_color_manual(values = c("Diagnosis" = "red", "MMR" = "orange")) +
      theme_minimal()
    ggsave("./multitask_loss_detail_diag_mmr_progressive.png", p2, width = 8, height = 6)
  }
  
  # 相似性正则化损失曲线
  if (!is.null(similarity_loss_vec) && length(similarity_loss_vec) > 0 && sum(similarity_loss_vec) > 0) {
    sim_loss_df <- data.frame(Epoch = 1:length(similarity_loss_vec), Similarity_Loss = similarity_loss_vec)
    
    p3 <- ggplot(sim_loss_df, aes(x = Epoch, y = Similarity_Loss)) +
      geom_line(color = "darkgreen", linewidth = 1) +
      geom_vline(xintercept = 50, linetype = "dashed", color = "red", alpha = 0.7) +
      labs(title = "训练过程 - 任务头相似性损失", y = "Similarity Loss", x = "Epoch") +
      theme_minimal()
    ggsave("./multitask_similarity_loss_diag_mmr_progressive.png", p3, width = 8, height = 6)
  }
  
  # 任务权重变化曲线
  if (!is.null(task_weights_history) && length(task_weights_history) > 0) {
    weights_df <- data.frame(
      Epoch = 1:length(task_weights_history),
      Diag_Weight = sapply(task_weights_history, function(x) x[0]),
      MMR_Weight = sapply(task_weights_history, function(x) x[1])
    )
    
    p4 <- ggplot(weights_df, aes(x = Epoch)) +
      geom_line(aes(y = Diag_Weight, color = "Diagnosis"), linewidth = 1) +
      geom_line(aes(y = MMR_Weight, color = "MMR"), linewidth = 1) +
      geom_vline(xintercept = 50, linetype = "dashed", color = "red", alpha = 0.7) +
      labs(title = "任务权重变化（Uncertainty Weighting）", y = "Task Weight", x = "Epoch") +
      scale_color_manual(values = c("Diagnosis" = "red", "MMR" = "orange")) +
      theme_minimal()
    ggsave("./multitask_task_weights_diag_mmr_progressive.png", p4, width = 8, height = 6)
  }
  
  cat("\n损失曲线已保存:\n")
  cat("  - 总损失: ./multitask_loss_curve_diag_mmr_progressive.png\n")
  cat("  - 分项损失: ./multitask_loss_detail_diag_mmr_progressive.png\n")
  if (!is.null(similarity_loss_vec) && sum(similarity_loss_vec) > 0) {
    cat("  - 相似性损失: ./multitask_similarity_loss_diag_mmr_progressive.png\n")
  }
  if (!is.null(task_weights_history) && length(task_weights_history) > 0) {
    cat("  - 任务权重: ./multitask_task_weights_diag_mmr_progressive.png\n")
  }
}

# ============================================================
# 15. 绘制ROC曲线
# ============================================================
# 诊断ROC
valid_diag <- !is.na(test_labels$diag)
if (sum(valid_diag) > 0 && length(unique(test_labels$diag[valid_diag])) > 1) {
  png("./multitask_diag_roc_diag_mmr_progressive.png", width = 6, height = 6, units = "in", res = 300)
  roc_diag <- roc(test_labels$diag[valid_diag], test_pred_diag[valid_diag])
  plot(roc_diag, main = paste0("诊断任务 ROC (渐进式学习)\nAUC = ", round(roc_diag$auc, 3)), 
       col = "blue", lwd = 2)
  dev.off()
  cat("诊断ROC曲线已保存: ./multitask_diag_roc_diag_mmr_progressive.png\n")
}

# MMR ROC
if (length(cancer_idx_test) > 0) {
  valid_mmr <- !is.na(test_labels$mmr[cancer_idx_test])
  if (sum(valid_mmr) > 0 && length(unique(test_labels$mmr[cancer_idx_test][valid_mmr])) > 1) {
    png("./multitask_mmr_roc_diag_mmr_progressive.png", width = 6, height = 6, units = "in", res = 300)
    roc_mmr <- roc(test_labels$mmr[cancer_idx_test][valid_mmr], 
                   test_pred_mmr[cancer_idx_test][valid_mmr])
    plot(roc_mmr, main = paste0("MMR任务 ROC (渐进式学习)\nAUC = ", round(roc_mmr$auc, 3)), 
         col = "orange", lwd = 2)
    dev.off()
    cat("MMR ROC曲线已保存: ./multitask_mmr_roc_diag_mmr_progressive.png\n")
  }
}

cat("\n========================================\n")
cat("渐进式学习多任务模型（诊断 + MMR）训练完成！\n")
cat("========================================\n")