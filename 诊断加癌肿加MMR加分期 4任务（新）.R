# ============================================================
# 多任务深度学习模型：诊断 + MMR预测 + 癌肿分型 + 分期（带优化）
# 优化点1: 加权不确定性损失（主任务偏好）
# 优化点2: 任务头相似性正则化
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

# 查看各列的唯一值
cat("\ngroup2 列的唯一值（训练集）:\n")
print(unique(train_data$group2))

cat("\nMMR.state 列的唯一值（训练集）:\n")
print(unique(train_data$MMR.state))

cat("\nStage 列的唯一值（训练集）:\n")
print(unique(train_data$Stage))

# 定义特征列
feature_cols <- c("gender", "age", "BMI", "Smoking_history", "drinking_history", 
                  "Family_history_of_cancer", "LYMPH_percentage", "MONO_percentage", 
                  "HGB", "MCV", "MCHC", "PLT", "DBIL", "IBIL", "ALB", "GLB", "ALT")

# ============================================================
# 2. 创建多任务标签函数（诊断 + MMR + 癌肿分型 + 分期）
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
  
  # 任务4: 分期标签 (仅cancer样本: early=0, late=1)
  stage_label <- rep(NA, n)
  for (idx in cancer_idx) {
    stage_val <- data$Stage[idx]
    if (is.null(stage_val) || length(stage_val) == 0) next
    if (is.na(stage_val)) next
    stage_char <- as.character(stage_val)
    if (stage_char %in% c("", "NA", "NULL", "null")) next
    if (stage_char == "cancer_early") {
      stage_label[idx] <- 0
    } else if (stage_char == "cancer_late") {
      stage_label[idx] <- 1
    }
  }
  
  return(data.frame(
    diag = diag_label,
    mmr = mmr_label,
    cancer_type = cancer_type_label,
    stage = stage_label
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
cat("  分期 - 早期:", sum(train_labels$stage == 0, na.rm = TRUE), 
    ", 晚期:", sum(train_labels$stage == 1, na.rm = TRUE), "\n")
cat("  分期 - 未知(不参与训练):", sum(is.na(train_labels$stage) & train_labels$diag == 1), "\n")

cat("\n测试集:\n")
cat("  诊断 - 癌症:", sum(test_labels$diag == 1, na.rm = TRUE), 
    ", 对照:", sum(test_labels$diag == 0, na.rm = TRUE), "\n")
cat("  MMR - PMMR:", sum(test_labels$mmr == 0, na.rm = TRUE), 
    ", dMMR:", sum(test_labels$mmr == 1, na.rm = TRUE), "\n")
cat("  MMR - 未知(不参与评估):", sum(is.na(test_labels$mmr) & test_labels$diag == 1), "\n")
cat("  癌肿分型 - 结直肠癌:", sum(test_labels$cancer_type == 0, na.rm = TRUE), 
    ", 胃癌:", sum(test_labels$cancer_type == 1, na.rm = TRUE), "\n")
cat("  癌肿分型 - 未知(不参与评估):", sum(is.na(test_labels$cancer_type) & test_labels$diag == 1), "\n")
cat("  分期 - 早期:", sum(test_labels$stage == 0, na.rm = TRUE), 
    ", 晚期:", sum(test_labels$stage == 1, na.rm = TRUE), "\n")
cat("  分期 - 未知(不参与评估):", sum(is.na(test_labels$stage) & test_labels$diag == 1), "\n")

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
# 5. 定义 PyTorch 模型（Python部分）- 4任务带优化版本
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
        
        # 诊断头 (主任务)
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
        # 分期头
        self.stage_head = nn.Sequential(
            nn.Linear(prev_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )
    
    def forward(self, x):
        shared = self.shared_backbone(x)
        out_diag = self.diag_head(shared)
        out_mmr = self.mmr_head(shared)
        out_type = self.cancer_type_head(shared)
        out_stage = self.stage_head(shared)
        return out_diag, out_mmr, out_type, out_stage
    
    def get_task_heads(self):
        \"\"\"返回所有任务头的第一层权重（用于相似性正则化）\"\"\"
        return [
            self.diag_head[0],      # 诊断头第一层Linear
            self.mmr_head[0],       # MMR头第一层Linear
            self.cancer_type_head[0],  # 癌肿分型头第一层Linear
            self.stage_head[0]      # 分期头第一层Linear
        ]


# ============================================================
# 优化点1: 加权不确定性损失（主任务偏好）
# ============================================================
class WeightedUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=4, primary_task_idx=0, primary_weight=2.0):
        super(WeightedUncertaintyLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.primary_idx = primary_task_idx
        self.primary_weight = primary_weight
    
    def forward(self, losses):
        precision = torch.exp(-self.log_vars)
        weighted_losses = precision * losses + self.log_vars
        # 对主任务（诊断）施加额外权重
        weighted_losses[self.primary_idx] = weighted_losses[self.primary_idx] * self.primary_weight
        return weighted_losses.sum()


# ============================================================
# 优化点2: 任务头相似性正则化（扩展版：所有任务头相互相似）
# ============================================================
def task_similarity_loss(task_heads, penalty_weight=0.005, use_all_pairs=False):
    \"\"\"
    强制不同任务头的第一层权重相似，促进任务间更平滑的知识迁移
    
    Args:
        task_heads: list of task head layers (第一层Linear)
        penalty_weight: 正则化强度
        use_all_pairs: True=所有任务对之间计算相似性，False=只计算与主任务的相似性
    \"\"\"
    if len(task_heads) < 2:
        return torch.tensor(0.0)
    
    loss = 0.0
    num_pairs = 0
    
    if use_all_pairs:
        # 方案A: 所有任务对之间都计算相似性（计算量大，但任务间信息共享更充分）
        for i in range(len(task_heads)):
            for j in range(i + 1, len(task_heads)):
                w_i = task_heads[i].weight
                w_j = task_heads[j].weight
                loss += torch.norm(w_i - w_j, p='fro')
                num_pairs += 1
    else:
        # 方案B: 只计算与主任务（第一个任务头）的相似性（计算量小，更稳定）
        head_primary = task_heads[0].weight
        for i in range(1, len(task_heads)):
            w_i = task_heads[i].weight
            loss += torch.norm(head_primary - w_i, p='fro')
            num_pairs += 1
    
    loss = loss / max(num_pairs, 1) * penalty_weight
    return loss


def create_dataloader(X, y_diag, y_mmr, y_type, y_stage, batch_size=128, shuffle=True):
    X_t = torch.FloatTensor(X)
    y_diag_t = torch.FloatTensor(y_diag).view(-1, 1)
    y_mmr_t = torch.FloatTensor(y_mmr).view(-1, 1)
    y_type_t = torch.FloatTensor(y_type).view(-1, 1)
    y_stage_t = torch.FloatTensor(y_stage).view(-1, 1)
    dataset = TensorDataset(X_t, y_diag_t, y_mmr_t, y_type_t, y_stage_t)
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
y_stage_train <- train_labels$stage
y_stage_test <- test_labels$stage

py$X_train <- X_train
py$X_test <- X_test
py$y_diag_train <- y_diag_train
py$y_diag_test <- y_diag_test
py$y_mmr_train <- y_mmr_train
py$y_mmr_test <- y_mmr_test
py$y_type_train <- y_type_train
py$y_type_test <- y_type_test
py$y_stage_train <- y_stage_train
py$y_stage_test <- y_stage_test

# ============================================================
# 7. 使用全部训练数据
# ============================================================
cat("\n=== 数据划分 ===\n")
cat("训练集样本数:", nrow(X_train), "\n")
cat("测试集样本数:", nrow(X_test), "\n")

py$X_train_final <- X_train
py$y_diag_train_final <- y_diag_train
py$y_mmr_train_final <- y_mmr_train
py$y_type_train_final <- y_type_train
py$y_stage_train_final <- y_stage_train

# ============================================================
# 8. 训练函数（4个任务，带优化）
# ============================================================
py_run_string("
import torch.optim as optim

def train_multitask_model(X_train, y_diag_train, y_mmr_train, y_type_train, y_stage_train,
                           input_dim=17, epochs=100, lr=0.001, batch_size=128,
                           primary_weight=2.0, similarity_penalty=0.005, use_all_pairs=False):
    \"\"\"
    训练多任务模型（带主任务加权 + 任务头相似性正则化）
    
    Args:
        primary_weight: 主任务（诊断）的权重倍数
        similarity_penalty: 任务头相似性正则化强度
        use_all_pairs: 是否对所有任务对计算相似性（默认只与主任务计算）
    \"\"\"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'训练配置: epochs={epochs}, lr={lr}, batch_size={batch_size}')
    print(f'网络结构: [128, 64, 32], dropout=0.3, weight_decay=1e-5')
    print(f'优化配置: primary_weight={primary_weight}, similarity_penalty={similarity_penalty}, use_all_pairs={use_all_pairs}')
    
    train_loader = create_dataloader(X_train, y_diag_train, y_mmr_train, y_type_train, y_stage_train, batch_size, True)
    
    model = MultiTaskModel(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout=0.3).to(device)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    # 优化点1: 使用加权不确定性损失，诊断任务为主任务（idx=0）
    weighted_unc_loss = WeightedUncertaintyLoss(
        num_tasks=4, 
        primary_task_idx=0, 
        primary_weight=primary_weight
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {
        'train_loss': [], 
        'task_losses': {'diag': [], 'mmr': [], 'type': [], 'stage': []},
        'similarity_loss': [],
        'task_weights': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        diag_loss_sum = 0.0
        mmr_loss_sum = 0.0
        type_loss_sum = 0.0
        stage_loss_sum = 0.0
        similarity_loss_sum = 0.0
        
        for X_batch, y_d, y_m, y_t, y_s in train_loader:
            X_batch, y_d, y_m, y_t, y_s = X_batch.to(device), y_d.to(device), y_m.to(device), y_t.to(device), y_s.to(device)
            out_d, out_m, out_t, out_s = model(X_batch)
            
            mask_d = ~torch.isnan(y_d)
            mask_m = ~torch.isnan(y_m)
            mask_t = ~torch.isnan(y_t)
            mask_s = ~torch.isnan(y_s)
            
            loss_d = bce_loss(out_d[mask_d], y_d[mask_d]).mean() if mask_d.any() else torch.tensor(0.0).to(device)
            loss_m = bce_loss(out_m[mask_m], y_m[mask_m]).mean() if mask_m.any() else torch.tensor(0.0).to(device)
            loss_t = bce_loss(out_t[mask_t], y_t[mask_t]).mean() if mask_t.any() else torch.tensor(0.0).to(device)
            loss_s = bce_loss(out_s[mask_s], y_s[mask_s]).mean() if mask_s.any() else torch.tensor(0.0).to(device)
            
            # 优化点1: 加权不确定性损失
            total_loss = weighted_unc_loss(torch.stack([loss_d, loss_m, loss_t, loss_s]))
            
            # 优化点2: 任务头相似性正则化
            task_heads = model.get_task_heads()
            sim_loss = task_similarity_loss(task_heads, penalty_weight=similarity_penalty, use_all_pairs=use_all_pairs)
            total_loss = total_loss + sim_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            diag_loss_sum += loss_d.item()
            mmr_loss_sum += loss_m.item()
            type_loss_sum += loss_t.item()
            stage_loss_sum += loss_s.item()
            similarity_loss_sum += sim_loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        history['task_losses']['diag'].append(diag_loss_sum / len(train_loader))
        history['task_losses']['mmr'].append(mmr_loss_sum / len(train_loader))
        history['task_losses']['type'].append(type_loss_sum / len(train_loader))
        history['task_losses']['stage'].append(stage_loss_sum / len(train_loader))
        history['similarity_loss'].append(similarity_loss_sum / len(train_loader))
        
        # 记录任务权重变化
        with torch.no_grad():
            log_vars = weighted_unc_loss.log_vars.cpu().numpy()
            task_weights = np.exp(-log_vars)
            history['task_weights'].append(task_weights.copy())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Total={train_loss:.4f}, '
                  f'Diag={diag_loss_sum/len(train_loader):.4f}, '
                  f'MMR={mmr_loss_sum/len(train_loader):.4f}, '
                  f'Type={type_loss_sum/len(train_loader):.4f}, '
                  f'Stage={stage_loss_sum/len(train_loader):.4f}, '
                  f'Sim={similarity_loss_sum/len(train_loader):.6f}')
    
    # 打印最终学习到的任务权重
    with torch.no_grad():
        log_vars = weighted_unc_loss.log_vars.cpu().numpy()
        task_weights = np.exp(-log_vars)
        print(f'\\n最终学习到的任务权重:')
        print(f'  诊断={task_weights[0]:.4f}, MMR={task_weights[1]:.4f}, '
              f'癌肿分型={task_weights[2]:.4f}, 分期={task_weights[3]:.4f}')
    
    return model, history
")

# ============================================================
# 9. 训练模型（带优化参数）
# ============================================================
cat("\n========================================\n")
cat("开始训练多任务模型（诊断 + MMR + 癌肿分型 + 分期）- 带优化\n")
cat("优化点1: 诊断任务权重 x2.0\n")
cat("优化点2: 任务头相似性正则化 (λ=0.005)\n")
cat("========================================\n\n")

py_run_string("
model, history = train_multitask_model(
    X_train_final, y_diag_train_final, y_mmr_train_final, y_type_train_final, y_stage_train_final,
    input_dim=17, epochs=100, lr=0.001, batch_size=128,
    primary_weight=2.0,      # 诊断任务权重是其他任务的2倍
    similarity_penalty=0.005, # 任务头相似性正则化强度
    use_all_pairs=False       # False=只与主任务相似，True=所有任务对相似
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
    out_d_train, out_m_train, out_t_train, out_s_train = model(X_train_t)
    train_pred_diag = torch.sigmoid(out_d_train).cpu().numpy().flatten()
    train_pred_mmr = torch.sigmoid(out_m_train).cpu().numpy().flatten()
    train_pred_type = torch.sigmoid(out_t_train).cpu().numpy().flatten()
    train_pred_stage = torch.sigmoid(out_s_train).cpu().numpy().flatten()

# 测试集预测
X_test_t = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    out_d_test, out_m_test, out_t_test, out_s_test = model(X_test_t)
    test_pred_diag = torch.sigmoid(out_d_test).cpu().numpy().flatten()
    test_pred_mmr = torch.sigmoid(out_m_test).cpu().numpy().flatten()
    test_pred_type = torch.sigmoid(out_t_test).cpu().numpy().flatten()
    test_pred_stage = torch.sigmoid(out_s_test).cpu().numpy().flatten()
")

# 传递所有预测结果到R
train_pred_diag <- py$train_pred_diag
train_pred_mmr <- py$train_pred_mmr
train_pred_type <- py$train_pred_type
train_pred_stage <- py$train_pred_stage
test_pred_diag <- py$test_pred_diag
test_pred_mmr <- py$test_pred_mmr
test_pred_type <- py$test_pred_type
test_pred_stage <- py$test_pred_stage

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
cat("优化后多任务模型评估结果（诊断任务）\n")
cat("========================================\n")

train_metrics <- calculate_full_metrics(train_labels$diag, train_pred_diag, "Training")
test_metrics <- calculate_full_metrics(test_labels$diag, test_pred_diag, "Testing")

results_df <- rbind(train_metrics, test_metrics)
cat("\n=== 诊断任务结果汇总 ===\n")
print(results_df)
write.csv(results_df, "./multitask_diag_metrics_4tasks_optimized.csv", row.names = FALSE)
cat("\n诊断任务结果已保存到: ./multitask_diag_metrics_4tasks_optimized.csv\n")

# ============================================================
# 13. MMR任务评估
# ============================================================
cat("\n========================================\n")
cat("优化后多任务模型评估结果（MMR任务 - 测试集）\n")
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
    cat(sprintf("MMR任务: AUC = %.3f (95%% CI: %.3f-%.3f), N=%d\n", 
                auc_mmr, ci_mmr[1], ci_mmr[3], sum(valid_mmr)))
    
    mmr_result <- data.frame(
      Task = "MMR",
      AUC = round(auc_mmr, 3),
      AUC_lower = round(ci_mmr[1], 3),
      AUC_upper = round(ci_mmr[3], 3),
      N = sum(valid_mmr)
    )
    write.csv(mmr_result, "./multitask_mmr_metrics_4tasks_optimized.csv", row.names = FALSE)
  } else {
    cat("MMR任务: 样本不足，无法计算AUC\n")
  }
}

# ============================================================
# 14. 癌肿分型任务评估
# ============================================================
cat("\n========================================\n")
cat("优化后多任务模型评估结果（癌肿分型任务 - 测试集）\n")
cat("========================================\n")

if (length(cancer_idx_test) > 0) {
  type_true <- test_labels$cancer_type[cancer_idx_test]
  type_pred <- test_pred_type[cancer_idx_test]
  
  valid_type <- !is.na(type_true)
  if (sum(valid_type) > 0 && length(unique(type_true[valid_type])) > 1) {
    roc_type <- roc(type_true[valid_type], type_pred[valid_type])
    auc_type <- auc(roc_type)
    ci_type <- ci.auc(roc_type, conf.level = 0.95)
    cat(sprintf("癌肿分型任务: AUC = %.3f (95%% CI: %.3f-%.3f), N=%d\n", 
                auc_type, ci_type[1], ci_type[3], sum(valid_type)))
    
    type_result <- data.frame(
      Task = "Cancer_Type",
      AUC = round(auc_type, 3),
      AUC_lower = round(ci_type[1], 3),
      AUC_upper = round(ci_type[3], 3),
      N = sum(valid_type)
    )
    write.csv(type_result, "./multitask_type_metrics_4tasks_optimized.csv", row.names = FALSE)
  } else {
    cat("癌肿分型任务: 样本不足，无法计算AUC\n")
  }
}

# ============================================================
# 15. 分期任务评估
# ============================================================
cat("\n========================================\n")
cat("优化后多任务模型评估结果（分期任务 - 测试集）\n")
cat("========================================\n")

if (length(cancer_idx_test) > 0) {
  stage_true <- test_labels$stage[cancer_idx_test]
  stage_pred <- test_pred_stage[cancer_idx_test]
  
  valid_stage <- !is.na(stage_true)
  if (sum(valid_stage) > 0 && length(unique(stage_true[valid_stage])) > 1) {
    roc_stage <- roc(stage_true[valid_stage], stage_pred[valid_stage])
    auc_stage <- auc(roc_stage)
    ci_stage <- ci.auc(roc_stage, conf.level = 0.95)
    cat(sprintf("分期任务: AUC = %.3f (95%% CI: %.3f-%.3f), N=%d\n", 
                auc_stage, ci_stage[1], ci_stage[3], sum(valid_stage)))
    
    stage_result <- data.frame(
      Task = "Stage",
      AUC = round(auc_stage, 3),
      AUC_lower = round(ci_stage[1], 3),
      AUC_upper = round(ci_stage[3], 3),
      N = sum(valid_stage)
    )
    write.csv(stage_result, "./multitask_stage_metrics_4tasks_optimized.csv", row.names = FALSE)
  } else {
    cat("分期任务: 样本不足，无法计算AUC\n")
  }
}

# ============================================================
# 16. 绘制训练曲线（增强版）
# ============================================================
train_loss_vec <- tryCatch(unlist(py$history['train_loss']), error = function(e) NULL)
similarity_loss_vec <- tryCatch(unlist(py$history['similarity_loss']), error = function(e) NULL)
task_losses <- tryCatch(py$history['task_losses'], error = function(e) NULL)
task_weights_history <- tryCatch(py$history['task_weights'], error = function(e) NULL)

if (!is.null(train_loss_vec) && length(train_loss_vec) > 0) {
  # 总损失曲线
  loss_df <- data.frame(Epoch = 1:length(train_loss_vec), Train_Loss = train_loss_vec)
  p1 <- ggplot(loss_df, aes(x = Epoch, y = Train_Loss)) +
    geom_line(color = "blue", linewidth = 1) +
    labs(title = "训练过程 - 总损失曲线（4任务优化版）", y = "Total Loss", x = "Epoch") +
    theme_minimal()
  ggsave("./multitask_loss_curve_4tasks_optimized.png", p1, width = 8, height = 6)
  
  # 分项损失曲线
  if (!is.null(task_losses)) {
    loss_detail_df <- data.frame(
      Epoch = 1:length(task_losses$diag),
      Diag_Loss = task_losses$diag,
      MMR_Loss = task_losses$mmr,
      Type_Loss = task_losses$type,
      Stage_Loss = task_losses$stage
    )
    
    p2 <- ggplot(loss_detail_df, aes(x = Epoch)) +
      geom_line(aes(y = Diag_Loss, color = "Diagnosis"), linewidth = 1) +
      geom_line(aes(y = MMR_Loss, color = "MMR"), linewidth = 1) +
      geom_line(aes(y = Type_Loss, color = "Cancer_Type"), linewidth = 1) +
      geom_line(aes(y = Stage_Loss, color = "Stage"), linewidth = 1) +
      labs(title = "训练过程 - 分项损失曲线", y = "Loss", x = "Epoch") +
      scale_color_manual(values = c("Diagnosis" = "red", "MMR" = "orange", 
                                    "Cancer_Type" = "green", "Stage" = "purple")) +
      theme_minimal()
    ggsave("./multitask_loss_detail_4tasks_optimized.png", p2, width = 8, height = 6)
  }
  
  # 相似性正则化损失曲线
  if (!is.null(similarity_loss_vec) && length(similarity_loss_vec) > 0) {
    sim_loss_df <- data.frame(Epoch = 1:length(similarity_loss_vec), Similarity_Loss = similarity_loss_vec)
    p3 <- ggplot(sim_loss_df, aes(x = Epoch, y = Similarity_Loss)) +
      geom_line(color = "darkgreen", linewidth = 1) +
      labs(title = "训练过程 - 任务头相似性损失", y = "Similarity Loss", x = "Epoch") +
      theme_minimal()
    ggsave("./multitask_similarity_loss_4tasks_optimized.png", p3, width = 8, height = 6)
  }
  
  # 任务权重变化曲线（不确定性加权）
  if (!is.null(task_weights_history) && length(task_weights_history) > 0) {
    weights_df <- data.frame(
      Epoch = 1:length(task_weights_history),
      Diag_Weight = sapply(task_weights_history, function(x) x[1]),
      MMR_Weight = sapply(task_weights_history, function(x) x[2]),
      Type_Weight = sapply(task_weights_history, function(x) x[3]),
      Stage_Weight = sapply(task_weights_history, function(x) x[4])
    )
    
    p4 <- ggplot(weights_df, aes(x = Epoch)) +
      geom_line(aes(y = Diag_Weight, color = "Diagnosis"), linewidth = 1) +
      geom_line(aes(y = MMR_Weight, color = "MMR"), linewidth = 1) +
      geom_line(aes(y = Type_Weight, color = "Cancer_Type"), linewidth = 1) +
      geom_line(aes(y = Stage_Weight, color = "Stage"), linewidth = 1) +
      labs(title = "任务权重变化（Uncertainty Weighting）", y = "Task Weight", x = "Epoch") +
      scale_color_manual(values = c("Diagnosis" = "red", "MMR" = "orange",
                                    "Cancer_Type" = "green", "Stage" = "purple")) +
      theme_minimal()
    ggsave("./multitask_task_weights_4tasks_optimized.png", p4, width = 8, height = 6)
  }
  
  cat("\n损失曲线已保存:\n")
  cat("  - 总损失: ./multitask_loss_curve_4tasks_optimized.png\n")
  cat("  - 分项损失: ./multitask_loss_detail_4tasks_optimized.png\n")
  cat("  - 相似性损失: ./multitask_similarity_loss_4tasks_optimized.png\n")
  cat("  - 任务权重: ./multitask_task_weights_4tasks_optimized.png\n")
}

# ============================================================
# 17. 绘制ROC曲线
# ============================================================
# 诊断ROC
valid_diag <- !is.na(test_labels$diag)
if (sum(valid_diag) > 0 && length(unique(test_labels$diag[valid_diag])) > 1) {
  png("./multitask_diag_roc_4tasks_optimized.png", width = 6, height = 6, units = "in", res = 300)
  roc_diag <- roc(test_labels$diag[valid_diag], test_pred_diag[valid_diag])
  plot(roc_diag, main = paste0("诊断任务 ROC (4任务优化版)\nAUC = ", round(roc_diag$auc, 3)), 
       col = "blue", lwd = 2)
  dev.off()
  cat("诊断ROC曲线已保存: ./multitask_diag_roc_4tasks_optimized.png\n")
}

# MMR ROC
if (length(cancer_idx_test) > 0) {
  valid_mmr <- !is.na(test_labels$mmr[cancer_idx_test])
  if (sum(valid_mmr) > 0 && length(unique(test_labels$mmr[cancer_idx_test][valid_mmr])) > 1) {
    png("./multitask_mmr_roc_4tasks_optimized.png", width = 6, height = 6, units = "in", res = 300)
    roc_mmr <- roc(test_labels$mmr[cancer_idx_test][valid_mmr], 
                   test_pred_mmr[cancer_idx_test][valid_mmr])
    plot(roc_mmr, main = paste0("MMR任务 ROC (4任务优化版)\nAUC = ", round(roc_mmr$auc, 3)), 
         col = "orange", lwd = 2)
    dev.off()
    cat("MMR ROC曲线已保存: ./multitask_mmr_roc_4tasks_optimized.png\n")
  }
}

cat("\n========================================\n")
cat("优化后多任务模型（4任务）训练完成！\n")
cat("========================================\n")