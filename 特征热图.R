rm(list = ls())
#################################################
## glmnet模型解释代码（类别权重版本）         ##
## 功能：类别权重模型 + 特征重要性 + 可视化   ##
## 参数：alpha = 0.5281, lambda = 0.004597   ##
## 版本：v2.0 (2024-11-06)                    ##
#################################################

library(caret)
library(glmnet)
library(pROC)
library(ggsignif)
library(corrplot)
library(rms)
library(dplyr)
library(ggplot2)
library(stringr)
library(vip)
library(DALEX)
library(ggrepel)
library(shapviz)
library(fastshap)
library(patchwork)
library(reshape2)

set.seed(278)  # 可重复性

# 输出目录
FIG_DIR <- "figures_glmnet_weighted/"    # 图片输出目录
DATA_DIR <- "data_glmnet_weighted/"      # 数据输出目录

# 创建目录
if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR, recursive = TRUE)
if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR, recursive = TRUE)

### 1. 数据加载和预处理 ###
cat("Loading data...\n")

# 加载数据
load(file = ".left_data.rdata")

# 定义特征列
feature_cols <- c("gender", "age", "BMI", "Smoking_history", "drinking_history", 
                  "Family_history_of_cancer", "LYMPH_percentage", "MONO_percentage", 
                  "HGB", "MCV", "MCHC", "PLT", "DBIL", "IBIL", "ALB", "GLB", "ALT")

# 数据预处理函数（包含标准化）
preprocess_data <- function(data) {
  features <- data[, feature_cols]
  group <- data$group
  
  # 标准化特征
  features_scaled <- as.data.frame(scale(features))
  
  # 处理缺失值
  features_scaled[is.na(features_scaled)] <- 0
  
  return(data.frame(group = group, features_scaled))
}

# 预处理数据
train_processed <- preprocess_data(train_data)
test_processed <- preprocess_data(test_data)

# 准备矩阵和标签
x_train <- as.matrix(train_processed[, -1])
x_test <- as.matrix(test_processed[, -1])
y_train <- train_processed$group
y_test <- test_processed$group

# 数值标签（用于模型训练）
y_train_binary <- ifelse(y_train == "cancer", 1, 0)

cat("Data Information:\n")
cat("Training samples:", nrow(x_train), "\n")
cat("Test samples:", nrow(x_test), "\n")
cat("Number of features:", ncol(x_train), "\n")
cat("Cancer prevalence - Training:", mean(y_train == "cancer"), 
    "Test:", mean(y_test == "cancer"), "\n")

### 2. 计算类别权重 ###
cat("\n=== 类别权重计算 ===\n")
cancer_count <- sum(y_train_binary == 1)
control_count <- sum(y_train_binary == 0)

# 逆频率权重（给少数类更高的权重）
weight_cancer <- control_count / cancer_count
weight_control <- 1

cat(sprintf("癌症样本数: %d, 对照样本数: %d\n", cancer_count, control_count))
cat(sprintf("类别权重: 癌症=%.3f, 对照=%.3f\n", weight_cancer, weight_control))

# 创建权重向量
sample_weights <- ifelse(y_train_binary == 1, weight_cancer, weight_control)

### 3. 训练类别权重glmnet模型 ###
cat("\nTraining weighted glmnet model with fixed parameters...\n")
cat("alpha = 0.5281, lambda = 0.004597\n")

# 训练带权重的模型
weighted_model <- glmnet(
  x = x_train,
  y = y_train_binary,
  family = "binomial",
  alpha = 0.5281,
  lambda = 0.004597,
  weights = sample_weights,
  standardize = FALSE,  # 数据已预先标准化
  intercept = TRUE,
  thresh = 1e-7,
  maxit = 1000
)

cat("Model training completed.\n")

### 4. 获取预测分数 ###
cat("\nGenerating predictions...\n")

# 获取模型在训练集和测试集的预测分数
train_pred <- predict(weighted_model, newx = x_train, type = "response")[, 1]
test_pred <- predict(weighted_model, newx = x_test, type = "response")[, 1]

# 将预测分数添加到数据中
train_data_with_pred <- train_processed
train_data_with_pred$pred_prob <- train_pred
train_data_with_pred$original_group <- train_data$group  # 保留原始分组信息

test_data_with_pred <- test_processed
test_data_with_pred$pred_prob <- test_pred
test_data_with_pred$original_group <- test_data$group

### 5. 模型性能评估 ###
cat("\n=== 模型性能评估 ===\n")

# 评估函数
calculate_metrics <- function(true_labels, pred_probs, dataset_name) {
  true_numeric <- ifelse(true_labels == "cancer", 1, 0)
  
  # 计算AUC和置信区间
  roc_obj <- roc(true_numeric, pred_probs, ci = TRUE, quiet = TRUE)
  auc_val <- auc(roc_obj)
  auc_ci <- ci.auc(roc_obj, conf.level = 0.95)
  
  # 阈值优化（使用Youden指数）
  coords <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"))
  best_thresh <- coords[1, "threshold"]
  
  # 使用最佳阈值进行分类
  pred_class <- ifelse(pred_probs > best_thresh, "cancer", "control")
  pred_class <- factor(pred_class, levels = c("control", "cancer"))
  true_class <- factor(true_labels, levels = c("control", "cancer"))
  
  cm <- confusionMatrix(pred_class, true_class, positive = "cancer")
  
  # 计算Brier评分
  brier_score <- mean((pred_probs - true_numeric)^2)
  
  # 计算各项指标
  acc <- cm$overall["Accuracy"]
  sens <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  ppv <- cm$byClass["Pos Pred Value"]
  npv <- cm$byClass["Neg Pred Value"]
  ydi <- sens + spec - 1
  f1 <- 2 * (ppv * sens) / (ppv + sens)
  
  result <- data.frame(
    Dataset = dataset_name,
    AUC = round(auc_val, 3),
    AUC_95CI = sprintf("%.3f (%.3f-%.3f)", auc_val, auc_ci[1], auc_ci[3]),
    AUC_lower = round(auc_ci[1], 3),
    AUC_upper = round(auc_ci[3], 3),
    Brier_Score = round(brier_score, 4),
    ACC = round(acc, 3),
    SENS = round(sens, 3),
    SPEC = round(spec, 3),
    PPV = round(ppv, 3),
    NPV = round(npv, 3),
    YDI = round(ydi, 3),
    F1 = round(f1, 3),
    Best_Threshold = round(best_thresh, 4)
  )
  
  return(list(metrics = result, roc_obj = roc_obj, cm = cm))
}

# 计算训练集和测试集指标
train_result <- calculate_metrics(y_train, train_pred, "Training")
test_result <- calculate_metrics(y_test, test_pred, "Testing")

# 打印结果
cat("\n训练集性能:\n")
print(train_result$metrics)
cat("\n测试集性能:\n")
print(test_result$metrics)

# 保存性能指标
performance_df <- rbind(train_result$metrics, test_result$metrics)
write.csv(performance_df, file.path(DATA_DIR, "weighted_model_performance.csv"), row.names = FALSE)

### 6. 特征重要性（非零系数）###
cat("\n=== 特征重要性 ===\n")

# 获取系数
coefficients <- coef(weighted_model)
coef_matrix <- as.matrix(coefficients)
nonzero_idx <- which(coef_matrix != 0)
feature_names <- rownames(coef_matrix)

# 提取非零系数（排除截距项）
nonzero_idx <- nonzero_idx[feature_names[nonzero_idx] != "(Intercept)"]
feature_importance <- data.frame(
  Feature = feature_names[nonzero_idx],
  Coefficient = coef_matrix[nonzero_idx, 1]
)
feature_importance$Abs_Coefficient <- abs(feature_importance$Coefficient)
feature_importance <- feature_importance[order(-feature_importance$Abs_Coefficient), ]

cat(sprintf("非零系数数量（不包括截距）: %d / %d\n", 
            nrow(feature_importance), ncol(x_train)))

# 保存特征重要性
write.csv(feature_importance, file.path(DATA_DIR, "weighted_model_feature_importance.csv"), 
          row.names = FALSE)

# 打印Top 10特征
cat("\nTop 10 重要特征:\n")
print(head(feature_importance, 10))
# 使用你实际有的临床特征（根据你的热图）
clinical_features <- c("pred_prob", "ALB","age","HGB", "DBIL", "gender", "GLB")

# 检查哪些特征在数据中存在
available_features <- clinical_features[clinical_features %in% colnames(train_data_with_pred)]
cat("Available features for correlation analysis:", paste(available_features, collapse = ", "), "\n")

# 使用训练集数据计算相关性
if (length(available_features) > 1) {
  # 提取相关数据
  cor_data <- train_data_with_pred[, available_features]
  
  # 确保数值型
  for (col in available_features) {
    if (is.factor(cor_data[[col]])) {
      cor_data[[col]] <- as.numeric(cor_data[[col]])
    } else if (is.character(cor_data[[col]])) {
      cor_data[[col]] <- as.numeric(factor(cor_data[[col]]))
    }
  }
  
  # 计算Spearman相关性矩阵
  cor_matrix <- cor(cor_data, method = "spearman", use = "complete.obs")
  
  # 重命名行列以增加可读性
  feature_names <- c(
    "pred_prob", "ALB","age","HGB", "DBIL", "gender", "GLB"
  )[1:length(available_features)]
  
  rownames(cor_matrix) <- colnames(cor_matrix) <- feature_names
  
  # 方法1：使用corrplot直接绘制三角热图
  tiff(file.path(FIG_DIR, "glmnet_fixed_correlation_triangle_corrplot.tiff"), 
       width = 6, height = 6, units = "in", res = 300, compression = "lzw")
  
  # 绘制下三角热图
  col <- colorRampPalette(c(  "#e41a1c","#F7F7F7","#4daf4a"))(200)
  corrplot(cor_matrix,
           method = "square",
           type = "lower",           # 只显示下三角
           order = "original",       # 保持原始顺序
           tl.col = "black",
           tl.srt = 45,              # 标签旋转角度
           tl.cex = 1.4,             # 标签字体大小
           cl.cex = 1.4,             # 图例字体大小
           addCoef.col = "black",    # 系数颜色
           number.cex = 1.2,         # 系数字体大小
           diag = TRUE,              # 显示对角线
           col = col,       # 颜色方案
           mar = c(0, 0, 2, 0),      # 边距
           title = "")
  
  dev.off()
  
  
}

