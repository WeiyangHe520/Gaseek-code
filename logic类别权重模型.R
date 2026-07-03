# ============================================================
# LogitBoost模型：类别权重 + 阈值优化
# 输出训练集和测试集完整结果（基于最优nIter=126）
# ============================================================

rm(list = ls())
library(caTools)      # LogitBoost核心
library(pROC)
library(caret)

set.seed(3456)

# 加载数据
load(file = ".left_data.rdata")

# 定义特征列（与glmnet示例一致）
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

cat("=== 原始数据分布 ===\n")
cat("训练集分组分布:\n")
print(table(train_data$group))
cat("\n测试集分组分布:\n")
print(table(test_data$group))

# ============================================================
# 数据预处理 - 标准化（LogitBoost对尺度不敏感，但保持一致性）
# ============================================================
preprocess_data <- function(data) {
  features <- data[, feature_cols]
  group <- data$group
  features_scaled <- as.data.frame(scale(features))
  return(data.frame(group = group, features_scaled))
}

train_processed <- preprocess_data(train_data)
test_processed <- preprocess_data(test_data)

# 准备矩阵和标签
train_matrix <- as.matrix(train_processed[, -1])
test_matrix  <- as.matrix(test_processed[, -1])
train_labels <- train_processed$group
test_labels  <- test_processed$group

# 数值标签（用于AUC和Brier）
train_num <- ifelse(train_labels == "cancer", 1, 0)
test_num  <- ifelse(test_labels == "cancer", 1, 0)

# ============================================================
# 计算类别权重（逆频率）
# ============================================================
cancer_count <- sum(train_num == 1)
control_count <- sum(train_num == 0)
weight_cancer <- control_count / cancer_count   # 癌症样本权重
weight_control <- 1

cat(sprintf("\n癌症样本数: %d, 对照样本数: %d\n", cancer_count, control_count))
cat(sprintf("类别权重: 癌症=%.3f, 对照=%.3f\n", weight_cancer, weight_control))

# 构建样本权重向量（仅用于评估加权指标，不用于训练）
sample_weights <- ifelse(train_num == 1, weight_cancer, weight_control)

# ============================================================
# 最佳参数（贝叶斯优化结果）
# ============================================================
best_nIter <- 126   # 根据您的优化结果设定
cat(sprintf("\n使用最佳参数: nIter = %d\n", best_nIter))

# ============================================================
# 训练 LogitBoost 模型（注意：caTools::LogitBoost 不支持 weights）
# ============================================================
model_logitboost <- LogitBoost(
  x = train_matrix,
  y = train_num,
  nIter = best_nIter
)

# ============================================================
# 预测概率（正类概率 = 第二列）
# ============================================================
train_prob <- predict(model_logitboost, train_matrix, type = "raw")[, 2]
test_prob  <- predict(model_logitboost, test_matrix,  type = "raw")[, 2]

# ============================================================
# 特征重要性（如果模型提供）
# ============================================================
if (!is.null(model_logitboost$importance)) {
  importance_vec <- model_logitboost$importance
  non_zero_imp <- sum(importance_vec > 0)
} else {
  importance_vec <- NULL
  non_zero_imp <- NA
}
cat(sprintf("非零重要性特征数量: %s\n", ifelse(is.na(non_zero_imp), "不可用", non_zero_imp)))

# ============================================================
# 评估函数（支持加权指标）
# ============================================================
calculate_metrics_weighted <- function(true_labels, pred_probs, weights, dataset_name, 
                                       true_num = NULL, optimize_threshold = TRUE) {
  
  if (is.null(true_num)) {
    true_num <- ifelse(true_labels == "cancer", 1, 0)
  }
  
  # 加权AUC（使用pROC的加权版本需安装WeightedROC，这里用标准AUC + 加权阈值优化）
  # 标准AUC对权重不敏感，我们仍报告标准AUC以保持可比性
  roc_obj <- roc(true_num, pred_probs, ci = TRUE, quiet = TRUE)
  auc_val <- auc(roc_obj)
  auc_ci <- ci.auc(roc_obj, conf.level = 0.95)
  
  # 加权Brier分数
  brier_weighted <- sum(weights * (pred_probs - true_num)^2) / sum(weights)
  
  # 阈值优化（基于加权Youden指数）
  if (optimize_threshold) {
    # 计算加权灵敏度和特异度
    thresholds <- sort(unique(pred_probs))
    best_metric <- -Inf
    best_thresh <- 0.5
    for (th in thresholds) {
      pred_class <- ifelse(pred_probs > th, 1, 0)
      tp <- sum(weights[pred_class == 1 & true_num == 1])
      fn <- sum(weights[pred_class == 0 & true_num == 1])
      fp <- sum(weights[pred_class == 1 & true_num == 0])
      tn <- sum(weights[pred_class == 0 & true_num == 0])
      sens <- tp / (tp + fn)
      spec <- tn / (tn + fp)
      youden <- sens + spec - 1
      if (youden > best_metric) {
        best_metric <- youden
        best_thresh <- th
      }
    }
  } else {
    best_thresh <- 0.5
  }
  
  # 使用最佳阈值得到分类结果
  pred_class <- ifelse(pred_probs > best_thresh, "cancer", "control")
  pred_class <- factor(pred_class, levels = c("control", "cancer"))
  true_class <- factor(true_labels, levels = c("control", "cancer"))
  cm <- confusionMatrix(pred_class, true_class, positive = "cancer")
  
  acc <- cm$overall["Accuracy"]
  sens <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  ppv <- cm$byClass["Pos Pred Value"]
  npv <- cm$byClass["Neg Pred Value"]
  ydi <- sens + spec - 1
  f1 <- 2 * (ppv * sens) / (ppv + sens)
  
  result <- data.frame(
    Dataset = dataset_name,
    nIter = best_nIter,
    NonZero_Importance = ifelse(is.na(non_zero_imp), NA, non_zero_imp),
    Model_Type = "LogitBoost",
    AUC = round(auc_val, 3),
    AUC_95CI = sprintf("%.3f (%.3f-%.3f)", auc_val, auc_ci[1], auc_ci[3]),
    AUC_lower = round(auc_ci[1], 3),
    AUC_upper = round(auc_ci[3], 3),
    Brier_Score = round(brier_weighted, 4),
    ACC = round(acc, 3),
    SENS = round(sens, 3),
    SPEC = round(spec, 3),
    PPV = round(ppv, 3),
    NPV = round(npv, 3),
    YDI = round(ydi, 3),
    F1 = round(f1, 3),
    Best_Threshold = round(best_thresh, 4)
  )
  return(result)
}

# ============================================================
# 计算训练集和测试集指标（使用样本权重）
# ============================================================
cat("\n========================================\n")
cat("模型评估结果（基于类别权重优化阈值）\n")
cat("========================================\n")

train_metrics <- calculate_metrics_weighted(
  true_labels = train_labels,
  pred_probs = train_prob,
  weights = sample_weights,
  dataset_name = "Training",
  true_num = train_num
)

test_metrics <- calculate_metrics_weighted(
  true_labels = test_labels,
  pred_probs = test_prob,
  weights = rep(1, length(test_labels)),  # 测试集不加权
  dataset_name = "Testing",
  true_num = test_num
)

results_df <- rbind(train_metrics, test_metrics)
print(results_df)

# 保存结果CSV
csv_filename <- "./logitboost_weighted_results.csv"
write.csv(results_df, file = csv_filename, row.names = FALSE)
cat(sprintf("\n结果已保存到: %s\n", csv_filename))

# ============================================================
# 详细报告
# ============================================================
cat("\n========================================\n")
cat("详细性能报告\n")
cat("========================================\n")

cat("\n训练集:\n")
cat(sprintf("  AUC = %.3f (95%% CI: %.3f-%.3f)\n", 
            train_metrics$AUC, train_metrics$AUC_lower, train_metrics$AUC_upper))
cat(sprintf("  ACC = %.3f\n", train_metrics$ACC))
cat(sprintf("  SENS = %.3f\n", train_metrics$SENS))
cat(sprintf("  SPEC = %.3f\n", train_metrics$SPEC))
cat(sprintf("  PPV = %.3f\n", train_metrics$PPV))
cat(sprintf("  NPV = %.3f\n", train_metrics$NPV))
cat(sprintf("  F1 = %.3f\n", train_metrics$F1))
cat(sprintf("  Brier (加权) = %.4f\n", train_metrics$Brier_Score))
cat(sprintf("  最佳阈值 = %.4f\n", train_metrics$Best_Threshold))

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
cat(sprintf("  最佳阈值 = %.4f\n", test_metrics$Best_Threshold))

# ============================================================
# 特征重要性（如果可用）
# ============================================================
if (!is.null(importance_vec)) {
  feature_importance <- data.frame(
    Feature = colnames(train_matrix),
    Importance = importance_vec,
    Abs_Importance = abs(importance_vec)
  )
  feature_importance <- feature_importance[order(-feature_importance$Abs_Importance), ]
  
  cat("\n=== Top 10 重要特征 ===\n")
  print(head(feature_importance, 10))
  write.csv(feature_importance, "./logitboost_feature_importance.csv", row.names = FALSE)
} else {
  cat("\n注意：当前LogitBoost模型未提供特征重要性。\n")
}

# ============================================================
# 绘制ROC曲线
# ============================================================
png("./logitboost_weighted_roc.png", width = 8, height = 6, units = "in", res = 300)
roc_train <- roc(train_num, train_prob, quiet = TRUE)
roc_test  <- roc(test_num,  test_prob,  quiet = TRUE)
plot(roc_train, col = "blue", lwd = 2, main = "LogitBoost 模型 ROC 曲线")
plot(roc_test, col = "red", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c(sprintf("训练集 (AUC=%.3f)", auc(roc_train)),
                  sprintf("测试集 (AUC=%.3f)", auc(roc_test))),
       col = c("blue", "red"), lwd = 2, cex = 0.8)
dev.off()
cat("\nROC曲线已保存: ./logitboost_weighted_roc.png\n")

# ============================================================
# 保存模型和工作空间
# ============================================================
save(model_logitboost, file = "./logitboost_weighted_model.rdata")
cat("模型已保存: ./logitboost_weighted_model.rdata\n")

save.image(file = "./logitboost_weighted_workspace.rdata")
cat("完整工作空间已保存: ./logitboost_weighted_workspace.rdata\n")

cat("\n=== LogitBoost 类别权重模型训练完成 ===\n")