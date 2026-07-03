# ============================================================
# 弹性网逻辑回归模型 - 带样本权重和特征标准化
# ============================================================

rm(list = ls())
set.seed(278)

# 加载必要的包
library(caret)
library(pROC)
library(glmnet)

# ============================================================
# 数据预处理 - 标准化特征
# ============================================================
preprocess_data <- function(data, feature_cols) {
  features <- data[, feature_cols]
  group <- data$group
  
  # 标准化特征
  features_scaled <- as.data.frame(scale(features))
  
  return(data.frame(group = group, features_scaled))
}

# 定义特征列
feature_cols <- c("gender", "age", "BMI", "Smoking_history", "drinking_history", 
                  "Family_history_of_cancer", "LYMPH_percentage", "MONO_percentage", 
                  "HGB", "MCV", "MCHC", "PLT", "DBIL", "IBIL", "ALB", "GLB", "ALT")

# 加载原始数据
load(file = ".left_data.rdata")

# 预处理数据
train_processed <- preprocess_data(train_data, feature_cols)
test_processed <- preprocess_data(test_data, feature_cols)

# 准备矩阵和标签
train_matrix <- as.matrix(train_processed[, -1])
test_matrix <- as.matrix(test_processed[, -1])

train_labels <- ifelse(train_processed$group == "cancer", 1, 0)
test_labels <- ifelse(test_processed$group == "cancer", 1, 0)

# 保存原始标签（用于评估）
train_group <- train_processed$group
test_group <- test_processed$group

cat("=== 数据预处理完成 ===\n")
cat(sprintf("训练集样本数: %d, 特征数: %d\n", nrow(train_matrix), ncol(train_matrix)))
cat(sprintf("测试集样本数: %d, 特征数: %d\n", nrow(test_matrix), ncol(test_matrix)))
cat("训练集类别分布:\n")
print(table(train_group))
cat("测试集类别分布:\n")
print(table(test_group))

# ============================================================
# 计算类别权重（逆频率加权）
# ============================================================
cancer_count <- sum(train_labels == 1)
control_count <- sum(train_labels == 0)

# 逆频率权重：给少数类更高的权重
weight_cancer <- control_count / cancer_count
weight_control <- 1

cat("\n=== 类别权重计算 ===\n")
cat(sprintf("癌症样本数: %d, 对照样本数: %d\n", cancer_count, control_count))
cat(sprintf("类别权重: 癌症=%.3f, 对照=%.3f\n", weight_cancer, weight_control))

# 创建权重向量
sample_weights <- ifelse(train_labels == 1, weight_cancer, weight_control)

# ============================================================
# 定义glmnet参数
# ============================================================
alpha_value <- 0.5281   # 弹性网混合参数 (0=Ridge, 1=Lasso)
lambda_value <- 0.004597  # 正则化强度

cat("\n=== 模型参数配置 ===\n")
cat(sprintf("alpha = %.4f (L1/L2混合参数)\n", alpha_value))
cat(sprintf("lambda = %.6f (正则化强度)\n", lambda_value))
cat("standardize = FALSE (数据已预先标准化)\n")
cat("intercept = TRUE\n")
cat("family = binomial\n")

# ============================================================
# 训练带权重的glmnet模型
# ============================================================
cat("\n=== 训练弹性网逻辑回归模型 ===\n")

model_glmnet_weighted <- glmnet(
  x = train_matrix,
  y = train_labels,
  family = "binomial",
  alpha = alpha_value,
  lambda = lambda_value,
  weights = sample_weights,
  standardize = FALSE,      # 数据已预先标准化
  intercept = TRUE,
  thresh = 1e-7,
  maxit = 1000
)

cat("模型训练完成\n")

# ============================================================
# 预测概率
# ============================================================
train_prob <- predict(model_glmnet_weighted, newx = train_matrix, type = "response")[,1]
test_prob <- predict(model_glmnet_weighted, newx = test_matrix, type = "response")[,1]

cat(sprintf("\n预测概率范围:\n"))
cat(sprintf("  训练集: [%.4f, %.4f]\n", min(train_prob), max(train_prob)))
cat(sprintf("  测试集: [%.4f, %.4f]\n", min(test_prob), max(test_prob)))

# ============================================================
# 获取非零系数数量（修复S4类错误）
# ============================================================
# 方法1：使用 as.matrix() 转换系数矩阵
coef_matrix <- as.matrix(coef(model_glmnet_weighted))
non_zero_coef <- sum(coef_matrix != 0) - 1  # 减去截距项

# 获取特征名称（截距项是 "(Intercept)"）
feature_names <- rownames(coef_matrix)

cat(sprintf("\n非零系数数量（不包括截距）: %d / %d\n", non_zero_coef, ncol(train_matrix)))
cat(sprintf("模型稀疏度: %.1f%%\n", (1 - non_zero_coef/ncol(train_matrix)) * 100))

# 打印非零系数
if (non_zero_coef > 0) {
  # 提取非零系数（排除截距项）
  nonzero_idx <- which(coef_matrix != 0)
  # 排除截距项（通常是第一个）
  nonzero_idx <- nonzero_idx[feature_names[nonzero_idx] != "(Intercept)"]
  
  coef_df <- data.frame(
    Feature = feature_names[nonzero_idx],
    Coefficient = coef_matrix[nonzero_idx, 1]
  )
  coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]
  cat("\n非零系数:\n")
  print(coef_df)
  
  # 保存系数到CSV
  write.csv(coef_df, "./glmnet_weighted_coefficients.csv", row.names = FALSE)
  cat("\n系数已保存到: ./glmnet_weighted_coefficients.csv\n")
}

# ============================================================
# 评估函数（包含训练集和测试集）
# ============================================================
calculate_metrics <- function(true_labels, pred_probs, dataset_name, alpha_val, lambda_val, nz_coef, optimize_threshold = TRUE) {
  
  true_numeric <- ifelse(true_labels == "cancer", 1, 0)
  
  # 计算AUC和置信区间
  roc_obj <- roc(true_numeric, pred_probs, ci = TRUE, quiet = TRUE)
  auc_val <- auc(roc_obj)
  auc_ci <- ci.auc(roc_obj, conf.level = 0.95)
  
  # 阈值优化（使用Youden指数）
  if (optimize_threshold) {
    coords <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"))
    best_thresh <- coords[1, "threshold"]
  } else {
    best_thresh <- 0.5
  }
  
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
    Alpha = alpha_val,
    Lambda = lambda_val,
    NonZero_Features = nz_coef,
    Model_Type = "glmnet_weighted",
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

# ============================================================
# 计算训练集和测试集指标
# ============================================================
cat("\n========================================\n")
cat("模型评估结果\n")
cat("========================================\n")

train_result <- calculate_metrics(train_group, train_prob, "Training", 
                                  alpha_value, lambda_value, non_zero_coef)
test_result <- calculate_metrics(test_group, test_prob, "Testing",
                                 alpha_value, lambda_value, non_zero_coef)

# 合并结果
results_df <- rbind(train_result$metrics, test_result$metrics)

# 打印结果
cat("\n=== 结果汇总 ===\n")
print(results_df)

# ============================================================
# 保存结果到CSV
# ============================================================
csv_filename <- "./glmnet_weighted_results.csv"
write.csv(results_df, file = csv_filename, row.names = FALSE)
cat(sprintf("\n结果已保存到: %s\n", csv_filename))

# ============================================================
# 打印详细报告
# ============================================================
cat("\n========================================\n")
cat("详细性能报告\n")
cat("========================================\n")

cat("\n训练集:\n")
cat(sprintf("  AUC = %.3f (95%% CI: %.3f-%.3f)\n", 
            train_result$metrics$AUC, 
            train_result$metrics$AUC_lower, 
            train_result$metrics$AUC_upper))
cat(sprintf("  ACC = %.3f\n", train_result$metrics$ACC))
cat(sprintf("  SENS = %.3f\n", train_result$metrics$SENS))
cat(sprintf("  SPEC = %.3f\n", train_result$metrics$SPEC))
cat(sprintf("  PPV = %.3f\n", train_result$metrics$PPV))
cat(sprintf("  NPV = %.3f\n", train_result$metrics$NPV))
cat(sprintf("  F1 = %.3f\n", train_result$metrics$F1))
cat(sprintf("  YDI = %.3f\n", train_result$metrics$YDI))
cat(sprintf("  Brier = %.4f\n", train_result$metrics$Brier_Score))
cat(sprintf("  最佳阈值 = %.4f\n", train_result$metrics$Best_Threshold))

cat("\n测试集:\n")
cat(sprintf("  AUC = %.3f (95%% CI: %.3f-%.3f)\n", 
            test_result$metrics$AUC, 
            test_result$metrics$AUC_lower, 
            test_result$metrics$AUC_upper))
cat(sprintf("  ACC = %.3f\n", test_result$metrics$ACC))
cat(sprintf("  SENS = %.3f\n", test_result$metrics$SENS))
cat(sprintf("  SPEC = %.3f\n", test_result$metrics$SPEC))
cat(sprintf("  PPV = %.3f\n", test_result$metrics$PPV))
cat(sprintf("  NPV = %.3f\n", test_result$metrics$NPV))
cat(sprintf("  F1 = %.3f\n", test_result$metrics$F1))
cat(sprintf("  YDI = %.3f\n", test_result$metrics$YDI))
cat(sprintf("  Brier = %.4f\n", test_result$metrics$Brier_Score))
cat(sprintf("  最佳阈值 = %.4f\n", test_result$metrics$Best_Threshold))

# ============================================================
# 绘制ROC曲线
# ============================================================
cat("\n=== 绘制ROC曲线 ===\n")


# ============================================================
# 保存模型和结果
# ============================================================
saveRDS(model_glmnet_weighted, file = "./glmnet_weighted_model.rds")
cat("模型已保存: ./glmnet_weighted_model.rds\n")

# 保存完整结果
full_results <- list(
  model = model_glmnet_weighted,
  train_predictions = train_prob,
  test_predictions = test_prob,
  train_metrics = train_result$metrics,
  test_metrics = test_result$metrics,
  coefficients = if(exists("coef_df")) coef_df else NULL,
  parameters = list(
    alpha = alpha_value,
    lambda = lambda_value,
    non_zero_features = non_zero_coef,
    weight_cancer = weight_cancer,
    weight_control = weight_control
  )
)

save(full_results, file = "./glmnet_weighted_full_results.rdata")
cat("完整结果已保存: ./glmnet_weighted_full_results.rdata\n")

# ============================================================
# 完成
# ============================================================
cat("\n========================================\n")
cat("弹性网逻辑回归模型分析完成！\n")
cat("========================================\n")
cat("\n生成的文件:\n")
cat("  1. glmnet_weighted_results.csv - 性能指标汇总\n")
cat("  2. glmnet_weighted_coefficients.csv - 模型系数\n")
cat("  3. glmnet_weighted_model.rds - 训练好的模型\n")
cat("  4. glmnet_weighted_full_results.rdata - 完整结果\n")
cat("  5. glmnet_weighted_train_roc.png - 训练集ROC曲线\n")
cat("  6. glmnet_weighted_test_roc.png - 测试集ROC曲线\n")
cat("  7. glmnet_weighted_roc_comparison.png - ROC对比图\n")
# ============================================================
# 在评估函数之后添加ROC数据提取函数
# ============================================================
extract_roc_points <- function(roc_obj, dataset_name) {
  if (is.null(roc_obj)) return(NULL)
  
  roc_points <- data.frame(
    threshold = roc_obj$thresholds,
    sensitivity = roc_obj$sensitivities,
    specificity = roc_obj$specificities,
    FPR = 1 - roc_obj$specificities,
    TPR = roc_obj$sensitivities
  )
  
  # 添加Youden指数
  roc_points$youden <- roc_points$sensitivity + roc_points$specificity - 1
  
  # 找到最佳阈值点
  best_idx <- which.max(roc_points$youden)
  roc_points$is_best <- FALSE
  roc_points$is_best[best_idx] <- TRUE
  roc_points$best_threshold <- roc_points$threshold[best_idx]
  
  return(roc_points)
}

# ============================================================
# 在计算完指标后添加
# ============================================================
# 提取ROC曲线点
train_roc_points <- extract_roc_points(train_result$roc_obj, "Training")
test_roc_points <- extract_roc_points(test_result$roc_obj, "Testing")

# 保存ROC曲线点数据
param_str <- paste0("alpha", round(alpha_value, 4), "_lambda", round(lambda_value, 6))
param_str <- gsub("\\.", "_", param_str)

if (!is.null(train_roc_points)) {
  train_roc_filename <- paste0("glmnet_train_roc_curve_points_", param_str, ".csv")
  write.csv(train_roc_points, train_roc_filename, row.names = FALSE)
  cat(sprintf("\n训练集ROC曲线点已保存: %s (%d个点)\n", train_roc_filename, nrow(train_roc_points)))
}

if (!is.null(test_roc_points)) {
  test_roc_filename <- paste0("glmnet_test_roc_curve_points_", param_str, ".csv")
  write.csv(test_roc_points, test_roc_filename, row.names = FALSE)
  cat(sprintf("测试集ROC曲线点已保存: %s (%d个点)\n", test_roc_filename, nrow(test_roc_points)))
}

# 可选：保存简化的ROC数据（仅FPR和TPR用于绘图）
if (!is.null(train_roc_points)) {
  train_roc_simple <- train_roc_points[, c("FPR", "TPR", "threshold")]
  write.csv(train_roc_simple, paste0("glmnet_train_roc_simple_", param_str, ".csv"), row.names = FALSE)
}

if (!is.null(test_roc_points)) {
  test_roc_simple <- test_roc_points[, c("FPR", "TPR", "threshold")]
  write.csv(test_roc_simple, paste0("glmnet_test_roc_simple_", param_str, ".csv"), row.names = FALSE)
}