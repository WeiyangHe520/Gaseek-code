# ============================================================
# glmnet模型：类别权重 + 阈值优化
# 输出训练集和测试集完整结果
# ============================================================

rm(list = ls())
library(glmnet)
library(pROC)
library(caret)

set.seed(3456)

# 加载数据
load(file = ".left_data.rdata")

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

cat("=== 原始数据分布 ===\n")
cat("训练集分组分布:\n")
print(table(train_data$group))
cat("\n测试集分组分布:\n")
print(table(test_data$group))

# ============================================================
# 数据预处理 - 标准化特征
# ============================================================
preprocess_data <- function(data) {
  features <- data[, feature_cols]
  group <- data$group
  
  # 标准化特征
  features_scaled <- as.data.frame(scale(features))
  
  return(data.frame(group = group, features_scaled))
}

# 预处理数据
train_processed <- preprocess_data(train_data)
test_processed <- preprocess_data(test_data)

# 准备矩阵和标签
train_matrix <- as.matrix(train_processed[, -1])
test_matrix <- as.matrix(test_processed[, -1])

train_labels <- ifelse(train_processed$group == "cancer", 1, 0)
test_labels <- ifelse(test_processed$group == "cancer", 1, 0)

# ============================================================
# 计算类别权重
# ============================================================
cancer_count <- sum(train_labels == 1)
control_count <- sum(train_labels == 0)

# 逆频率权重
weight_cancer <- control_count / cancer_count
weight_control <- 1

cat(sprintf("\n癌症样本数: %d, 对照样本数: %d\n", cancer_count, control_count))
cat(sprintf("类别权重: 癌症=%.3f, 对照=%.3f\n", weight_cancer, weight_control))

# 创建权重向量
sample_weights <- ifelse(train_labels == 1, weight_cancer, weight_control)

# ============================================================
# 定义glmnet参数
# ============================================================
alpha_value <- 0.5281
lambda_value <- 0.004597

# ============================================================
# 训练带权重的glmnet模型
# ============================================================
model_glmnet_weighted <- glmnet(
  x = train_matrix,
  y = train_labels,
  family = "binomial",
  alpha = alpha_value,
  lambda = lambda_value,
  weights = sample_weights,
  standardize = FALSE,
  intercept = TRUE,
  thresh = 1e-7,
  maxit = 1000
)

# ============================================================
# 预测概率
# ============================================================
train_prob <- predict(model_glmnet_weighted, newx = train_matrix, type = "response")[,1]
test_prob <- predict(model_glmnet_weighted, newx = test_matrix, type = "response")[,1]

# ============================================================
# 获取非零系数数量
# ============================================================
coefficients <- coef(model_glmnet_weighted)
non_zero_coef <- sum(coefficients != 0) - 1
cat(sprintf("\n非零系数数量（不包括截距）: %d\n", non_zero_coef))

# ============================================================
# 评估函数（包含训练集和测试集）
# ============================================================
calculate_metrics <- function(true_labels, pred_probs, dataset_name, optimize_threshold = TRUE) {
  
  true_numeric <- ifelse(true_labels == "cancer", 1, 0)
  
  # 计算AUC和置信区间
  roc_obj <- roc(true_numeric, pred_probs, ci = TRUE)
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
    Alpha = alpha_value,
    Lambda = lambda_value,
    NonZero_Features = non_zero_coef,
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
  
  return(result)
}

# ============================================================
# 计算训练集和测试集指标
# ============================================================
cat("\n========================================\n")
cat("模型评估结果\n")
cat("========================================\n")

train_metrics <- calculate_metrics(train_data$group, train_prob, "Training")
test_metrics <- calculate_metrics(test_data$group, test_prob, "Testing")

# 合并结果
results_df <- rbind(train_metrics, test_metrics)

# 打印结果
cat("\n=== 结果汇总 ===\n")
print(results_df)

# 保存结果到CSV
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
            train_metrics$AUC, train_metrics$AUC_lower, train_metrics$AUC_upper))
cat(sprintf("  ACC = %.3f\n", train_metrics$ACC))
cat(sprintf("  SENS = %.3f\n", train_metrics$SENS))
cat(sprintf("  SPEC = %.3f\n", train_metrics$SPEC))
cat(sprintf("  PPV = %.3f\n", train_metrics$PPV))
cat(sprintf("  NPV = %.3f\n", train_metrics$NPV))
cat(sprintf("  F1 = %.3f\n", train_metrics$F1))
cat(sprintf("  Brier = %.4f\n", train_metrics$Brier_Score))
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
# 特征重要性
# ============================================================
feature_importance <- data.frame(
  Feature = rownames(coefficients)[-1],
  Coefficient = as.vector(coefficients[-1]),
  Abs_Coefficient = abs(as.vector(coefficients[-1]))
)
feature_importance <- feature_importance[order(-feature_importance$Abs_Coefficient), ]

cat("\n=== Top 10 重要特征 ===\n")
print(head(feature_importance, 10))

# 保存特征重要性
write.csv(feature_importance, "./glmnet_weighted_feature_importance.csv", row.names = FALSE)

# ============================================================
# 绘制ROC曲线
# ============================================================
png("./glmnet_weighted_roc.png", width = 8, height = 6, units = "in", res = 300)

roc_train <- roc(ifelse(train_data$group == "cancer", 1, 0), train_prob)
roc_test <- roc(ifelse(test_data$group == "cancer", 1, 0), test_prob)

plot(roc_train, col = "blue", lwd = 2, main = "类别权重glmnet模型ROC曲线")
plot(roc_test, col = "red", lwd = 2, add = TRUE)

legend("bottomright", 
       legend = c(sprintf("训练集 (AUC=%.3f)", auc(roc_train)),
                  sprintf("测试集 (AUC=%.3f)", auc(roc_test))),
       col = c("blue", "red"), lwd = 2, cex = 0.8)

dev.off()
cat("\nROC曲线已保存: ./glmnet_weighted_roc.png\n")

# ============================================================
# 保存模型
# ============================================================
save(model_glmnet_weighted, file = "./glmnet_weighted_model.rdata")
cat("模型已保存: ./glmnet_weighted_model.rdata\n")

cat("\n=== glmnet类别权重模型训练完成 ===\n")