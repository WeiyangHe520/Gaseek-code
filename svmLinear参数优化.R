# ============================================================
# svmLinear模型：类别权重 + 阈值优化
# 输出训练集和测试集完整结果
# ============================================================

rm(list = ls())
library(e1071)
library(pROC)
library(caret)
library(ggplot2)

set.seed(3456)

# 加载数据
load(file = ".left_data.rdata")

# 定义特征列
feature_cols <- c("gender", "age", "BMI", "Smoking_history", "drinking_history", 
                  "Family_history_of_cancer", "LYMPH_percentage", "MONO_percentage", 
                  "HGB", "MCV", "MCHC", "PLT", "DBIL", "IBIL", "ALB", "GLB", "ALT")

cat("=== 原始数据分布 ===\n")
cat("训练集分组分布:\n")
print(table(train_data$group))
cat("\n测试集分组分布:\n")
print(table(test_data$group))

# ============================================================
# 数据预处理 - 标准化特征（SVM对尺度敏感，必须标准化）
# ============================================================
preprocess_data <- function(data, train_stats = NULL) {
  features <- data[, feature_cols]
  group <- data$group
  
  # 标准化特征 - 使用训练集的均值和标准差
  if (is.null(train_stats)) {
    # 训练集：计算均值和标准差
    means <- apply(features, 2, mean, na.rm = TRUE)
    sds <- apply(features, 2, sd, na.rm = TRUE)
    features_scaled <- as.data.frame(scale(features, center = means, scale = sds))
    stats <- list(means = means, sds = sds)
    return(list(data = data.frame(group = group, features_scaled), stats = stats))
  } else {
    # 测试集：使用训练集的均值和标准差
    features_scaled <- as.data.frame(scale(features, 
                                           center = train_stats$means, 
                                           scale = train_stats$sds))
    return(data.frame(group = group, features_scaled))
  }
}

# 预处理数据
train_result <- preprocess_data(train_data)
train_processed <- train_result$data
train_stats <- train_result$stats
test_processed <- preprocess_data(test_data, train_stats)

# 准备矩阵和标签
train_matrix <- train_processed[, -1]
test_matrix <- test_processed[, -1]

train_labels <- train_processed$group
test_labels <- test_processed$group

# 转换为因子（e1071 svm要求）
train_labels_factor <- factor(train_labels, levels = c("control", "cancer"))
test_labels_factor <- factor(test_labels, levels = c("control", "cancer"))

# ============================================================
# 计算类别权重（用于成本敏感学习）
# ============================================================
cancer_count <- sum(train_labels == "cancer")
control_count <- sum(train_labels == "control")

# 逆频率权重
weight_cancer <- control_count / cancer_count
weight_control <- 1

cat(sprintf("\n癌症样本数: %d, 对照样本数: %d\n", cancer_count, control_count))
cat(sprintf("类别权重: 癌症=%.3f, 对照=%.3f\n", weight_cancer, weight_control))

# 创建类别权重向量（用于svm的class.weights参数）
class_weights <- c("control" = weight_control, "cancer" = weight_cancer)

# ============================================================
# 定义svmLinear参数
# ============================================================
C_value <- 3.830174  # 成本参数

# ============================================================
# 训练带权重的svmLinear模型
# ============================================================
cat("\n训练svmLinear模型...\n")
model_svm_weighted <- svm(
  x = train_matrix,
  y = train_labels_factor,
  kernel = "linear",           # 线性核
  cost = C_value,              # 成本参数
  class.weights = class_weights,  # 类别权重
  probability = TRUE,          # 启用概率预测
  scale = FALSE,               # 数据已经手动标准化
  type = "C-classification",   # 分类类型
  tolerance = 0.001,           # 收敛容差
  cache_size = 40              # 缓存大小(MB)
)

# ============================================================
# 预测概率
# ============================================================
cat("预测概率...\n")
# 训练集预测
train_pred_prob_attr <- predict(model_svm_weighted, train_matrix, probability = TRUE)
train_prob <- attr(train_pred_prob_attr, "probabilities")[, "cancer"]

# 测试集预测
test_pred_prob_attr <- predict(model_svm_weighted, test_matrix, probability = TRUE)
test_prob <- attr(test_pred_prob_attr, "probabilities")[, "cancer"]

# 获取支持向量数量
n_support_vectors <- nrow(model_svm_weighted$SV)
cat(sprintf("\n支持向量数量: %d (%.1f%% of training data)\n", 
            n_support_vectors, 100 * n_support_vectors / nrow(train_matrix)))

# ============================================================
# 获取系数（仅对线性核有效）
# ============================================================
# 提取权重向量
w <- t(model_svm_weighted$coefs) %*% model_svm_weighted$SV
if (ncol(w) == 0) {
  # 如果没有支持向量（不应该发生），则设为零向量
  coefficients <- rep(0, ncol(train_matrix))
} else {
  coefficients <- as.vector(w)
}
intercept <- -model_svm_weighted$rho

# 计算特征重要性（基于系数的绝对值）
feature_importance <- data.frame(
  Feature = colnames(train_matrix),
  Coefficient = coefficients,
  Abs_Coefficient = abs(coefficients)
)
feature_importance <- feature_importance[order(-feature_importance$Abs_Coefficient), ]

# 统计非零系数数量（考虑数值精度）
non_zero_coef <- sum(abs(coefficients) > 1e-6)
cat(sprintf("非零系数数量: %d / %d\n", non_zero_coef, length(coefficients)))

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
    C_Parameter = C_value,
    Kernel = "linear",
    NonZero_Features = non_zero_coef,
    N_SupportVectors = n_support_vectors,
    Model_Type = "svmLinear_weighted",
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

train_metrics <- calculate_metrics(train_labels, train_prob, "Training")
test_metrics <- calculate_metrics(test_labels, test_prob, "Testing")

# 合并结果
results_df <- rbind(train_metrics, test_metrics)

# 打印结果
cat("\n=== 结果汇总 ===\n")
print(results_df)

# 保存结果到CSV
csv_filename <- "./svmLinear_weighted_results.csv"
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
cat("\n=== Top 10 重要特征 ===\n")
print(head(feature_importance, 10))

# 保存特征重要性
write.csv(feature_importance, "./svmLinear_weighted_feature_importance.csv", row.names = FALSE)

# ============================================================
# 绘制ROC曲线
# ============================================================
png("./svmLinear_weighted_roc.png", width = 8, height = 6, units = "in", res = 300)

roc_train <- roc(ifelse(train_labels == "cancer", 1, 0), train_prob)
roc_test <- roc(ifelse(test_labels == "cancer", 1, 0), test_prob)

plot(roc_train, col = "blue", lwd = 2, main = "类别权重svmLinear模型ROC曲线")
plot(roc_test, col = "red", lwd = 2, add = TRUE)

legend("bottomright", 
       legend = c(sprintf("训练集 (AUC=%.3f)", auc(roc_train)),
                  sprintf("测试集 (AUC=%.3f)", auc(roc_test))),
       col = c("blue", "red"), lwd = 2, cex = 0.8)

dev.off()
cat("\nROC曲线已保存: ./svmLinear_weighted_roc.png\n")

# ============================================================
# 绘制决策边界（如果特征数>=2）
# ============================================================
if (ncol(train_matrix) >= 2) {
  # 选择最重要的两个特征
  top2_features <- head(feature_importance$Feature, 2)
  
  # 创建网格
  grid_size <- 100
  feature1_range <- seq(min(train_matrix[[top2_features[1]]]), 
                        max(train_matrix[[top2_features[1]]]), 
                        length.out = grid_size)
  feature2_range <- seq(min(train_matrix[[top2_features[2]]]), 
                        max(train_matrix[[top2_features[2]]]), 
                        length.out = grid_size)
  grid <- expand.grid(
    setNames(list(feature1_range, feature2_range), top2_features)
  )
  
  # 补全其他特征为均值
  other_features <- setdiff(colnames(train_matrix), top2_features)
  for (feat in other_features) {
    grid[[feat]] <- mean(train_matrix[[feat]])
  }
  
  # 预测网格
  grid_pred <- predict(model_svm_weighted, grid, probability = TRUE)
  grid_prob <- attr(grid_pred, "probabilities")[, "cancer"]
  grid$Decision <- grid_prob
  
  # 准备可视化数据
  train_viz <- train_matrix[, top2_features]
  train_viz$group <- train_labels
  train_viz$Is_SV <- rownames(train_viz) %in% rownames(model_svm_weighted$SV)
  
  p <- ggplot() +
    # 决策边界
    geom_contour(data = grid, aes(x = .data[[top2_features[1]]], 
                                  y = .data[[top2_features[2]]], 
                                  z = Decision),
                 breaks = 0.5, color = "black", size = 0.8) +
    # 数据点
    geom_point(data = train_viz, 
               aes(x = .data[[top2_features[1]]], 
                   y = .data[[top2_features[2]]], 
                   color = group, 
                   shape = Is_SV),
               alpha = 0.7, size = 2) +
    scale_color_manual(values = c("control" = "blue", "cancer" = "red")) +
    scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 4), 
                       labels = c(`FALSE` = "样本点", `TRUE` = "支持向量")) +
    labs(title = "svmLinear决策边界可视化",
         subtitle = sprintf("Top2特征: %s 和 %s (C=%.3f)", 
                            top2_features[1], top2_features[2], C_value),
         x = top2_features[1],
         y = top2_features[2],
         color = "类别",
         shape = "类型") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
  
  ggsave("./svmLinear_weighted_decision_boundary.png", p, width = 10, height = 8, dpi = 300)
  cat("决策边界图已保存: ./svmLinear_weighted_decision_boundary.png\n")
}

# ============================================================
# 保存模型
# ============================================================
save(model_svm_weighted, train_stats, feature_importance, 
     file = "./svmLinear_weighted_model.rdata")
cat("模型已保存: ./svmLinear_weighted_model.rdata\n")

# ============================================================
# 生成预测结果文件
# ============================================================
# 创建包含预测概率的完整结果
train_results <- data.frame(
  Sample_ID = rownames(train_matrix),
  Group = train_labels,
  Pred_Prob_Cancer = train_prob,
  Pred_Class = ifelse(train_prob > test_metrics$Best_Threshold, "cancer", "control"),
  Set = "Training"
)

test_results <- data.frame(
  Sample_ID = rownames(test_matrix),
  Group = test_labels,
  Pred_Prob_Cancer = test_prob,
  Pred_Class = ifelse(test_prob > test_metrics$Best_Threshold, "cancer", "control"),
  Set = "Testing"
)

all_results <- rbind(train_results, test_results)
write.csv(all_results, "./svmLinear_weighted_predictions.csv", row.names = FALSE)
cat("完整预测结果已保存: ./svmLinear_weighted_predictions.csv\n")

# ============================================================
# 输出模型摘要
# ============================================================
cat("\n========================================\n")
cat("svmLinear模型摘要\n")
cat("========================================\n")
cat(sprintf("模型类型: 线性核支持向量机 (svmLinear)\n"))
cat(sprintf("成本参数C: %.6f\n", C_value))
cat(sprintf("类别权重: 对照=%.3f, 癌症=%.3f\n", class_weights["control"], class_weights["cancer"]))
cat(sprintf("支持向量数量: %d (%.1f%%)\n", n_support_vectors, 
            100 * n_support_vectors / nrow(train_matrix)))
cat(sprintf("非零系数特征数: %d / %d\n", non_zero_coef, ncol(train_matrix)))
cat(sprintf("训练集样本数: %d\n", nrow(train_matrix)))
cat(sprintf("测试集样本数: %d\n", nrow(test_matrix)))
cat(sprintf("训练集AUC: %.3f\n", train_metrics$AUC))
cat(sprintf("测试集AUC: %.3f\n", test_metrics$AUC))

cat("\n=== svmLinear类别权重模型训练完成 ===\n")