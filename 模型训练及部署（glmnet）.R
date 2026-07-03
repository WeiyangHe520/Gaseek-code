rm(list = ls())
#########################################################
## glmnet模型完整解释与评估代码                        ##
## 版本：v3.0 (2024-05-25) - 集成类别权重            ##
#########################################################

# 加载必要包
library(caret)
library(glmnet)
library(pROC)
library(ggplot2)
library(dplyr)
library(patchwork)
library(shapviz)
library(fastshap)
library(corrplot)
library(rms)
library(viridis)
library(ggrepel)
library(tidyr)
library(plotROC)
set.seed(3456)

# 创建输出目录
FIG_DIR <- "glmnet_interpretation_figures/"
DATA_DIR <- "glmnet_interpretation_data/"
if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR)
if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR)

### 1. 数据加载 ###
cat("=== 1. 数据加载 ===\n")
load(file = ".left_data.rdata")

# 定义特征列
feature_cols <- c("gender", "age", "BMI", "Smoking_history", "drinking_history", 
                  "Family_history_of_cancer", "LYMPH_percentage", "MONO_percentage", 
                  "HGB", "MCV", "MCHC", "PLT", "DBIL", "IBIL", "ALB", "GLB", "ALT")

# ============================================================
# 注意：您的数据已经在这里完成了标准化！
# 以下代码假设您已经执行过：
# train_pre <- preProcess(train_data, method = c("center", "scale", "YeoJohnson"))
# train_data <- predict(train_pre, train_data)
# test_pre <- preProcess(test_data, method = c("center", "scale", "YeoJohnson"))
# test_data <- predict(test_pre, test_data)
# ============================================================

cat("\n=== 数据已标准化，直接使用 ===\n")

# 直接使用已经标准化的数据
x_train <- as.matrix(train_data[, feature_cols])
y_train <- ifelse(train_data$group == "cancer", 1, 0)
x_test <- as.matrix(test_data[, feature_cols])
y_test <- ifelse(test_data$group == "cancer", 1, 0)

cat("训练集样本数:", nrow(train_data), "\n")
cat("测试集样本数:", nrow(test_data), "\n")
cat("特征数量:", length(feature_cols), "\n")
cat(sprintf("训练集 - 癌症: %d, 对照: %d\n", sum(y_train == 1), sum(y_train == 0)))
cat(sprintf("测试集 - 癌症: %d, 对照: %d\n", sum(y_test == 1), sum(y_test == 0)))

# 验证数据已标准化（可选）
cat("\n标准化验证（前3个特征的均值≈0，标准差≈1）:\n")
for(i in 1:min(3, length(feature_cols))) {
  cat(sprintf("  %s: mean=%.6f, sd=%.6f\n", 
              feature_cols[i], 
              mean(x_train[, i]), 
              sd(x_train[, i])))
}

### 2. 计算类别权重 ###
cat("\n=== 2. 计算类别权重 ===\n")

cancer_count <- sum(y_train == 1)
control_count <- sum(y_train == 0)

# 逆频率权重
weight_cancer <- control_count / cancer_count
weight_control <- 1

cat(sprintf("癌症样本数: %d, 对照样本数: %d\n", cancer_count, control_count))
cat(sprintf("类别权重: 癌症=%.3f, 对照=%.3f\n", weight_cancer, weight_control))

# 创建权重向量
sample_weights <- ifelse(y_train == 1, weight_cancer, weight_control)

### 3. glmnet模型训练（带类别权重）###
cat("\n=== 3. glmnet模型训练（带类别权重）===\n")

# 定义glmnet参数
alpha_value <- 0.5281
lambda_value <- 0.004597

# 训练带权重的glmnet模型
# 重要：设置 standardize = FALSE，因为数据已经标准化
final_model <- glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = alpha_value,
  lambda = lambda_value,
  weights = sample_weights,
  standardize = FALSE,  # 关键：数据已标准化，不需要再次标准化
  intercept = TRUE,
  thresh = 1e-7,
  maxit = 1000
)

# 获取非零系数数量
coefficients <- coef(final_model)
non_zero_coef <- sum(coefficients[-1] != 0)
cat(sprintf("非零系数数量（不包括截距）: %d\n", non_zero_coef))

# 显示特征重要性
feature_importance <- data.frame(
  Feature = rownames(coefficients)[-1],
  Coefficient = as.vector(coefficients[-1]),
  Abs_Coefficient = abs(as.vector(coefficients[-1]))
)
feature_importance <- feature_importance[order(-feature_importance$Abs_Coefficient), ]
cat("\nTop 10 重要特征:\n")
print(head(feature_importance, 10))

# 保存特征重要性
write.csv(feature_importance, file.path(DATA_DIR, "glmnet_feature_importance.csv"), row.names = FALSE)

### 4. 模型性能评估 ###
cat("\n=== 4. 模型性能评估 ===\n")

# 预测概率
train_pred <- predict(final_model, newx = x_train, type = "response")[, 1]
test_pred <- predict(final_model, newx = x_test, type = "response")[, 1]

# 评估函数
calculate_metrics <- function(true_labels, pred_probs, dataset_name, optimize_threshold = TRUE) {
  
  true_numeric <- ifelse(true_labels == "cancer", 1, 0)
  true_factor <- factor(true_labels, levels = c("control", "cancer"))
  
  # 计算AUC和置信区间
  roc_obj <- roc(true_numeric, pred_probs, ci = TRUE)
  auc_val <- auc(roc_obj)
  auc_ci <- ci.auc(roc_obj, conf.level = 0.95)
  
  # 阈值优化
  if (optimize_threshold) {
    coords <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"))
    best_thresh <- coords[1, "threshold"]
  } else {
    best_thresh <- 0.5
  }
  
  # 分类
  pred_class <- ifelse(pred_probs > best_thresh, "cancer", "control")
  pred_class <- factor(pred_class, levels = c("control", "cancer"))
  
  cm <- confusionMatrix(pred_class, true_factor, positive = "cancer")
  
  # Brier评分
  brier_score <- mean((pred_probs - true_numeric)^2)
  
  # 各项指标
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
  
  return(list(metrics = result, roc_obj = roc_obj, best_thresh = best_thresh))
}

# 计算指标
train_result <- calculate_metrics(train_data$group, train_pred, "Training")
test_result <- calculate_metrics(test_data$group, test_pred, "Testing")

train_metrics <- train_result$metrics
test_metrics <- test_result$metrics

# 保存性能指标
metrics_df <- rbind(train_metrics, test_metrics)
write.csv(metrics_df, file.path(DATA_DIR, "glmnet_performance_metrics.csv"), row.names = FALSE)

# 打印结果
cat("\n=== 训练集性能 ===\n")
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

cat("\n=== 测试集性能 ===\n")
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

# 保存模型
save(final_model, file = file.path(DATA_DIR, "glmnet_weighted_model.rdata"))
cat(sprintf("模型已保存至: %s\n", file.path(DATA_DIR, "glmnet_weighted_model.rdata")))

# 继续后续分析...
cat("\n=== glmnet模型解释与评估完成 ===\n")




### 新增代码：阈值分析与决策区域可视化 ###

# 5.1 阈值性能曲线（类似Fig.6F）
cat("\n=== 5.1 生成阈值性能曲线（类似Fig.6F） ===\n")

# 完全重写的阈值性能计算函数 - 修正版
# 完全重写的、健壮的阈值分析函数
calculate_threshold_metrics_robust <- function(y_true, y_pred) {
  # 确保输入格式正确
  cat("\n=== 输入数据检查 ===\n")
  
  # 转换y_true为数值向量（0/1）
  if (is.factor(y_true)) {
    cat("y_true是因子，转换为数值...\n")
    y_true_numeric <- as.integer(as.character(y_true))
  } else {
    y_true_numeric <- as.integer(y_true)
  }
  
  # 确保是0/1编码
  unique_vals <- unique(y_true_numeric)
  cat("y_true唯一值:", paste(sort(unique_vals), collapse=", "), "\n")
  
  # 如果是其他编码（如1/2），转换为0/1
  if (!all(c(0, 1) %in% unique_vals)) {
    cat("重新编码为0/1...\n")
    if (min(unique_vals) == 1 && max(unique_vals) == 2) {
      # 如果是1/2编码，转换为0/1
      y_true_numeric <- y_true_numeric - 1
    } else if (min(unique_vals) == 0 && max(unique_vals) == 1) {
      # 已经是0/1，无需处理
    } else {
      # 其他情况，取第一个唯一值作为0
      y_true_numeric <- ifelse(y_true_numeric == min(unique_vals), 0, 1)
    }
  }
  
  # 检查test_pred
  cat("\ntest_pred类型:", class(test_pred)[1], "\n")
  cat("test_pred范围:", range(test_pred, na.rm=TRUE), "\n")
  
  # 如果是概率，确保在[0,1]范围内
  if (max(test_pred) > 1 || min(test_pred) < 0) {
    cat("警告：预测值不在[0,1]范围内，可能不是概率值\n")
  }
  
  thresholds <- seq(0, 1, by = 0.01)
  metrics <- data.frame(
    threshold = thresholds,
    NPV = NA,
    PPV = NA,
    accuracy = NA,
    sensitivity = NA,
    specificity = NA
  )
  
  # 打印基本统计
  cat("\n=== 数据基本统计 ===\n")
  cat(sprintf("总样本数: %d\n", length(y_true_numeric)))
  cat(sprintf("阳性数(1): %d (%.1f%%)\n", 
              sum(y_true_numeric == 1), 
              100 * mean(y_true_numeric == 1)))
  cat(sprintf("阴性数(0): %d (%.1f%%)\n", 
              sum(y_true_numeric == 0), 
              100 * mean(y_true_numeric == 0)))
  
  # 先做一个简单的阈值验证
  cat("\n=== 简单阈值验证 ===\n")
  
  # 阈值0.0时
  pred_class_0 <- ifelse(test_pred > 0, 1, 0)  # 所有预测为阳性
  cat("阈值0.0:\n")
  cat(sprintf("  预测阳性数: %d\n", sum(pred_class_0 == 1)))
  cat(sprintf("  预测阴性数: %d\n", sum(pred_class_0 == 0)))
  cat(sprintf("  实际阳性数: %d\n", sum(y_true_numeric == 1)))
  cat(sprintf("  实际阴性数: %d\n", sum(y_true_numeric == 0)))
  
  # 手动计算
  TP_simple <- sum(pred_class_0 == 1 & y_true_numeric == 1)
  FP_simple <- sum(pred_class_0 == 1 & y_true_numeric == 0)
  TN_simple <- sum(pred_class_0 == 0 & y_true_numeric == 0)
  FN_simple <- sum(pred_class_0 == 0 & y_true_numeric == 1)
  
  cat(sprintf("  TP=%d, FP=%d, TN=%d, FN=%d\n", 
              TP_simple, FP_simple, TN_simple, FN_simple))
  
  # 主循环
  for (i in seq_along(thresholds)) {
    thresh <- thresholds[i]
    pred_class <- ifelse(test_pred > thresh, 1, 0)
    
    # 直接计算，避免table()的潜在问题
    TP <- sum(pred_class == 1 & y_true_numeric == 1)
    FP <- sum(pred_class == 1 & y_true_numeric == 0)
    TN <- sum(pred_class == 0 & y_true_numeric == 0)
    FN <- sum(pred_class == 0 & y_true_numeric == 1)
    
    # 验证逻辑
    if (i == 1) {  # 阈值0.00
      cat("\n=== 详细验证 - 阈值0.00 ===\n")
      cat(sprintf("预测阳性数: %d (应等于总样本数%d)\n", 
                  sum(pred_class == 1), length(y_true_numeric)))
      cat(sprintf("预测阴性数: %d (应为0)\n", sum(pred_class == 0)))
      cat(sprintf("TP=%d (应等于实际阳性数%d)\n", TP, sum(y_true_numeric == 1)))
      cat(sprintf("FP=%d (应等于实际阴性数%d)\n", FP, sum(y_true_numeric == 0)))
      cat(sprintf("FN=%d (应为0)\n", FN))
      cat(sprintf("TN=%d (应为0)\n", TN))
    }
    
    if (i == length(thresholds)) {  # 阈值1.00
      cat("\n=== 详细验证 - 阈值1.00 ===\n")
      cat(sprintf("预测阳性数: %d (应为0)\n", sum(pred_class == 1)))
      cat(sprintf("预测阴性数: %d (应等于总样本数%d)\n", 
                  sum(pred_class == 0), length(y_true_numeric)))
      cat(sprintf("TP=%d (应为0)\n", TP))
      cat(sprintf("FP=%d (应为0)\n", FP))
      cat(sprintf("FN=%d (应等于实际阳性数%d)\n", FN, sum(y_true_numeric == 1)))
      cat(sprintf("TN=%d (应等于实际阴性数%d)\n", TN, sum(y_true_numeric == 0)))
    }
    
    # 计算指标
    metrics$PPV[i] <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
    metrics$NPV[i] <- ifelse((TN + FN) > 0, TN / (TN + FN), NA)
    metrics$accuracy[i] <- (TP + TN) / length(y_true_numeric)
    metrics$sensitivity[i] <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
    metrics$specificity[i] <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
  }
  
  # 理论验证
  cat("\n=== 理论值验证 ===\n")
  prevalence <- mean(y_true_numeric == 1)
  cat(sprintf("患病率 (阳性比例): %.3f\n", prevalence))
  cat(sprintf("阈值0.00 PPV理论值: %.3f (等于患病率)\n", prevalence))
  cat(sprintf("阈值0.00 PPV实际值: %.3f\n", metrics$PPV[1]))
  cat(sprintf("阈值1.00 NPV理论值: %.3f (等于1-患病率)\n", 1 - prevalence))
  cat(sprintf("阈值1.00 NPV实际值: %.3f\n", tail(metrics$NPV, 1)))
  
  # 正确的趋势
  cat("\n=== 期望的趋势 ===\n")
  cat("1. PPV: 从患病率(阈值0.00)逐渐增加到接近1(阈值1.00)\n")
  cat("2. NPV: 从接近0(低阈值)上升到峰值，然后下降到实际阴性比例(阈值1.00) - 先升后降\n")
  cat("3. 准确率: 呈钟形曲线，在最优阈值处最高\n")
  
  return(metrics)
}

# 运行这个版本
cat("\n")
cat(rep("=", 60), collapse = "")
cat("\n运行健壮版本的阈值分析\n")
cat(rep("=", 60), collapse = "")
cat("\n\n")

test_metrics_df_robust <- calculate_threshold_metrics_robust(y_test, test_pred)


# 现在test_metrics_df_robust是正确的
test_metrics_df <- test_metrics_df_robust

# 5.1 寻找关键阈值
cat("\n=== 5.1 寻找关键阈值 ===\n")

find_optimal_threshold <- function(metrics_df, target_ppv = 0.9, target_npv = 0.9) {
  results <- list()
  
  # 1. 最大准确率阈值
  max_acc_idx <- which.max(metrics_df$accuracy)
  results$max_acc <- list(
    threshold = metrics_df$threshold[max_acc_idx],
    value = metrics_df$accuracy[max_acc_idx],
    sensitivity = metrics_df$sensitivity[max_acc_idx],
    specificity = metrics_df$specificity[max_acc_idx]
  )
  
  # 2. 高PPV阈值（>= target_ppv）
  ppv_above_target <- metrics_df$PPV >= target_ppv
  if (any(ppv_above_target, na.rm = TRUE)) {
    high_ppv_idx <- min(which(ppv_above_target))
    results$high_ppv <- list(
      threshold = metrics_df$threshold[high_ppv_idx],
      value = metrics_df$PPV[high_ppv_idx],
      sensitivity = metrics_df$sensitivity[high_ppv_idx],
      specificity = metrics_df$specificity[high_ppv_idx]
    )
  } else {
    # 如果没有达到target_ppv，取最大值
    high_ppv_idx <- which.max(metrics_df$PPV)
    results$high_ppv <- list(
      threshold = metrics_df$threshold[high_ppv_idx],
      value = metrics_df$PPV[high_ppv_idx],
      sensitivity = metrics_df$sensitivity[high_ppv_idx],
      specificity = metrics_df$specificity[high_ppv_idx]
    )
  }
  
  # 3. 高NPV阈值（>= target_npv）
  npv_above_target <- metrics_df$NPV >= target_npv
  if (any(npv_above_target, na.rm = TRUE)) {
    high_npv_idx <- max(which(npv_above_target))
    results$high_npv <- list(
      threshold = metrics_df$threshold[high_npv_idx],
      value = metrics_df$NPV[high_npv_idx],
      sensitivity = metrics_df$sensitivity[high_npv_idx],
      specificity = metrics_df$specificity[high_npv_idx]
    )
  } else {
    # 如果没有达到target_npv，取最大值
    high_npv_idx <- which.max(metrics_df$NPV)
    results$high_npv <- list(
      threshold = metrics_df$threshold[high_npv_idx],
      value = metrics_df$NPV[high_npv_idx],
      sensitivity = metrics_df$sensitivity[high_npv_idx],
      specificity = metrics_df$specificity[high_npv_idx]
    )
  }
  
  return(results)
}

# 计算关键阈值
threshold_results <- find_optimal_threshold(test_metrics_df)

cat("\n关键阈值:\n")
cat(sprintf("  最大准确率阈值: %.3f\n", threshold_results$max_acc$threshold))
cat(sprintf("    准确率: %.3f, 敏感度: %.3f, 特异度: %.3f\n", 
            threshold_results$max_acc$value,
            threshold_results$max_acc$sensitivity,
            threshold_results$max_acc$specificity))

cat(sprintf("\n  高PPV阈值 (>=95%%): %.3f\n", threshold_results$high_ppv$threshold))
cat(sprintf("    PPV: %.3f, 敏感度: %.3f, 特异度: %.3f\n", 
            threshold_results$high_ppv$value,
            threshold_results$high_ppv$sensitivity,
            threshold_results$high_ppv$specificity))

cat(sprintf("\n  高NPV阈值 (>=95%%): %.3f\n", threshold_results$high_npv$threshold))
cat(sprintf("    NPV: %.3f, 敏感度: %.3f, 特异度: %.3f\n", 
            threshold_results$high_npv$value,
            threshold_results$high_npv$sensitivity,
            threshold_results$high_npv$specificity))

# 5.2 绘制阈值性能曲线
cat("\n=== 5.2 绘制阈值性能曲线 ===\n")

library(tidyr)
library(ggplot2)
library(ggrepel)

# 准备数据
threshold_data_long <- test_metrics_df %>%
  select(threshold, NPV, PPV, accuracy, sensitivity, specificity) %>%
  pivot_longer(cols = -threshold, names_to = "metric", values_to = "value") %>%
  mutate(
    metric = factor(metric,
                    levels = c("accuracy", "sensitivity", "specificity", "PPV", "NPV"),
                    labels = c("accuracy", "sensitivity", "specificity", "PPV", "NPV"))
  )

# 创建标注数据
annotation_data <- data.frame(
  metric = factor(c("accuracy", "PPV", "NPV"), 
                  levels = c("accuracy", "PPV", "NPV")),
  x = c(threshold_results$max_acc$threshold,
        threshold_results$high_ppv$threshold,
        threshold_results$high_npv$threshold),
  y = c(threshold_results$max_acc$value,
        threshold_results$high_ppv$value,
        threshold_results$high_npv$value),
  label = c(
    sprintf("accuracy=%.3f\nCutoff=%.3f", 
            threshold_results$max_acc$value, 
            threshold_results$max_acc$threshold),
    sprintf("PPV=%.3f\nCutoff=%.3f", 
            threshold_results$high_ppv$value, 
            threshold_results$high_ppv$threshold),
    sprintf("NPV=%.3f\nCutoff=%.3f", 
            threshold_results$high_npv$value, 
            threshold_results$high_npv$threshold)
  )
)

# 定义颜色
metric_colors <- c(
  "Accuracy" = "#4daf4a",
  "Sensitivity" = "#984ea3", 
  "Specificity" = "#e41a1c",
  "PPV" = "#e41a1c",
  "NPV" = "#377eb8"
)

# 绘制阈值性能曲线
p_threshold <- ggplot(threshold_data_long %>% 
                        filter(metric %in% c("accuracy", "PPV", "NPV")), 
                      aes(x = threshold, y = value, color = metric)) +
  geom_line(size = 1.2, alpha = 0.8) +
  
  # 添加阈值线
  geom_vline(xintercept = threshold_results$max_acc$threshold, 
             linetype = "dashed", color = "gray80", size = 1.2, alpha = 0.8) +
  geom_vline(xintercept = threshold_results$high_ppv$threshold, 
             linetype = "dashed", color = "#e41a1c", size = 1.2, alpha = 0.8) +
  geom_vline(xintercept = threshold_results$high_npv$threshold, 
             linetype = "dashed", color = "#377eb8", size = 1.2, alpha = 0.8) +
  
  # 添加标注点
  geom_point(data = annotation_data, aes(x = x, y = y), size = 3) +
  geom_text_repel(
    data = annotation_data,
    aes(x = x, y = y, label = label),
    box.padding = 0.5,
    point.padding = 0.3,
    size = 6,
    segment.color = "gray50",
    max.overlaps = 20,
    nudge_x = c(0, 0, 0),      # accuracy不动，PPV右移，NPV左移
    nudge_y = c(-0.28, -0.6, -0.4)    # accuracy上移，PPV上移，NPV下移
  ) +
  
  scale_color_manual(values = metric_colors) +
  
  labs(
    title = "Cutoff selection in the discovery set 
   ",
    x = "Cutoff",
    y = "Predictive score",
    color = ""
  ) +
  
  theme_minimal(base_size = 18) +
  theme(
    legend.position = "none",
    axis.text = element_text(size = 18),      # 坐标轴刻度标签
    axis.title = element_text(size = 18),     # 坐标轴标题
    axis.text.x = element_text(size = 17),    # X轴刻度（单独设置）
    axis.text.y = element_text(size = 17),    # Y轴刻度（单独设置）
    axis.title.x = element_text(size = 17, face = "bold"),   # X轴标题
    axis.title.y = element_text(size = 17, face = "bold"),   # Y轴标题
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 0.8),  # 添加黑色边框
    plot.title = element_text(size = 19,hjust = 0.5, face = "bold"),
  ) +
  
  scale_y_continuous(
    limits = c(0, 1), 
    breaks = seq(0, 1, 0.2),
    labels = scales::percent_format(accuracy = 1)
  ) +
  
  scale_x_continuous(
    limits = c(0, 1), 
    breaks = seq(0, 1, 0.1)
  ) +
  
  # 添加网格线强调
  annotate("segment", x = 0, xend = 1, y = 0.5, yend = 0.5, 
           color = "gray80", linetype = "dotted") +
  annotate("segment", x = 0.5, xend = 0.5, y = 0, yend = 1, 
           color = "gray80", linetype = "dotted")

print(p_threshold)

# 保存图片
if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR, recursive = TRUE)
ggsave(file.path(FIG_DIR, "threshold_performance_curve.tiff"), 
       p_threshold, 
       width = 7, 
       height = 5.1,
       dpi = 300,          # 设置分辨率
       device = "tiff",    # 指定设备为TIFF
       compression = "lzw") # 压缩选项（可选）
cat(sprintf("\n阈值性能曲线已保存至: %s\n", file.path(FIG_DIR, "threshold_performance_curve.Tiff")))



# 5.4 输出总结报告
cat("\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("阈值分析总结报告\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat(sprintf("\n1. 总体统计:\n"))
cat(sprintf("   总样本数: %d\n", length(y_test)))
cat(sprintf("   阳性样本(malignant): %d (%.1f%%)\n", 
            sum(y_test == 1), 100 * mean(y_test == 1)))
cat(sprintf("   阴性样本(benign): %d (%.1f%%)\n", 
            sum(y_test == 0), 100 * mean(y_test == 0)))

cat(sprintf("\n2. 关键阈值:\n"))
cat(sprintf("   • 最大准确率阈值: %.3f (准确率: %.1f%%)\n", 
            threshold_results$max_acc$threshold, 
            threshold_results$max_acc$value * 100))
cat(sprintf("   • 高PPV阈值 (>=95%%): %.3f (PPV: %.1f%%)\n", 
            threshold_results$high_ppv$threshold, 
            threshold_results$high_ppv$value * 100))
cat(sprintf("   • 高NPV阈值 (>=95%%): %.3f (NPV: %.1f%%)\n", 
            threshold_results$high_npv$threshold, 
            threshold_results$high_npv$value * 100))






### 8. 决策曲线分析 - 使用rmda包 ###
cat("\n=== 8. 决策曲线分析 ===\n")

# 尝试安装和使用rmda包
if (!requireNamespace("rmda", quietly = TRUE)) {
  cat("正在安装rmda包...\n")
  install.packages("rmda")
}

library(rmda)

# 准备DCA数据
dca_data <- data.frame(
  outcome = y_test,
  prediction = test_pred
)

# 使用rmda进行决策曲线分析
tryCatch({
  # 构建决策曲线分析模型
  dca_model <- decision_curve(
    outcome ~ prediction,
    data = dca_data,
    family = binomial(link = "logit"),
    thresholds = seq(0, 1, by = 0.01),
    confidence.intervals = 0.9,
    bootstraps = 50  # 可根据计算能力调整
  )
  
  tiff(file.path(FIG_DIR, "glmnet_decision_curve.tiff"), 
    width=6, height =5, units="in", res = 300, compression = "lzw")
  # 设置全局绘图参数 - 必须在plot_decision_curve之前
  par(cex.lab= 1.6,      # 坐标轴标签字体大小
    cex.axis= 1.6,     # 坐标轴刻度字体大小
    cex.main= 1.6,     # 主标题字体大小
    mgp = c(2.5, 1, 0))
  plot_decision_curve(
    dca_model,
    curve.names = "glmnet",
    cost.benefit.axis = FALSE,
    standardize = FALSE,
    confidence.intervals = TRUE,
    col = c("#4daf4a", "#e41a1c", "#377eb8"),
    lty = c(2, 2, 2),
    lwd = c(3, 3, 3),
    xlim = c(0, 1),
    ylim = c(-0.05, 0.6), 
    legend.position= "none", 
    xlab= "High risk threshold",
    ylab= "Net benefit", 
    font.lab = 2 # y轴标签
  )
  legend("topright",
         legend = c("Gaseek", "All", "None"),
         col = c("#4daf4a", "#e41a1c", "#377eb8"),
         lty = c(2, 2, 2),
         lwd = c(3, 3, 3),
         bty = "n",      # 无边框
         cex = 1.6)      # 完全控制图例字体大小
  title("Decision Curve", cex.main = 1.7)
  dev.off()
  
  cat("决策曲线分析完成\n")
  
  # 输出DCA结果摘要
  cat("\n决策曲线分析结果摘要:\n")
  print(summary(dca_model))
  
  # 保存DCA数据
  write.csv(dca_model$derived.data, 
            file.path(DATA_DIR, "decision_curve_analysis.csv"),
            row.names = FALSE)
  
}, error = function(e) {
  cat("rmda包决策曲线分析失败:", e$message, "\n")
  cat("尝试备用方法...\n")
})


### 14. 打印总结报告 ###
cat("\n", strrep("=", 60), "\n", sep = "")
cat("GLMNET模型解释与评估完成！\n")
cat(strrep("=", 60), "\n\n", sep = "")
cat("模型性能总结:\n")

