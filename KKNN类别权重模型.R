# ============================================================
# KKNN模型：使用最佳参数组合
# 输出训练集和测试集完整结果
# ============================================================

rm(list = ls())
library(kknn)
library(pROC)
library(caret)

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
# 数据准备
# ============================================================
# 提取特征和标签
train_features <- train_data[, feature_cols]
train_labels <- train_data$group

test_features <- test_data[, feature_cols]
test_labels <- test_data$group

# 数据已经进行了正态变换，无需额外标准化

# ============================================================
# KKNN最佳参数
# ============================================================
k_best <- 30
kernel_best <- "rectangular"  # rectangular = 均匀核
distance_best <- 1.0000

cat("\n=== KKNN参数设置 ===\n")
cat(sprintf("k: %d\n", k_best))
cat(sprintf("kernel: %s\n", kernel_best))
cat(sprintf("distance: %.4f\n", distance_best))

# ============================================================
# 训练KKNN模型
# ============================================================
cat("\n正在训练KKNN模型...\n")

# 使用train.kknn进行训练
model_kknn <- train.kknn(
  formula = group ~ .,
  data = cbind(group = train_labels, train_features),
  kmax = k_best,           # 最大k值
  kknn = TRUE,             # 使用kknn包
  kernel = kernel_best,    # 核函数类型
  distance = distance_best # 距离参数（Minkowski距离的p值）
)

# 由于train.kknn会训练多个k值，我们提取最佳k值的模型
# 或者直接使用kknn函数重新训练指定参数

# 方法2：直接使用kknn函数训练指定参数
model_kknn_fit <- kknn(
  formula = group ~ .,
  train = cbind(group = train_labels, train_features),
  test = train_features,
  k = k_best,
  kernel = kernel_best,
  distance = distance_best
)

# ============================================================
# 预测概率
# ============================================================
# 训练集预测
train_fit <- kknn(
  formula = group ~ .,
  train = cbind(group = train_labels, train_features),
  test = train_features,
  k = k_best,
  kernel = kernel_best,
  distance = distance_best
)

# 测试集预测
test_fit <- kknn(
  formula = group ~ .,
  train = cbind(group = train_labels, train_features),
  test = test_features,
  k = k_best,
  kernel = kernel_best,
  distance = distance_best
)

# 提取概率（癌症的概率）
train_prob <- train_fit$prob[, "cancer"]
test_prob <- test_fit$prob[, "cancer"]

# 获取预测类别
train_pred_class <- train_fit$fitted.values
test_pred_class <- test_fit$fitted.values

# ============================================================
# 评估函数
# ============================================================
calculate_metrics <- function(true_labels, pred_probs, pred_classes, dataset_name, 
                              optimize_threshold = TRUE) {
  
  true_numeric <- ifelse(true_labels == "cancer", 1, 0)
  
  # 计算AUC和置信区间
  roc_obj <- roc(true_numeric, pred_probs, ci = TRUE)
  auc_val <- auc(roc_obj)
  auc_ci <- ci.auc(roc_obj, conf.level = 0.95)
  
  # 阈值优化（使用Youden指数）
  if (optimize_threshold) {
    coords <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"))
    best_thresh <- coords[1, "threshold"]
    # 使用最佳阈值重新分类
    pred_class_opt <- ifelse(pred_probs > best_thresh, "cancer", "control")
    pred_class_opt <- factor(pred_class_opt, levels = c("control", "cancer"))
  } else {
    best_thresh <- 0.5
    pred_class_opt <- factor(pred_classes, levels = c("control", "cancer"))
  }
  
  true_class <- factor(true_labels, levels = c("control", "cancer"))
  
  cm <- confusionMatrix(pred_class_opt, true_class, positive = "cancer")
  
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
    Model_Type = "KKNN",
    k = k_best,
    Kernel = kernel_best,
    Distance = distance_best,
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
cat("KKNN模型评估结果\n")
cat("========================================\n")

train_metrics <- calculate_metrics(train_labels, train_prob, train_pred_class, "Training")
test_metrics <- calculate_metrics(test_labels, test_prob, test_pred_class, "Testing")

# 合并结果
results_df <- rbind(train_metrics, test_metrics)

# 打印结果
cat("\n=== 结果汇总 ===\n")
print(results_df)

# 保存结果到CSV
csv_filename <- "./kknn_best_results.csv"
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
# 混淆矩阵详细输出
# ============================================================
cat("\n=== 测试集混淆矩阵（使用最佳阈值）===\n")
true_numeric_test <- ifelse(test_labels == "cancer", 1, 0)
roc_test <- roc(true_numeric_test, test_prob)
coords_test <- coords(roc_test, "best", ret = "threshold")
best_thresh_test <- coords_test[1, "threshold"]
test_pred_class_opt <- ifelse(test_prob > best_thresh_test, "cancer", "control")
test_pred_class_opt <- factor(test_pred_class_opt, levels = c("control", "cancer"))
test_true_class <- factor(test_labels, levels = c("control", "cancer"))
cm_test <- confusionMatrix(test_pred_class_opt, test_true_class, positive = "cancer")
print(cm_test$table)

# ============================================================
# 绘制ROC曲线
# ============================================================
png("./kknn_best_roc.png", width = 8, height = 6, units = "in", res = 300)

roc_train <- roc(ifelse(train_labels == "cancer", 1, 0), train_prob)
roc_test <- roc(ifelse(test_labels == "cancer", 1, 0), test_prob)

plot(roc_train, col = "blue", lwd = 2, main = sprintf("KKNN模型ROC曲线 (k=%d, kernel=%s)", 
                                                      k_best, kernel_best))
plot(roc_test, col = "red", lwd = 2, add = TRUE)

legend("bottomright", 
       legend = c(sprintf("训练集 (AUC=%.3f)", auc(roc_train)),
                  sprintf("测试集 (AUC=%.3f)", auc(roc_test))),
       col = c("blue", "red"), lwd = 2, cex = 0.8)

dev.off()
cat("\nROC曲线已保存: ./kknn_best_roc.png\n")

# ============================================================
# 绘制训练集和测试集概率分布
# ============================================================
png("./kknn_best_prob_dist.png", width = 10, height = 5, units = "in", res = 300)

par(mfrow = c(1, 2))

# 训练集概率分布
hist(train_prob[train_labels == "control"], col = rgb(0,0,1,0.5), 
     xlim = c(0,1), breaks = 20, main = "训练集预测概率分布", 
     xlab = "癌症预测概率", ylab = "频数")
hist(train_prob[train_labels == "cancer"], col = rgb(1,0,0,0.5), 
     breaks = 20, add = TRUE)
legend("topright", legend = c("对照", "癌症"), 
       fill = c(rgb(0,0,1,0.5), rgb(1,0,0,0.5)), cex = 0.8)

# 测试集概率分布
hist(test_prob[test_labels == "control"], col = rgb(0,0,1,0.5), 
     xlim = c(0,1), breaks = 20, main = "测试集预测概率分布", 
     xlab = "癌症预测概率", ylab = "频数")
hist(test_prob[test_labels == "cancer"], col = rgb(1,0,0,0.5), 
     breaks = 20, add = TRUE)
legend("topright", legend = c("对照", "癌症"), 
       fill = c(rgb(0,0,1,0.5), rgb(1,0,0,0.5)), cex = 0.8)

dev.off()
cat("概率分布图已保存: ./kknn_best_prob_dist.png\n")

# ============================================================
# 保存模型
# ============================================================
# 保存训练好的模型对象
save(model_kknn_fit, model_kknn, train_features, train_labels, 
     file = "./kknn_best_model.rdata")
cat("模型已保存: ./kknn_best_model.rdata\n")

# 保存预测概率
pred_results <- data.frame(
  train_prob = train_prob,
  train_pred_class = train_pred_class,
  test_prob = test_prob,
  test_pred_class = test_pred_class
)
write.csv(pred_results, "./kknn_best_predictions.csv", row.names = FALSE)
cat("预测结果已保存: ./kknn_best_predictions.csv\n")

cat("\n=== KKNN模型训练完成 ===\n")
cat(sprintf("最佳参数: k=%d, kernel=%s, distance=%.4f\n", 
            k_best, kernel_best, distance_best))