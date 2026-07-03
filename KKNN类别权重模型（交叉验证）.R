# ============================================================
# KKNN模型：类别权重 + 阈值优化 + 5折交叉验证
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
# 数据准备（数据已经过Yeo-Johnson变换和标准化）
# ============================================================
train_features <- train_data[, feature_cols]
train_labels <- train_data$group

test_features <- test_data[, feature_cols]
test_labels <- test_data$group

# 数值标签
train_num <- ifelse(train_labels == "cancer", 1, 0)
test_num <- ifelse(test_labels == "cancer", 1, 0)

# ============================================================
# 计算类别权重
# ============================================================
cancer_count <- sum(train_labels == "cancer")
control_count <- sum(train_labels == "control")

# 逆频率权重
weight_cancer <- control_count / cancer_count
weight_control <- 1

cat(sprintf("\n癌症样本数: %d, 对照样本数: %d\n", cancer_count, control_count))
cat(sprintf("类别权重: 癌症=%.3f, 对照=%.3f\n", weight_cancer, weight_control))

# 创建权重向量
sample_weights <- ifelse(train_num == 1, weight_cancer, weight_control)

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
# 5折交叉验证函数（KKNN版本）
# ============================================================
perform_cv_kknn <- function(X, y, y_numeric, weights, k, kernel, distance, nfolds = 5) {
  set.seed(3456)
  
  # 创建分层折叠
  folds <- createFolds(y, k = nfolds, list = TRUE, returnTrain = FALSE)
  
  # 存储每折的结果
  cv_results <- list()
  cv_predictions <- list()
  
  for (i in 1:nfolds) {
    cat(sprintf("\n--- 交叉验证第 %d/%d 折 ---\n", i, nfolds))
    
    # 划分训练和验证集
    val_indices <- folds[[i]]
    train_indices <- setdiff(1:nrow(X), val_indices)
    
    X_train_fold <- X[train_indices, ]
    y_train_fold <- y[train_indices]
    weights_train_fold <- weights[train_indices]
    
    X_val_fold <- X[val_indices, ]
    y_val_fold <- y[val_indices]
    y_val_numeric <- y_numeric[val_indices]
    
    # 准备训练数据
    train_data_fold <- cbind(group = y_train_fold, X_train_fold)
    
    # 训练KKNN模型
    model_fold <- tryCatch({
      kknn(
        formula = group ~ .,
        train = train_data_fold,
        test = X_val_fold,
        k = k,
        kernel = kernel,
        distance = distance
      )
    }, error = function(e) {
      cat("KKNN训练出错:", e$message, "\n")
      return(NULL)
    })
    
    if (is.null(model_fold)) {
      cat("模型训练失败，跳过此折\n")
      next
    }
    
    # 预测概率
    pred_prob <- model_fold$prob[, "cancer"]
    
    # 保存结果
    cv_predictions[[i]] <- pred_prob
    
    # 计算验证集权重（用于加权指标）
    val_weights <- ifelse(y_val_numeric == 1, weight_cancer, weight_control)
    
    # 计算AUC
    roc_obj <- roc(y_val_numeric, pred_prob, quiet = TRUE)
    auc_val <- auc(roc_obj)
    
    # 计算最佳阈值（基于加权Youden指数）
    thresholds <- sort(unique(pred_prob))
    best_metric <- -Inf
    best_thresh <- 0.5
    
    for (th in thresholds) {
      pred_class <- ifelse(pred_prob > th, 1, 0)
      tp <- sum(val_weights[pred_class == 1 & y_val_numeric == 1])
      fn <- sum(val_weights[pred_class == 0 & y_val_numeric == 1])
      fp <- sum(val_weights[pred_class == 1 & y_val_numeric == 0])
      tn <- sum(val_weights[pred_class == 0 & y_val_numeric == 0])
      
      sens <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
      spec <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
      youden <- sens + spec - 1
      
      if (youden > best_metric) {
        best_metric <- youden
        best_thresh <- th
      }
    }
    
    # 使用最佳阈值进行分类
    pred_class <- ifelse(pred_prob > best_thresh, "cancer", "control")
    pred_class <- factor(pred_class, levels = c("control", "cancer"))
    true_class <- factor(y_val_fold, levels = c("control", "cancer"))
    
    cm <- confusionMatrix(pred_class, true_class, positive = "cancer")
    
    # 计算加权Brier评分
    brier_weighted <- sum(val_weights * (pred_prob - y_val_numeric)^2) / sum(val_weights)
    
    # 提取指标
    acc <- cm$overall["Accuracy"]
    sens <- cm$byClass["Sensitivity"]
    spec <- cm$byClass["Specificity"]
    ppv <- cm$byClass["Pos Pred Value"]
    npv <- cm$byClass["Neg Pred Value"]
    ydi <- sens + spec - 1
    f1 <- 2 * (ppv * sens) / (ppv + sens)
    
    # 存储结果
    cv_results[[i]] <- data.frame(
      Fold = i,
      AUC = auc_val,
      Brier_Score = brier_weighted,
      ACC = acc,
      SENS = sens,
      SPEC = spec,
      PPV = ppv,
      NPV = npv,
      YDI = ydi,
      F1 = f1,
      Best_Threshold = best_thresh,
      N_Val = length(val_indices)
    )
    
    cat(sprintf("  验证集样本数: %d\n", length(val_indices)))
    cat(sprintf("  验证集癌症/对照: %d/%d\n", 
                sum(y_val_numeric == 1), sum(y_val_numeric == 0)))
    cat(sprintf("  AUC = %.3f\n", auc_val))
    cat(sprintf("  ACC = %.3f\n", acc))
    cat(sprintf("  SENS = %.3f\n", sens))
    cat(sprintf("  SPEC = %.3f\n", spec))
  }
  
  # 合并所有折的结果
  if (length(cv_results) > 0) {
    cv_results_df <- do.call(rbind, cv_results)
    
    # 计算平均值和标准差
    cv_summary <- data.frame(
      Metric = c("AUC", "Brier_Score", "ACC", "SENS", "SPEC", "PPV", "NPV", "YDI", "F1"),
      Mean = apply(cv_results_df[, c("AUC", "Brier_Score", "ACC", "SENS", "SPEC", "PPV", "NPV", "YDI", "F1")], 2, mean, na.rm = TRUE),
      SD = apply(cv_results_df[, c("AUC", "Brier_Score", "ACC", "SENS", "SPEC", "PPV", "NPV", "YDI", "F1")], 2, sd, na.rm = TRUE)
    )
  } else {
    cv_results_df <- NULL
    cv_summary <- NULL
  }
  
  return(list(
    results = cv_results_df,
    summary = cv_summary,
    predictions = cv_predictions,
    folds = folds
  ))
}

# ============================================================
# 执行5折交叉验证
# ============================================================
cat("\n========================================\n")
cat("开始5折交叉验证\n")
cat("========================================\n")

cv_output <- perform_cv_kknn(
  X = train_features,
  y = train_labels,
  y_numeric = train_num,
  weights = sample_weights,
  k = k_best,
  kernel = kernel_best,
  distance = distance_best,
  nfolds = 5
)

# 保存交叉验证结果
write.csv(cv_output$results, "./kknn_best_cv_fold_results.csv", row.names = FALSE)
write.csv(cv_output$summary, "./kknn_best_cv_summary.csv", row.names = FALSE)

cat("\n========================================\n")
cat("交叉验证结果汇总 (5折平均)\n")
cat("========================================\n")
print(cv_output$summary)

# ============================================================
# 训练最终模型（使用全部训练数据）
# ============================================================
cat("\n========================================\n")
cat("训练最终模型\n")
cat("========================================\n")

train_data_full <- cbind(group = train_labels, train_features)

# 训练集预测
train_fit <- kknn(
  formula = group ~ .,
  train = train_data_full,
  test = train_features,
  k = k_best,
  kernel = kernel_best,
  distance = distance_best
)

# 测试集预测
test_fit <- kknn(
  formula = group ~ .,
  train = train_data_full,
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
# 评估函数（支持加权指标）
# ============================================================
calculate_metrics <- function(true_labels, pred_probs, weights, dataset_name, 
                              true_num = NULL, optimize_threshold = TRUE) {
  
  if (is.null(true_num)) {
    true_num <- ifelse(true_labels == "cancer", 1, 0)
  }
  
  # 计算AUC和置信区间
  roc_obj <- roc(true_num, pred_probs, ci = TRUE, quiet = TRUE)
  auc_val <- auc(roc_obj)
  auc_ci <- ci.auc(roc_obj, conf.level = 0.95)
  
  # 加权Brier评分
  brier_weighted <- sum(weights * (pred_probs - true_num)^2) / sum(weights)
  
  # 阈值优化（基于加权Youden指数）
  if (optimize_threshold) {
    thresholds <- sort(unique(pred_probs))
    best_metric <- -Inf
    best_thresh <- 0.5
    for (th in thresholds) {
      pred_class <- ifelse(pred_probs > th, 1, 0)
      tp <- sum(weights[pred_class == 1 & true_num == 1])
      fn <- sum(weights[pred_class == 0 & true_num == 1])
      fp <- sum(weights[pred_class == 1 & true_num == 0])
      tn <- sum(weights[pred_class == 0 & true_num == 0])
      
      sens <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
      spec <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
      youden <- sens + spec - 1
      
      if (youden > best_metric) {
        best_metric <- youden
        best_thresh <- th
      }
    }
  } else {
    best_thresh <- 0.5
  }
  
  # 使用最佳阈值进行分类
  pred_class <- ifelse(pred_probs > best_thresh, "cancer", "control")
  pred_class <- factor(pred_class, levels = c("control", "cancer"))
  true_class <- factor(true_labels, levels = c("control", "cancer"))
  
  cm <- confusionMatrix(pred_class, true_class, positive = "cancer")
  
  # 提取指标
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
# 计算训练集和测试集指标
# ============================================================
cat("\n========================================\n")
cat("最终模型评估结果\n")
cat("========================================\n")

train_metrics <- calculate_metrics(
  true_labels = train_labels,
  pred_probs = train_prob,
  weights = sample_weights,
  dataset_name = "Training",
  true_num = train_num
)

test_metrics <- calculate_metrics(
  true_labels = test_labels,
  pred_probs = test_prob,
  weights = rep(1, length(test_labels)),
  dataset_name = "Testing",
  true_num = test_num
)

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

cat("\n=== 5折交叉验证平均值 ===\n")
for (i in 1:nrow(cv_output$summary)) {
  cat(sprintf("  %s = %.3f (SD: %.3f)\n", 
              cv_output$summary$Metric[i],
              cv_output$summary$Mean[i],
              cv_output$summary$SD[i]))
}

cat("\n=== 最终模型性能 ===\n")

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
test_pred_class_opt <- ifelse(test_prob > test_metrics$Best_Threshold, "cancer", "control")
test_pred_class_opt <- factor(test_pred_class_opt, levels = c("control", "cancer"))
test_true_class <- factor(test_labels, levels = c("control", "cancer"))
cm_test <- confusionMatrix(test_pred_class_opt, test_true_class, positive = "cancer")
print(cm_test$table)



# ============================================================
# 保存模型和结果
# ============================================================
save(train_fit, test_fit, train_features, train_labels, cv_output, results_df,
     file = "./kknn_best_model.rdata")
cat("模型已保存: ./kknn_best_model.rdata\n")

# ============================================================
# 保存预测结果（完整）
# ============================================================
train_results <- data.frame(
  Sample_ID = rownames(train_features),
  Group = train_labels,
  Pred_Prob_Cancer = train_prob,
  Pred_Class = ifelse(train_prob > test_metrics$Best_Threshold, "cancer", "control"),
  Set = "Training"
)

test_results <- data.frame(
  Sample_ID = rownames(test_features),
  Group = test_labels,
  Pred_Prob_Cancer = test_prob,
  Pred_Class = ifelse(test_prob > test_metrics$Best_Threshold, "cancer", "control"),
  Set = "Testing"
)

all_results <- rbind(train_results, test_results)
write.csv(all_results, "./kknn_best_predictions.csv", row.names = FALSE)
cat("完整预测结果已保存: ./kknn_best_predictions.csv\n")

# ============================================================
# 输出模型摘要
# ============================================================
cat("\n========================================\n")
cat("KKNN模型摘要\n")
cat("========================================\n")
cat(sprintf("模型类型: KKNN (加权K近邻)\n"))
cat(sprintf("参数配置:\n"))
cat(sprintf("  - k (近邻数): %d\n", k_best))
cat(sprintf("  - kernel (核函数): %s\n", kernel_best))
cat(sprintf("  - distance (距离参数): %.4f\n", distance_best))
cat(sprintf("\n交叉验证 (5折) 平均性能:\n"))
for (i in 1:nrow(cv_output$summary)) {
  cat(sprintf("  - %s: %.3f (SD: %.3f)\n", 
              cv_output$summary$Metric[i],
              cv_output$summary$Mean[i],
              cv_output$summary$SD[i]))
}
cat(sprintf("\n数据信息:\n"))
cat(sprintf("  - 训练集样本数: %d\n", nrow(train_features)))
cat(sprintf("  - 测试集样本数: %d\n", nrow(test_features)))
cat(sprintf("  - 特征数量: %d\n", ncol(train_features)))
cat(sprintf("\n最终模型性能:\n"))
cat(sprintf("  - 训练集AUC: %.3f\n", train_metrics$AUC))
cat(sprintf("  - 测试集AUC: %.3f\n", test_metrics$AUC))
cat(sprintf("  - 测试集准确率: %.3f\n", test_metrics$ACC))
cat(sprintf("  - 测试集F1分数: %.3f\n", test_metrics$F1))

cat("\n输出文件:\n")
cat("  - kknn_best_results.csv (性能指标)\n")
cat("  - kknn_best_cv_fold_results.csv (CV每折结果)\n")
cat("  - kknn_best_cv_summary.csv (CV汇总)\n")
cat("  - kknn_best_predictions.csv (预测结果)\n")
cat("  - kknn_best_roc.png (ROC曲线)\n")
cat("  - kknn_best_cv_performance.png (CV性能图)\n")
cat("  - kknn_best_prob_dist.png (概率分布图)\n")
cat("  - kknn_best_model.rdata (模型文件)\n")

cat("\n=== KKNN模型训练完成 ===\n")
cat(sprintf("最佳参数: k=%d, kernel=%s, distance=%.4f\n", 
            k_best, kernel_best, distance_best))

