# ============================================================
# glmnet模型：类别权重 + 阈值优化 + 5折交叉验证
# 输出训练集和测试集完整结果
# ============================================================

rm(list = ls())
library(glmnet)
library(pROC)
library(caret)

set.seed(3456)

# 加载数据（数据已经过Yeo-Johnson变换和标准化）
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
# 准备矩阵和标签（数据已经预处理，直接使用）
# ============================================================
cat("\n=== 数据已预处理 (Yeo-Johnson变换 + 标准化) ===\n")

train_matrix <- as.matrix(train_data[, feature_cols])
test_matrix <- as.matrix(test_data[, feature_cols])

train_labels <- ifelse(train_data$group == "cancer", 1, 0)
test_labels <- ifelse(test_data$group == "cancer", 1, 0)

# ============================================================
# 计算类别权重
# ============================================================
cancer_count <- sum(train_labels == 1)
control_count <- sum(train_labels == 0)

# 逆频率权重（使得少数类获得更高权重）
weight_cancer <- control_count / cancer_count
weight_control <- 1

cat(sprintf("\n癌症样本数: %d, 对照样本数: %d\n", cancer_count, control_count))
cat(sprintf("类别权重: 癌症=%.3f, 对照=%.3f\n", weight_cancer, weight_control))

# 创建权重向量
sample_weights <- ifelse(train_labels == 1, weight_cancer, weight_control)

# ============================================================
# 定义glmnet参数
# ============================================================
alpha_value <- 0.5281  # 弹性网络混合参数
lambda_value <- 0.004597  # 正则化参数

# ============================================================
# 5折交叉验证函数
# ============================================================
perform_cv <- function(X, y, weights, alpha, lambda, nfolds = 5) {
  set.seed(3456)
  
  # 创建分层折叠
  folds <- createFolds(y, k = nfolds, list = TRUE, returnTrain = FALSE)
  
  # 存储每折的结果
  cv_results <- list()
  cv_predictions <- list()
  cv_models <- list()
  
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
    
    # 训练模型
    model_fold <- glmnet(
      x = X_train_fold,
      y = y_train_fold,
      family = "binomial",
      alpha = alpha,
      lambda = lambda,
      weights = weights_train_fold,
      standardize = FALSE,  # 数据已经标准化
      intercept = TRUE,
      thresh = 1e-7,
      maxit = 1000
    )
    
    # 预测验证集
    pred_prob <- predict(model_fold, newx = X_val_fold, type = "response")[,1]
    
    # 保存结果
    cv_predictions[[i]] <- pred_prob
    cv_models[[i]] <- model_fold
    
    # 计算验证集指标
    y_val_factor <- ifelse(y_val_fold == 1, "cancer", "control")
    
    # 计算AUC
    roc_obj <- roc(y_val_fold, pred_prob, quiet = TRUE)
    auc_val <- auc(roc_obj)
    
    # 计算最佳阈值
    coords <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"), 
                     transpose = FALSE)
    best_thresh <- coords[1, "threshold"]
    
    # 分类
    pred_class <- ifelse(pred_prob > best_thresh, "cancer", "control")
    pred_class <- factor(pred_class, levels = c("control", "cancer"))
    true_class <- factor(y_val_factor, levels = c("control", "cancer"))
    
    cm <- confusionMatrix(pred_class, true_class, positive = "cancer")
    
    # 计算Brier评分
    brier_score <- mean((pred_prob - y_val_fold)^2)
    
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
      Brier_Score = brier_score,
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
    cat(sprintf("  AUC = %.3f\n", auc_val))
    cat(sprintf("  ACC = %.3f\n", acc))
    cat(sprintf("  SENS = %.3f\n", sens))
    cat(sprintf("  SPEC = %.3f\n", spec))
  }
  
  # 合并所有折的结果
  cv_results_df <- do.call(rbind, cv_results)
  
  # 计算平均值和标准差
  cv_summary <- data.frame(
    Metric = c("AUC", "Brier_Score", "ACC", "SENS", "SPEC", "PPV", "NPV", "YDI", "F1"),
    Mean = apply(cv_results_df[, c("AUC", "Brier_Score", "ACC", "SENS", "SPEC", "PPV", "NPV", "YDI", "F1")], 2, mean, na.rm = TRUE),
    SD = apply(cv_results_df[, c("AUC", "Brier_Score", "ACC", "SENS", "SPEC", "PPV", "NPV", "YDI", "F1")], 2, sd, na.rm = TRUE)
  )
  
  return(list(
    results = cv_results_df,
    summary = cv_summary,
    predictions = cv_predictions,
    models = cv_models,
    folds = folds
  ))
}

# ============================================================
# 执行5折交叉验证
# ============================================================
cat("\n========================================\n")
cat("开始5折交叉验证\n")
cat("========================================\n")

cv_output <- perform_cv(
  X = train_matrix,
  y = train_labels,
  weights = sample_weights,
  alpha = alpha_value,
  lambda = lambda_value,
  nfolds = 5
)

# 保存交叉验证结果
write.csv(cv_output$results, "./glmnet_weighted_cv_fold_results.csv", row.names = FALSE)
write.csv(cv_output$summary, "./glmnet_weighted_cv_summary.csv", row.names = FALSE)

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

model_glmnet_weighted <- glmnet(
  x = train_matrix,
  y = train_labels,
  family = "binomial",
  alpha = alpha_value,
  lambda = lambda_value,
  weights = sample_weights,
  standardize = FALSE,  # 数据已经标准化
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
  roc_obj <- roc(true_numeric, pred_probs, ci = TRUE, quiet = TRUE)
  auc_val <- auc(roc_obj)
  auc_ci <- ci.auc(roc_obj, conf.level = 0.95)
  
  # 阈值优化（使用Youden指数）
  if (optimize_threshold) {
    coords <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"), 
                     transpose = FALSE)
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
cat("最终模型评估结果\n")
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



# ============================================================
# 保存模型
# ============================================================
save(model_glmnet_weighted, cv_output, file = "./glmnet_weighted_model.rdata")
cat("模型已保存: ./glmnet_weighted_model.rdata\n")

cat("\n=== glmnet类别权重模型训练完成 ===\n")