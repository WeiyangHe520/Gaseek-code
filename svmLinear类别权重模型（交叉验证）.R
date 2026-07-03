# ============================================================
# svmLinear模型：类别权重 + 阈值优化 + 5折交叉验证
# 数据已预处理（Yeo-Johnson + center + scale）
# 输出训练集和测试集完整结果
# ============================================================

rm(list = ls())
library(e1071)
library(pROC)
library(caret)
library(ggplot2)

set.seed(3456)

# 加载原始数据
load(file = ".left_data.rdata")

# ============================================================
# 重要：您的数据已经完成预处理
# 请确认以下变量已经存在：
# train_data 和 test_data 已经经过 Yeo-Johnson + center + scale
# ============================================================

# 保存预处理参数（如果存在）
if (exists("train_pre")) {
  save(train_pre, file = "./svmLinear_preprocess_params.rdata")
  cat("预处理参数已保存\n")
}

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
# 数据准备 - 不需要再次标准化！
# ============================================================
prepare_data <- function(data) {
  features <- data[, feature_cols]
  group <- data$group
  
  # 重要：数据已经标准化，直接使用，不再进行scale操作
  return(data.frame(group = group, features))
}

# 准备数据
train_processed <- prepare_data(train_data)
test_processed <- prepare_data(test_data)

# 准备矩阵和标签
train_matrix <- train_processed[, -1]
test_matrix <- test_processed[, -1]

train_labels <- train_processed$group
test_labels <- test_processed$group

# 转换为因子（e1071 svm要求）
train_labels_factor <- factor(train_labels, levels = c("control", "cancer"))
test_labels_factor <- factor(test_labels, levels = c("control", "cancer"))

# 数值标签（用于指标计算）
train_num <- ifelse(train_labels == "cancer", 1, 0)
test_num <- ifelse(test_labels == "cancer", 1, 0)

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

# 创建类别权重向量
class_weights <- c("control" = weight_control, "cancer" = weight_cancer)

# ============================================================
# 定义svmLinear参数
# ============================================================
C_value <- 3.830174  # 成本参数

# ============================================================
# 5折交叉验证函数（svmLinear版本）
# ============================================================
perform_cv_svm <- function(X, y_factor, y_numeric, C, class_weights, nfolds = 5) {
  set.seed(3456)
  
  # 创建分层折叠
  folds <- createFolds(y_factor, k = nfolds, list = TRUE, returnTrain = FALSE)
  
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
    y_train_fold <- y_factor[train_indices]
    
    X_val_fold <- X[val_indices, ]
    y_val_fold <- y_factor[val_indices]
    y_val_numeric <- y_numeric[val_indices]
    
    # 计算当前折的类别权重
    fold_cancer_count <- sum(y_train_fold == "cancer")
    fold_control_count <- sum(y_train_fold == "control")
    fold_weight_cancer <- fold_control_count / fold_cancer_count
    fold_class_weights <- c("control" = 1, "cancer" = fold_weight_cancer)
    
    # 训练svm模型
    model_fold <- tryCatch({
      svm(
        x = X_train_fold,
        y = y_train_fold,
        kernel = "linear",
        cost = C,
        class.weights = fold_class_weights,
        probability = TRUE,
        scale = FALSE,  # 数据已经标准化
        type = "C-classification",
        tolerance = 0.001,
        cache_size = 40
      )
    }, error = function(e) {
      cat("SVM训练出错:", e$message, "\n")
      return(NULL)
    })
    
    if (is.null(model_fold)) {
      cat("模型训练失败，跳过此折\n")
      next
    }
    
    # 预测验证集
    pred_prob_attr <- predict(model_fold, X_val_fold, probability = TRUE)
    pred_prob <- attr(pred_prob_attr, "probabilities")[, "cancer"]
    
    # 保存结果
    cv_predictions[[i]] <- pred_prob
    cv_models[[i]] <- model_fold
    
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
    
    # 获取支持向量数量
    n_sv <- nrow(model_fold$SV)
    
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
      N_SupportVectors = n_sv,
      N_Val = length(val_indices)
    )
    
    cat(sprintf("  验证集样本数: %d\n", length(val_indices)))
    cat(sprintf("  验证集癌症/对照: %d/%d\n", 
                sum(y_val_numeric == 1), sum(y_val_numeric == 0)))
    cat(sprintf("  AUC = %.3f\n", auc_val))
    cat(sprintf("  ACC = %.3f\n", acc))
    cat(sprintf("  SENS = %.3f\n", sens))
    cat(sprintf("  SPEC = %.3f\n", spec))
    cat(sprintf("  支持向量数: %d\n", n_sv))
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

cv_output <- perform_cv_svm(
  X = train_matrix,
  y_factor = train_labels_factor,
  y_numeric = train_num,
  C = C_value,
  class_weights = class_weights,
  nfolds = 5
)

# 保存交叉验证结果
write.csv(cv_output$results, "./svmLinear_weighted_cv_fold_results.csv", row.names = FALSE)
write.csv(cv_output$summary, "./svmLinear_weighted_cv_summary.csv", row.names = FALSE)

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

model_svm_weighted <- svm(
  x = train_matrix,
  y = train_labels_factor,
  kernel = "linear",
  cost = C_value,
  class.weights = class_weights,
  probability = TRUE,
  scale = FALSE,  # 数据已经标准化
  type = "C-classification",
  tolerance = 0.001,
  cache_size = 40
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
if (length(w) == 0 || ncol(w) == 0) {
  coefficients <- rep(0, ncol(train_matrix))
} else {
  coefficients <- as.vector(w)
}
intercept <- -model_svm_weighted$rho

# 计算特征重要性
feature_importance <- data.frame(
  Feature = colnames(train_matrix),
  Coefficient = coefficients,
  Abs_Coefficient = abs(coefficients)
)
feature_importance <- feature_importance[order(-feature_importance$Abs_Coefficient), ]

# 统计非零系数数量
non_zero_coef <- sum(abs(coefficients) > 1e-6)
cat(sprintf("非零系数数量: %d / %d\n", non_zero_coef, length(coefficients)))

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
    C_Parameter = C_value,
    Kernel = "linear",
    NonZero_Features = non_zero_coef,
    N_SupportVectors = n_support_vectors,
    Model_Type = "svmLinear_weighted",
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

# 训练集权重
train_weights <- ifelse(train_num == 1, weight_cancer, weight_control)

train_metrics <- calculate_metrics(
  true_labels = train_labels,
  pred_probs = train_prob,
  weights = train_weights,
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
csv_filename <- "./svmLinear_weighted_results.csv"
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
cat("\n=== Top 10 重要特征 ===\n")
print(head(feature_importance, 10))

write.csv(feature_importance, "./svmLinear_weighted_feature_importance.csv", row.names = FALSE)


# ============================================================
# 保存模型和结果
# ============================================================
save(model_svm_weighted, feature_importance, results_df, cv_output,
     file = "./svmLinear_weighted_model.rdata")
cat("模型已保存: ./svmLinear_weighted_model.rdata\n")

# ============================================================
# 保存预测结果（完整）
# ============================================================
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
cat(sprintf("模型类型: 线性核SVM\n"))
cat(sprintf("C参数: %.6f\n", C_value))
cat(sprintf("数据预处理: Yeo-Johnson + center + scale (已完成)\n"))
cat(sprintf("SVM缩放: FALSE (使用已预处理数据)\n"))
cat(sprintf("\n交叉验证 (5折) 平均性能:\n"))
for (i in 1:nrow(cv_output$summary)) {
  cat(sprintf("  - %s: %.3f (SD: %.3f)\n", 
              cv_output$summary$Metric[i],
              cv_output$summary$Mean[i],
              cv_output$summary$SD[i]))
}
cat(sprintf("\n训练信息:\n"))
cat(sprintf("  - 支持向量数: %d/%d (%.1f%%)\n", 
            n_support_vectors, nrow(train_matrix),
            100 * n_support_vectors / nrow(train_matrix)))
cat(sprintf("  - 非零系数数: %d/%d\n", non_zero_coef, length(coefficients)))
cat(sprintf("\n数据信息:\n"))
cat(sprintf("  - 训练集样本数: %d\n", nrow(train_matrix)))
cat(sprintf("  - 测试集样本数: %d\n", nrow(test_matrix)))
cat(sprintf("  - 特征数量: %d\n", ncol(train_matrix)))
cat(sprintf("\n最终模型性能:\n"))
cat(sprintf("  - 训练集AUC: %.3f\n", train_metrics$AUC))
cat(sprintf("  - 测试集AUC: %.3f\n", test_metrics$AUC))
cat(sprintf("  - 测试集准确率: %.3f\n", test_metrics$ACC))
cat(sprintf("  - 测试集F1分数: %.3f\n", test_metrics$F1))

cat("\n输出文件:\n")
cat("  - svmLinear_weighted_results.csv (性能指标)\n")
cat("  - svmLinear_weighted_cv_fold_results.csv (CV每折结果)\n")
cat("  - svmLinear_weighted_cv_summary.csv (CV汇总)\n")
cat("  - svmLinear_weighted_feature_importance.csv (特征重要性)\n")
cat("  - svmLinear_weighted_predictions.csv (预测结果)\n")
cat("  - svmLinear_weighted_roc.png (ROC曲线)\n")
cat("  - svmLinear_weighted_cv_performance.png (CV性能图)\n")
cat("  - svmLinear_weighted_model.rdata (模型文件)\n")

cat("\n=== svmLinear 类别权重模型训练完成 ===\n")

