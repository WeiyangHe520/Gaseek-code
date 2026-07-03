# ============================================================
# 随机森林模型：类别权重 + 阈值优化 + 5折交叉验证
# 输出训练集和测试集完整结果
# ============================================================

rm(list = ls())
library(randomForest)
library(pROC)
library(caret)
library(ggplot2)
library(dplyr)

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

train_matrix <- train_data[, feature_cols]
test_matrix <- test_data[, feature_cols]

train_labels <- train_data$group
test_labels <- test_data$group

# 转换为因子（randomForest要求）
train_labels_factor <- factor(train_labels, levels = c("control", "cancer"))
test_labels_factor <- factor(test_labels, levels = c("control", "cancer"))

# 数值型标签（用于计算指标）
train_labels_numeric <- ifelse(train_labels == "cancer", 1, 0)
test_labels_numeric <- ifelse(test_labels == "cancer", 1, 0)

# ============================================================
# 计算类别权重（用于平衡采样）
# ============================================================
cancer_count <- sum(train_labels == "cancer")
control_count <- sum(train_labels == "control")

# 逆频率权重
weight_cancer <- control_count / cancer_count
weight_control <- 1

cat(sprintf("\n癌症样本数: %d, 对照样本数: %d\n", cancer_count, control_count))
cat(sprintf("类别权重: 癌症=%.3f, 对照=%.3f\n", weight_cancer, weight_control))

# 创建平衡采样大小
if (cancer_count < control_count) {
  sampsize_balanced <- c(cancer = cancer_count, control = cancer_count)
} else {
  sampsize_balanced <- c(cancer = control_count, control = control_count)
}
cat(sprintf("平衡采样大小: 每类 %d 个样本\n", sampsize_balanced[1]))

# ============================================================
# 定义RF最佳参数（通过贝叶斯优化得到）
# ============================================================
mtry_value <- 4          # 每个节点随机选择的特征数
ntree_value <- 300       # 树的数量
nodesize_value <- 4      # 叶子节点最小样本数

cat(sprintf("\n模型参数:\n"))
cat(sprintf("  mtry: %d\n", mtry_value))
cat(sprintf("  ntree: %d\n", ntree_value))
cat(sprintf("  nodesize: %d\n", nodesize_value))
cat(sprintf("  sampsize: %d per class\n", sampsize_balanced[1]))

# ============================================================
# 5折交叉验证函数（随机森林版本）
# ============================================================
perform_cv_rf <- function(X, y_factor, y_numeric, mtry, ntree, nodesize, sampsize, nfolds = 5) {
  set.seed(3456)
  
  # 创建分层折叠（基于因子变量）
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
    
    # 计算当前折的类别权重（用于平衡采样）
    fold_cancer_count <- sum(y_train_fold == "cancer")
    fold_control_count <- sum(y_train_fold == "control")
    
    if (fold_cancer_count < fold_control_count) {
      fold_sampsize <- c(cancer = fold_cancer_count, control = fold_cancer_count)
    } else {
      fold_sampsize <- c(cancer = fold_control_count, control = fold_control_count)
    }
    
    # 训练随机森林模型
    model_fold <- randomForest(
      x = X_train_fold,
      y = y_train_fold,
      mtry = mtry,
      ntree = ntree,
      nodesize = nodesize,
      sampsize = fold_sampsize,  # 平衡采样处理类别不平衡
      importance = FALSE,         # 不计算重要性（加快速度）
      proximity = FALSE,
      keep.forest = TRUE,
      keep.inbag = FALSE
    )
    
    # 预测验证集
    pred_prob <- predict(model_fold, newdata = X_val_fold, type = "prob")[, "cancer"]
    
    # 保存结果
    cv_predictions[[i]] <- pred_prob
    cv_models[[i]] <- model_fold
    
    # 计算验证集指标
    # 计算AUC
    roc_obj <- roc(y_val_numeric, pred_prob, quiet = TRUE)
    auc_val <- auc(roc_obj)
    
    # 计算最佳阈值（使用Youden指数）
    coords <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"), 
                     transpose = FALSE)
    best_thresh <- coords[1, "threshold"]
    
    # 分类
    pred_class <- ifelse(pred_prob > best_thresh, "cancer", "control")
    pred_class <- factor(pred_class, levels = c("control", "cancer"))
    true_class <- factor(y_val_fold, levels = c("control", "cancer"))
    
    cm <- confusionMatrix(pred_class, true_class, positive = "cancer")
    
    # 计算Brier评分
    brier_score <- mean((pred_prob - y_val_numeric)^2)
    
    # 提取指标
    acc <- cm$overall["Accuracy"]
    sens <- cm$byClass["Sensitivity"]
    spec <- cm$byClass["Specificity"]
    ppv <- cm$byClass["Pos Pred Value"]
    npv <- cm$byClass["Neg Pred Value"]
    ydi <- sens + spec - 1
    f1 <- 2 * (ppv * sens) / (ppv + sens)
    
    # 获取OOB误差
    oob_error <- model_fold$err.rate[ntree, "OOB"]
    
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
      OOB_Error = oob_error,
      N_Val = length(val_indices)
    )
    
    cat(sprintf("  验证集样本数: %d\n", length(val_indices)))
    cat(sprintf("  训练集癌症/对照: %d/%d\n", fold_cancer_count, fold_control_count))
    cat(sprintf("  AUC = %.3f\n", auc_val))
    cat(sprintf("  ACC = %.3f\n", acc))
    cat(sprintf("  SENS = %.3f\n", sens))
    cat(sprintf("  SPEC = %.3f\n", spec))
    cat(sprintf("  OOB误差 = %.4f\n", oob_error))
  }
  
  # 合并所有折的结果
  cv_results_df <- do.call(rbind, cv_results)
  
  # 计算平均值和标准差
  cv_summary <- data.frame(
    Metric = c("AUC", "Brier_Score", "ACC", "SENS", "SPEC", "PPV", "NPV", "YDI", "F1", "OOB_Error"),
    Mean = apply(cv_results_df[, c("AUC", "Brier_Score", "ACC", "SENS", "SPEC", "PPV", "NPV", "YDI", "F1", "OOB_Error")], 2, mean, na.rm = TRUE),
    SD = apply(cv_results_df[, c("AUC", "Brier_Score", "ACC", "SENS", "SPEC", "PPV", "NPV", "YDI", "F1", "OOB_Error")], 2, sd, na.rm = TRUE)
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

cv_output <- perform_cv_rf(
  X = train_matrix,
  y_factor = train_labels_factor,
  y_numeric = train_labels_numeric,
  mtry = mtry_value,
  ntree = ntree_value,
  nodesize = nodesize_value,
  sampsize = sampsize_balanced,
  nfolds = 5
)

# 保存交叉验证结果
write.csv(cv_output$results, "./randomForest_weighted_cv_fold_results.csv", row.names = FALSE)
write.csv(cv_output$summary, "./randomForest_weighted_cv_summary.csv", row.names = FALSE)

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

model_rf_weighted <- randomForest(
  x = train_matrix,
  y = train_labels_factor,
  mtry = mtry_value,
  ntree = ntree_value,
  nodesize = nodesize_value,
  sampsize = sampsize_balanced,  # 平衡采样处理类别不平衡
  importance = TRUE,              # 计算特征重要性
  proximity = FALSE,              # 不计算邻近矩阵（节省内存）
  keep.forest = TRUE,
  keep.inbag = FALSE
)

# ============================================================
# 模型摘要信息
# ============================================================
cat("\n模型训练完成！\n")
cat(sprintf("OOB误差率: %.4f\n", model_rf_weighted$err.rate[ntree_value, "OOB"]))
cat(sprintf("OOB误差率（癌症）: %.4f\n", model_rf_weighted$err.rate[ntree_value, "cancer"]))
cat(sprintf("OOB误差率（对照）: %.4f\n", model_rf_weighted$err.rate[ntree_value, "control"]))

# ============================================================
# 预测概率
# ============================================================
cat("\n预测概率...\n")
# 训练集预测
train_prob <- predict(model_rf_weighted, train_matrix, type = "prob")[, "cancer"]
# 测试集预测
test_prob <- predict(model_rf_weighted, test_matrix, type = "prob")[, "cancer"]

# 检查预测概率范围
cat(sprintf("\n训练集预测概率范围: [%.4f, %.4f]\n", min(train_prob), max(train_prob)))
cat(sprintf("测试集预测概率范围: [%.4f, %.4f]\n", min(test_prob), max(test_prob)))

# ============================================================
# 获取特征重要性
# ============================================================
# 两种重要性指标
importance_gini <- importance(model_rf_weighted, type = 1)      # 基于基尼系数
importance_accuracy <- importance(model_rf_weighted, type = 2)  # 基于准确率

feature_importance <- data.frame(
  Feature = rownames(importance_gini),
  MeanDecreaseGini = importance_gini[, 1],
  MeanDecreaseAccuracy = importance_accuracy[, 1]
) %>%
  arrange(desc(MeanDecreaseGini))

# 统计重要特征数量
non_zero_features <- sum(feature_importance$MeanDecreaseGini > 0)
cat(sprintf("\n有效特征数量（Gini > 0）: %d / %d\n", non_zero_features, nrow(feature_importance)))

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
    mtry = mtry_value,
    ntree = ntree_value,
    nodesize = nodesize_value,
    sampsize = sampsize_balanced[1],
    OOB_Error = model_rf_weighted$err.rate[ntree_value, "OOB"],
    Model_Type = "RandomForest_weighted",
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

train_metrics <- calculate_metrics(train_labels, train_prob, "Training")
test_metrics <- calculate_metrics(test_labels, test_prob, "Testing")

# 合并结果
results_df <- rbind(train_metrics, test_metrics)

# 打印结果
cat("\n=== 结果汇总 ===\n")
print(results_df)

# 保存结果到CSV
csv_filename <- "./randomForest_weighted_results.csv"
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
# 特征重要性（Top 20）
# ============================================================
cat("\n=== Top 20 重要特征（基于基尼系数）===\n")
print(head(feature_importance, 20))

# 保存特征重要性
write.csv(feature_importance, "./randomForest_weighted_feature_importance.csv", row.names = FALSE)


# ============================================================
# 保存模型和完整结果
# ============================================================
save(model_rf_weighted, feature_importance, results_df, cv_output,
     file = "./randomForest_weighted_model.rdata")
cat("模型已保存: ./randomForest_weighted_model.rdata\n")

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
write.csv(all_results, "./randomForest_weighted_predictions.csv", row.names = FALSE)
cat("完整预测结果已保存: ./randomForest_weighted_predictions.csv\n")

# ============================================================
# 输出模型摘要
# ============================================================
cat("\n========================================\n")
cat("随机森林模型摘要\n")
cat("========================================\n")
cat(sprintf("模型类型: Random Forest (随机森林)\n"))
cat(sprintf("参数配置:\n"))
cat(sprintf("  - mtry (特征采样数): %d\n", mtry_value))
cat(sprintf("  - ntree (树的数量): %d\n", ntree_value))
cat(sprintf("  - nodesize (节点最小样本): %d\n", nodesize_value))
cat(sprintf("  - sampsize (平衡采样大小): %d per class\n", sampsize_balanced[1]))
cat(sprintf("\n交叉验证 (5折) 平均性能:\n"))
for (i in 1:nrow(cv_output$summary)) {
  cat(sprintf("  - %s: %.3f (SD: %.3f)\n", 
              cv_output$summary$Metric[i],
              cv_output$summary$Mean[i],
              cv_output$summary$SD[i]))
}
cat(sprintf("\n训练信息:\n"))
cat(sprintf("  - OOB误差率: %.4f\n", model_rf_weighted$err.rate[ntree_value, "OOB"]))
cat(sprintf("  - 癌症类OOB误差: %.4f\n", model_rf_weighted$err.rate[ntree_value, "cancer"]))
cat(sprintf("  - 对照类OOB误差: %.4f\n", model_rf_weighted$err.rate[ntree_value, "control"]))
cat(sprintf("\n数据信息:\n"))
cat(sprintf("  - 训练集样本数: %d\n", nrow(train_matrix)))
cat(sprintf("  - 测试集样本数: %d\n", nrow(test_matrix)))
cat(sprintf("  - 特征数量: %d\n", ncol(train_matrix)))
cat(sprintf("  - 有效特征数: %d\n", non_zero_features))
cat(sprintf("\n最终模型性能:\n"))
cat(sprintf("  - 训练集AUC: %.3f\n", train_metrics$AUC))
cat(sprintf("  - 测试集AUC: %.3f\n", test_metrics$AUC))
cat(sprintf("  - 测试集准确率: %.3f\n", test_metrics$ACC))
cat(sprintf("  - 测试集F1分数: %.3f\n", test_metrics$F1))

cat("\n=== 随机森林模型训练完成 ===\n")