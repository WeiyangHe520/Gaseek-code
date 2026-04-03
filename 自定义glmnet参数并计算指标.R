rm(list = ls())
library(glmnet)
library(pROC)
library(caret)

set.seed(3456)

# 加载数据
load(file = ".left_data.rdata")

# 数据预处理 - 标准化特征（与xgbLinear保持一致）
preprocess_data <- function(data) {
  features <- data[, -which(names(data) == "group")]
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

# 定义glmnet参数
alpha_value <- 0.5281 # 平衡L1和L2正则化
lambda_value <- 0.004597  # 正则化强度

cat("glmnet参数配置:\n")
cat("alpha =", alpha_value, "(L1/L2混合参数)\n")
cat("lambda =", lambda_value, "(正则化强度)\n")

# 训练glmnet模型
last_model_glmnet <- glmnet(
  x = train_matrix,
  y = train_labels,
  family = "binomial",
  alpha = alpha_value,
  lambda = lambda_value,
  standardize = FALSE,
  intercept = TRUE,
  thresh = 1e-7,
  maxit = 1000
)

# 预测概率
train_prob <- predict(last_model_glmnet, newx = train_matrix, type = "response")[,1]
test_prob <- predict(last_model_glmnet, newx = test_matrix, type = "response")[,1]

# 获取系数
coefficients <- coef(last_model_glmnet)
cat("\n模型系数摘要:\n")
print(summary(as.vector(coefficients)))

# 特征选择：统计非零系数数量
non_zero_coef <- sum(coefficients != 0) - 1
cat("非零系数数量（不包括截距）:", non_zero_coef, "\n")

# 最终预测（使用0.5阈值）
train_pred <- factor(ifelse(train_prob > 0.5, "cancer", "control"), 
                     levels = c("control", "cancer"))
test_pred <- factor(ifelse(test_prob > 0.5, "cancer", "control"), 
                    levels = c("control", "cancer"))

true_train_labels <- factor(train_data$group, levels = c("control", "cancer"))
true_test_labels <- factor(test_data$group, levels = c("control", "cancer"))

# 改进的评估函数：包含AUC 95%置信区间
calculate_metrics_with_ci <- function(true_labels, predictions, probabilities, dataset_name) {
  
  cm <- confusionMatrix(predictions, true_labels, positive = "cancer")
  
  acc <- cm$overall["Accuracy"]
  sens <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  ppv <- cm$byClass["Pos Pred Value"]
  npv <- cm$byClass["Neg Pred Value"]
  
  true_numeric <- ifelse(true_labels == "cancer", 1, 0)
  
  # 初始化AUC及其置信区间
  auc_value <- 0
  auc_ci_lower <- 0
  auc_ci_upper <- 0
  auc_95ci <- "0 (0-0)"
  
  if(length(unique(true_numeric)) < 2) {
    cat("警告：", dataset_name, "中只有一个类别，无法计算AUC\n")
    auc_value <- NA
  } else {
    # 计算ROC曲线
    roc_obj <- roc(true_numeric, probabilities, ci = TRUE, auc = TRUE)
    
    # 获取AUC值
    auc_value <- auc(roc_obj)[1]
    
    # 获取AUC的95%置信区间
    auc_ci <- ci.auc(roc_obj, conf.level = 0.95)
    
    # 提取置信区间上下限
    if (length(auc_ci) >= 3) {
      auc_ci_lower <- auc_ci[1]
      auc_ci_upper <- auc_ci[3]
      auc_95ci <- sprintf("%.3f (%.3f-%.3f)", auc_value, auc_ci_lower, auc_ci_upper)
    } else {
      # 如果无法计算置信区间，只使用AUC值
      auc_95ci <- sprintf("%.3f", auc_value)
    }
  }
  
  ydi <- sens + spec - 1
  f1 <- ifelse(is.na(ppv) | is.na(sens), NA, 2 * (ppv * sens) / (ppv + sens))
  
  cat("\n", dataset_name, "评估指标:\n")
  cat("AUC: ", round(auc_value, 3), "\n")
  cat("AUC 95%CI: ", auc_95ci, "\n")
  cat("ACC: ", round(acc, 3), "\n")
  cat("SENS: ", round(sens, 3), "\n")
  cat("SPEC: ", round(spec, 3), "\n")
  cat("PPV: ", round(ppv, 3), "\n")
  cat("NPV: ", round(npv, 3), "\n")
  cat("YDI: ", round(ydi, 3), "\n")
  cat("F1: ", round(f1, 3), "\n")
  
  return(list(
    AUC = round(auc_value, 3),
    AUC_95CI = auc_95ci,
    AUC_lower = round(auc_ci_lower, 3),
    AUC_upper = round(auc_ci_upper, 3),
    ACC = round(acc, 3),
    SENS = round(sens, 3),
    SPEC = round(spec, 3),
    PPV = round(ppv, 3),
    NPV = round(npv, 3),
    YDI = round(ydi, 3),
    F1 = round(f1, 3)
  ))
}

# 计算指标（包含AUC置信区间）
train_metrics <- calculate_metrics_with_ci(true_train_labels, train_pred, train_prob, "训练集")
test_metrics <- calculate_metrics_with_ci(true_test_labels, test_pred, test_prob, "测试集")

# 创建详细结果数据框
results_df_detailed <- data.frame(
  Dataset = c("Training", "Testing"),
  Alpha = c(alpha_value, alpha_value),
  Lambda = c(lambda_value, lambda_value),
  NonZero_Features = c(non_zero_coef, non_zero_coef),
  Model_Type = c("glmnet_elasticnet", "glmnet_elasticnet"),
  AUC = c(train_metrics$AUC, test_metrics$AUC),
  AUC_95CI = c(train_metrics$AUC_95CI, test_metrics$AUC_95CI),
  AUC_lower = c(train_metrics$AUC_lower, test_metrics$AUC_lower),
  AUC_upper = c(train_metrics$AUC_upper, test_metrics$AUC_upper),
  ACC = c(train_metrics$ACC, test_metrics$ACC),
  SENS = c(train_metrics$SENS, test_metrics$SENS),
  SPEC = c(train_metrics$SPEC, test_metrics$SPEC),
  PPV = c(train_metrics$PPV, test_metrics$PPV),
  NPV = c(train_metrics$NPV, test_metrics$NPV),
  YDI = c(train_metrics$YDI, test_metrics$YDI),
  F1 = c(train_metrics$F1, test_metrics$F1)
)

print("\n模型性能汇总（包含AUC 95%置信区间）:")
print(results_df_detailed)

# 保存为CSV文件
csv_filename <- "./glmnet_evaluation_metrics.csv"
write.csv(results_df_detailed, file = csv_filename, row.names = FALSE)
cat("\n完整指标已保存到CSV文件:", csv_filename, "\n")

# 显示CSV文件内容
cat("\nCSV文件内容预览:\n")
print(read.csv(csv_filename))

# 特征重要性（基于系数绝对值）
feature_importance <- data.frame(
  Feature = rownames(coefficients)[-1],  # 排除截距
  Coefficient = as.vector(coefficients[-1]),
  Abs_Coefficient = abs(as.vector(coefficients[-1]))
)

# 按系数绝对值排序
feature_importance <- feature_importance[order(-feature_importance$Abs_Coefficient), ]

cat("\nTop 10 最重要的特征:\n")
print(head(feature_importance, 10))

# 保存特征重要性到CSV文件
feature_csv_filename <- "./glmnet_feature_importance.csv"
write.csv(feature_importance, file = feature_csv_filename, row.names = FALSE)
cat("特征重要性已保存到CSV文件:", feature_csv_filename, "\n")

# 保存模型和结果
save(last_model_glmnet, file = "./last_model_glmnet.rdata")
save(results_df_detailed, file = "./glmnet_evaluation_results.rdata")
save(feature_importance, file = "./glmnet_feature_importance.rdata")

# 绘制ROC曲线（包含置信区间）
par(mfrow = c(1, 2))

train_true_numeric <- ifelse(true_train_labels == "cancer", 1, 0)
test_true_numeric <- ifelse(true_test_labels == "cancer", 1, 0)

train_roc <- roc(train_true_numeric, train_prob, ci = TRUE)
test_roc <- roc(test_true_numeric, test_prob, ci = TRUE)

# 训练集ROC曲线
plot(train_roc, main = "训练集ROC曲线", col = "blue")
legend_text_train <- sprintf("AUC = %.3f (95%% CI: %.3f-%.3f)", 
                             auc(train_roc), 
                             ci.auc(train_roc)[1], 
                             ci.auc(train_roc)[3])
legend("bottomright", legend = legend_text_train, cex = 0.8)

# 测试集ROC曲线
plot(test_roc, main = "测试集ROC曲线", col = "red")
legend_text_test <- sprintf("AUC = %.3f (95%% CI: %.3f-%.3f)", 
                            auc(test_roc), 
                            ci.auc(test_roc)[1], 
                            ci.auc(test_roc)[3])
legend("bottomright", legend = legend_text_test, cex = 0.8)

# 系数可视化
if(nrow(feature_importance) > 0) {
  par(mfrow = c(1, 1))
  top_n <- min(20, nrow(feature_importance))
  top_features <- head(feature_importance, top_n)
  
  barplot(top_features$Coefficient, 
          names.arg = top_features$Feature,
          las = 2, cex.names = 0.7,
          col = ifelse(top_features$Coefficient > 0, "blue", "red"),
          main = paste("Top", top_n, "特征系数"),
          ylab = "系数值")
  abline(h = 0, lty = 2)
}

cat("\n=== glmnet模型训练完成 ===\n")
cat("参数配置: alpha =", alpha_value, ", lambda =", lambda_value, "\n")
cat("非零特征数:", non_zero_coef, "\n")
cat("\n主要输出文件:\n")
cat("1. 评估指标CSV:", csv_filename, "\n")
cat("2. 特征重要性CSV:", feature_csv_filename, "\n")
cat("3. 模型文件: ./last_model_glmnet.rdata\n")

# 可选：交叉验证调优
cat("\n=== 交叉验证调优 ===\n")
cv_glmnet <- cv.glmnet(
  x = train_matrix,
  y = train_labels,
  family = "binomial",
  alpha = alpha_value,
  nfolds = 5,
  type.measure = "auc"
)

cat("交叉验证最佳lambda:", cv_glmnet$lambda.min, "\n")
cat("交叉验证最佳AUC:", max(cv_glmnet$cvm), "\n")

# 保存交叉验证结果
cv_results <- data.frame(
  Alpha = alpha_value,
  Best_Lambda = cv_glmnet$lambda.min,
  Best_AUC = max(cv_glmnet$cvm),
  Lambda_1SE = cv_glmnet$lambda.1se,
  AUC_1SE = cv_glmnet$cvm[which(cv_glmnet$lambda == cv_glmnet$lambda.1se)],
  N_Folds = 5
)
cv_csv_filename <- "./glmnet_cross_validation_results.csv"
write.csv(cv_results, file = cv_csv_filename, row.names = FALSE)
cat("交叉验证结果已保存到:", cv_csv_filename, "\n")

# 生成简化的汇总报告
cat("\n=== 模型性能汇总报告 ===\n")
cat(sprintf("训练集: AUC = %.3f (95%% CI: %.3f-%.3f), ACC = %.3f\n", 
            train_metrics$AUC, train_metrics$AUC_lower, train_metrics$AUC_upper, train_metrics$ACC))
cat(sprintf("测试集: AUC = %.3f (95%% CI: %.3f-%.3f), ACC = %.3f\n", 
            test_metrics$AUC, test_metrics$AUC_lower, test_metrics$AUC_upper, test_metrics$ACC))