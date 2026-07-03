# ============================================================
# 随机森林模型：类别权重 + 阈值优化
# 输出训练集和测试集完整结果
# ============================================================

rm(list = ls())
library(randomForest)
library(pROC)
library(caret)
library(ggplot2)
library(dplyr) 

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
# 数据预处理 - 随机森林不需要标准化，但为了与其他模型一致，保持相同预处理
# ============================================================
preprocess_data <- function(data) {
  features <- data[, feature_cols]
  group <- data$group
  
  # 随机森林对特征尺度不敏感，可以不标准化
  # 但为了与glmnet代码保持一致，这里也进行标准化（可选）
  # features_scaled <- as.data.frame(scale(features))  # 可选
  
  return(data.frame(group = group, features))
}

# 预处理数据
train_processed <- preprocess_data(train_data)
test_processed <- preprocess_data(test_data)

# 准备矩阵和标签
train_matrix <- train_processed[, -1]
test_matrix <- test_processed[, -1]

train_labels <- train_processed$group
test_labels <- test_processed$group

# 转换为因子（randomForest要求）
train_labels_factor <- factor(train_labels, levels = c("control", "cancer"))
test_labels_factor <- factor(test_labels, levels = c("control", "cancer"))

# ============================================================
# 计算类别权重（用于平衡采样）
# ============================================================
cancer_count <- sum(train_labels == "cancer")
control_count <- sum(train_labels == "control")

# 逆频率权重（用于计算采样大小）
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
# 训练带平衡采样的随机森林模型
# ============================================================
cat("\n训练随机森林模型...\n")
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
csv_filename <- "./randomForest_weighted_results.csv"
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
# 特征重要性（Top 20）
# ============================================================
cat("\n=== Top 20 重要特征（基于基尼系数）===\n")
print(head(feature_importance, 20))

# 保存特征重要性
write.csv(feature_importance, "./randomForest_weighted_feature_importance.csv", row.names = FALSE)

# ============================================================
# 可视化特征重要性
# ============================================================
p_importance <- ggplot(feature_importance %>% head(20), 
                       aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(title = "随机森林特征重要性 (Top 20)",
       subtitle = paste0("基于基尼系数减少量 | mtry=", mtry_value, 
                         ", ntree=", ntree_value, ", nodesize=", nodesize_value),
       x = "特征",
       y = "Mean Decrease Gini") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("./randomForest_feature_importance.png", p_importance, width = 12, height = 8, dpi = 300)
cat("\n特征重要性图已保存: ./randomForest_feature_importance.png\n")

# ============================================================
# 绘制ROC曲线
# ============================================================
png("./randomForest_weighted_roc.png", width = 8, height = 6, units = "in", res = 300)

roc_train <- roc(ifelse(train_labels == "cancer", 1, 0), train_prob, quiet = TRUE)
roc_test <- roc(ifelse(test_labels == "cancer", 1, 0), test_prob, quiet = TRUE)

plot(roc_train, col = "blue", lwd = 2, main = "随机森林模型ROC曲线（平衡采样）")
plot(roc_test, col = "red", lwd = 2, add = TRUE)

legend("bottomright", 
       legend = c(sprintf("训练集 (AUC=%.3f)", auc(roc_train)),
                  sprintf("测试集 (AUC=%.3f)", auc(roc_test))),
       col = c("blue", "red"), lwd = 2, cex = 0.8)

dev.off()
cat("ROC曲线已保存: ./randomForest_weighted_roc.png\n")

# ============================================================
# 绘制OOB误差随树数量变化图
# ============================================================
error_df <- data.frame(
  Trees = 1:ntree_value,
  OOB_Error = model_rf_weighted$err.rate[, "OOB"],
  Cancer_Error = model_rf_weighted$err.rate[, "cancer"],
  Control_Error = model_rf_weighted$err.rate[, "control"]
)

p_error <- ggplot(error_df, aes(x = Trees)) +
  geom_line(aes(y = OOB_Error, color = "Overall"), size = 1) +
  geom_line(aes(y = Cancer_Error, color = "Cancer"), size = 1) +
  geom_line(aes(y = Control_Error, color = "Control"), size = 1) +
  scale_color_manual(values = c("Overall" = "black", "Cancer" = "red", "Control" = "blue"),
                     name = "Error Type") +
  labs(title = "随机森林OOB误差随树数量变化",
       subtitle = sprintf("最终OOB误差: %.4f | mtry=%d, nodesize=%d", 
                          model_rf_weighted$err.rate[ntree_value, "OOB"],
                          mtry_value, nodesize_value),
       x = "树的数量",
       y = "误差率") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("./randomForest_oob_error.png", p_error, width = 10, height = 6, dpi = 300)
cat("OOB误差图已保存: ./randomForest_oob_error.png\n")

# ============================================================
# 保存模型和完整结果
# ============================================================
save(model_rf_weighted, feature_importance, results_df,
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
cat(sprintf("\n训练信息:\n"))
cat(sprintf("  - OOB误差率: %.4f\n", model_rf_weighted$err.rate[ntree_value, "OOB"]))
cat(sprintf("  - 癌症类OOB误差: %.4f\n", model_rf_weighted$err.rate[ntree_value, "cancer"]))
cat(sprintf("  - 对照类OOB误差: %.4f\n", model_rf_weighted$err.rate[ntree_value, "control"]))
cat(sprintf("\n数据信息:\n"))
cat(sprintf("  - 训练集样本数: %d\n", nrow(train_matrix)))
cat(sprintf("  - 测试集样本数: %d\n", nrow(test_matrix)))
cat(sprintf("  - 特征数量: %d\n", ncol(train_matrix)))
cat(sprintf("  - 有效特征数: %d\n", non_zero_features))
cat(sprintf("\n性能指标:\n"))
cat(sprintf("  - 训练集AUC: %.3f\n", train_metrics$AUC))
cat(sprintf("  - 测试集AUC: %.3f\n", test_metrics$AUC))
cat(sprintf("  - 测试集准确率: %.3f\n", test_metrics$ACC))
cat(sprintf("  - 测试集F1分数: %.3f\n", test_metrics$F1))

cat("\n=== 随机森林模型训练完成 ===\n")

