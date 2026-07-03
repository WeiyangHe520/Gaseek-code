rm(list = ls())
set.seed(3456)
FIG_DIR <- "figures_rf/" 
DATA_DIR <- "data_rf/"

if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR)
if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR)

library(caret)
library(ggplot2)
library(dplyr)
library(tidyr)
library(pROC)
library(rBayesianOptimization)
library(randomForest)

# 加载数据
load(file = ".left_data.rdata")

# 检查原始数据分布
cat("=== 原始数据检查 ===\n")
cat("训练集分组分布:\n")
print(table(train_data$group))
cat("\n测试集分组分布:\n")
print(table(test_data$group))

# 重要：不要在这里重新转换因子，保持原始标签
# 检查原始数据特征
cat("\n原始特征统计（前5个特征）:\n")
print(summary(train_data[, 1:5]))

# ============================================================
# 关键修改：先检查是否需要Yeojohnson变换
# ============================================================
# 计算原始数据的偏度
library(moments)
feature_skewness <- apply(train_data[, -ncol(train_data)], 2, skewness)
cat("\n特征偏度范围:", range(feature_skewness))
cat("\n高度偏斜特征数 (>|1|):", sum(abs(feature_skewness) > 1))



# 准备数据
x_train <- train_data[, -ncol(train_data)]
y_train <- train_data$group
x_test <- test_data[, -ncol(test_data)]
y_test <- test_data$group

cat(sprintf("\n训练集维度: %d x %d\n", nrow(x_train), ncol(x_train)))
cat(sprintf("测试集维度: %d x %d\n", nrow(x_test), ncol(x_test)))

# ============================================================
# 简化的贝叶斯优化函数（先测试基础版本）
# ============================================================
bayes_opt_rf_simple <- function(mtry, ntree, nodesize) {
  # 参数约束
  mtry_val <- max(1, min(round(mtry), ncol(x_train)))
  ntree_val <- max(50, round(ntree))
  nodesize_val <- max(1, round(nodesize))
  
  # 不使用平衡采样，先测试原始数据
  tryCatch({
    # 使用OOB误差而不是交叉验证（更快）
    rf_model <- randomForest(
      x = x_train,
      y = y_train,
      mtry = mtry_val,
      ntree = ntree_val,
      nodesize = nodesize_val,
      importance = FALSE
    )
    
    # 获取OOB预测概率
    oob_pred <- predict(rf_model, type = "prob")[, "cancer"]
    
    # 计算OOB AUC
    roc_obj <- roc(ifelse(y_train == "cancer", 1, 0), oob_pred, quiet = TRUE)
    auc_value <- auc(roc_obj)
    
    # 调试输出
    cat(sprintf("  mtry=%d, ntree=%d, nodesize=%d, OOB AUC=%.4f\n", 
                mtry_val, ntree_val, nodesize_val, auc_value))
    
    list(Score = auc_value, Pred = 0)
  }, error = function(e) {
    cat(sprintf("  错误: %s\n", e$message))
    list(Score = 0.5, Pred = 0)
  })
}

# 先用简单版本测试
cat("\n=== 测试简单版本（使用OOB AUC）===\n")
test_model <- randomForest(x_train, y_train, ntree = 100)
test_oob <- predict(test_model, type = "prob")[, "cancer"]
test_auc <- auc(roc(ifelse(y_train == "cancer", 1, 0), test_oob))
cat(sprintf("基础模型OOB AUC: %.4f\n", test_auc))

if (test_auc < 0.51) {
  cat("\n警告：基础模型AUC也接近0.5！\n")
  cat("可能原因：\n")
  cat("1. 数据标签可能有问题\n")
  cat("2. 特征与标签无关联\n")
  cat("3. 数据泄露或预处理错误\n")
  
  # 诊断：检查特征与标签的相关性
  cat("\n=== 诊断：特征与标签的相关性 ===\n")
  y_numeric <- ifelse(y_train == "cancer", 1, 0)
  correlations <- sapply(x_train, function(x) cor(x, y_numeric, use = "complete.obs"))
  cat("特征相关性范围:", range(abs(correlations)), "\n")
  cat("最大相关性特征:", names(which.max(abs(correlations))), 
      "相关性=", max(abs(correlations)), "\n")
  
  if (max(abs(correlations)) < 0.1) {
    cat("警告：所有特征与标签相关性都很低！数据可能有问题。\n")
  }
}

# 如果基础模型有效，再进行贝叶斯优化
if (test_auc > 0.51) {
  cat("\n基础模型有效，开始贝叶斯优化...\n")
  
  bounds_rf_simple <- list(
    mtry = c(1, min(10, ncol(x_train))),
    ntree = c(50, 300),
    nodesize = c(1, 10)
  )
  
  set.seed(123)
  opt_result_rf <- BayesianOptimization(
    FUN = bayes_opt_rf_simple,
    bounds = bounds_rf_simple,
    init_points = 10,
    n_iter = 15,
    acq = "ucb",
    kappa = 2.576,
    verbose = TRUE
  )
  
  # 提取最佳参数
  mtry_best <- round(opt_result_rf$Best_Par[["mtry"]])
  ntree_best <- round(opt_result_rf$Best_Par[["ntree"]])
  nodesize_best <- round(opt_result_rf$Best_Par[["nodesize"]])
  
  cat("\n最佳参数组合:\n")
  cat(sprintf("mtry: %d\n", mtry_best))
  cat(sprintf("ntree: %d\n", ntree_best))
  cat(sprintf("nodesize: %d\n", nodesize_best))
  cat(sprintf("最佳AUC: %.4f\n", opt_result_rf$Best_Value))
  
} else {
  cat("\n使用默认参数训练模型\n")
  mtry_best <- floor(sqrt(ncol(x_train)))
  ntree_best <- 200
  nodesize_best <- 5
  opt_result_rf <- NULL
}

# ============================================================
# 训练最终模型（使用平衡采样处理类别不平衡）
# ============================================================
cat("\n训练最终模型...\n")

# 处理类别不平衡
cancer_count <- sum(y_train == "cancer")
control_count <- sum(y_train == "control")
cat(sprintf("癌症样本: %d, 对照样本: %d\n", cancer_count, control_count))

if (cancer_count < control_count) {
  sampsize_balanced <- c(cancer = cancer_count, control = cancer_count)
} else {
  sampsize_balanced <- c(cancer = control_count, control = control_count)
}
cat(sprintf("平衡采样大小: %d per class\n", sampsize_balanced[1]))

# 训练最终模型
final_rf <- randomForest(
  x = x_train,
  y = y_train,
  mtry = mtry_best,
  ntree = ntree_best,
  nodesize = nodesize_best,
  sampsize = sampsize_balanced,
  importance = TRUE
)

# ============================================================
# 模型评估
# ============================================================
# 预测
train_pred <- predict(final_rf, type = "prob")[, "cancer"]
test_pred <- predict(final_rf, x_test, type = "prob")[, "cancer"]

# ROC评估
train_roc <- roc(ifelse(y_train == "cancer", 1, 0), train_pred)
test_roc <- roc(ifelse(y_test == "cancer", 1, 0), test_pred)

cat("\n最终模型性能:\n")
cat(sprintf("训练集AUC: %.4f\n", auc(train_roc)))
cat(sprintf("测试集AUC: %.4f\n", auc(test_roc)))

# 寻找最佳阈值
coords_test <- coords(test_roc, "best", ret = "threshold")
best_thresh <- coords_test[1, "threshold"]

# 混淆矩阵
pred_class <- factor(ifelse(test_pred >= best_thresh, "cancer", "control"),
                     levels = c("control", "cancer"))
conf_matrix <- confusionMatrix(pred_class, y_test, positive = "cancer")

cat("\n测试集性能:\n")
cat(sprintf("  准确率: %.4f\n", conf_matrix$overall["Accuracy"]))
cat(sprintf("  灵敏度: %.4f\n", conf_matrix$byClass["Sensitivity"]))
cat(sprintf("  特异度: %.4f\n", conf_matrix$byClass["Specificity"]))
cat(sprintf("  F1分数: %.4f\n", conf_matrix$byClass["F1"]))

# ============================================================
# 特征重要性
# ============================================================
importance_df <- data.frame(
  Feature = rownames(importance(final_rf)),
  MeanDecreaseGini = importance(final_rf, type = 1)
) %>% arrange(desc(MeanDecreaseGini))

cat("\nTop 10重要特征:\n")
print(head(importance_df, 10))

# ============================================================
# 简单可视化
# ============================================================
# ROC曲线
png(file.path(FIG_DIR, "rf_roc.png"), width = 8, height = 6, units = "in", res = 300)
plot(train_roc, col = "blue", main = "随机森林ROC曲线")
plot(test_roc, col = "red", add = TRUE)
legend("bottomright", 
       legend = c(sprintf("训练集 AUC=%.3f", auc(train_roc)),
                  sprintf("测试集 AUC=%.3f", auc(test_roc))),
       col = c("blue", "red"), lwd = 2)
dev.off()

# 特征重要性图
p_importance <- ggplot(importance_df %>% head(15), 
                       aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "特征重要性", x = "特征", y = "Mean Decrease Gini") +
  theme_minimal()

ggsave(file.path(FIG_DIR, "rf_feature_importance.pdf"), p_importance, width = 10, height = 6)

# ============================================================
# 保存结果
# ============================================================
save(final_rf, importance_df, train_roc, test_roc,
     file = file.path(DATA_DIR, "final_rf_model.rdata"))

# 保存预测结果
results_df <- data.frame(
  Sample = 1:nrow(x_test),
  True_Label = y_test,
  Pred_Prob = test_pred,
  Pred_Label = ifelse(test_pred >= best_thresh, "cancer", "control")
)
write.csv(results_df, file.path(DATA_DIR, "rf_predictions.csv"), row.names = FALSE)

cat("\n=== 随机森林模型完成 ===\n")
cat(sprintf("结果保存在: %s\n", DATA_DIR))