rm(list = ls())
###############################################
## kknn贝叶斯优化模型                         ##
## 功能：贝叶斯优化+多可视化+模型部署         ##
## 模型：加权K最近邻 (kknn)                   ##
## 版本：v1.0 (2025-03-09)                   ##
###############################################

set.seed(3456)
FIG_DIR <- "figures_kknn/" 
DATA_DIR <- "data_kknn/"

if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR)
if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR)

library(caret)
library(ggplot2)
library(dplyr)
library(tidyr)
library(pROC)
library(rBayesianOptimization)
library(kknn)          # 加权KNN核心包

# 加载数据
load(file = ".left_data.rdata")



## 准备数据矩阵（kknn接受数据框即可）
x_train <- train_data[, -ncol(train_data)]
y_train <- train_data$group
x_test <- test_data[, -ncol(test_data)]
y_test <- test_data$group

# 将分类变量转换为数值（如果存在因子特征，需处理；此处假定均为数值）
# 如果有因子，需用model.matrix，但本数据集特征均为数值，故略

## ========== 贝叶斯优化函数 for kknn ==========
# 参数说明：
#   k : 邻居数量（整数，1~50）
#   kernel : 核函数类型（映射为整数：1=rectangular, 2=triangular, 3=epanechnikov, 4=optimal）
#   distance : 距离幂参数（通常1=曼哈顿距离，2=欧氏距离，可1~3）
bayes_opt_kknn <- function(k, kernel_id, distance) {
  # 参数约束
  k_val <- max(1, min(round(k), nrow(x_train)))   # k不能超过训练集样本数
  kernel_map <- c("rectangular", "triangular", "epanechnikov", "optimal")
  kernel_val <- kernel_map[max(1, min(round(kernel_id), length(kernel_map)))]
  distance_val <- max(0.5, min(distance, 3))       # 距离参数通常在1~2之间，放宽范围
  
  # 使用5折交叉验证评估模型
  tryCatch({
    # 设置交叉验证
    n_folds <- 5
    fold_ids <- sample(rep(1:n_folds, length.out = nrow(x_train)))
    auc_values <- numeric(n_folds)
    
    for (fold in 1:n_folds) {
      # 分割数据
      train_idx <- which(fold_ids != fold)
      val_idx <- which(fold_ids == fold)
      
      x_train_fold <- x_train[train_idx, ]
      y_train_fold <- y_train[train_idx]
      x_val_fold <- x_train[val_idx, ]
      y_val_fold <- y_train[val_idx]
      
      # 训练kknn模型
      # 注意：kknn函数需要训练集和测试集同时传入，并返回预测概率
      model_fold <- kknn(
        formula = group ~ .,
        train = cbind(x_train_fold, group = y_train_fold),
        test = x_val_fold,
        k = k_val,
        kernel = kernel_val,
        distance = distance_val
      )
      
      # 提取预测概率（癌症类别的概率）
      pred_prob <- model_fold$prob[, "cancer"]
      
      # 计算AUC
      roc_obj <- roc(ifelse(y_val_fold == "cancer", 1, 0), pred_prob, quiet = TRUE)
      auc_values[fold] <- auc(roc_obj)
    }
    
    auc_value <- mean(auc_values, na.rm = TRUE)
    
    # 调试输出（每10次随机输出一次）
    if (runif(1) < 0.1) {
      cat(sprintf("  k=%d, kernel=%s, distance=%.2f, AUC=%.4f\n", 
                  k_val, kernel_val, distance_val, auc_value))
    }
    
    list(Score = auc_value, Pred = 0)
  }, error = function(e) {
    cat(sprintf("  错误: k=%d, kernel=%s, distance=%.2f, %s\n", 
                k_val, kernel_val, distance_val, e$message))
    list(Score = 0.5, Pred = 0)
  })
}

## 执行贝叶斯优化
bounds_kknn <- list(
  k = c(1, 30),           # 邻居数范围
  kernel_id = c(1, 4),    # 核函数索引：1-4
  distance = c(1, 2)      # 距离幂：1=曼哈顿，2=欧氏（可扩大至3）
)

set.seed(123)
cat("\n开始贝叶斯优化（kknn）...\n")
opt_result_kknn <- BayesianOptimization(
  FUN = bayes_opt_kknn,
  bounds = bounds_kknn,
  init_points = 10,
  n_iter = 15,
  acq = "ucb",
  kappa = 2.576,
  verbose = TRUE
)

## 输出最佳参数
k_best <- round(opt_result_kknn$Best_Par[["k"]])
kernel_id_best <- round(opt_result_kknn$Best_Par[["kernel_id"]])
kernel_map <- c("rectangular", "triangular", "epanechnikov", "optimal")
kernel_best <- kernel_map[max(1, min(kernel_id_best, length(kernel_map)))]
distance_best <- opt_result_kknn$Best_Par[["distance"]]

cat("\n最佳参数组合:\n")
cat(sprintf("k: %d\n", k_best))
cat(sprintf("kernel: %s\n", kernel_best))
cat(sprintf("distance: %.4f\n", distance_best))
cat(sprintf("最佳AUC: %.4f\n", opt_result_kknn$Best_Value))

# 提取优化历史数据
history_df <- as.data.frame(opt_result_kknn$History)

# 添加核函数名称列
history_df <- history_df %>%
  mutate(
    Iteration = Round,
    Is_Best = Value == max(Value),
    Point_Type = ifelse(Round <= 10, "Initial Design", "Bayesian Optimization"),
    Label = ifelse(Is_Best, sprintf("Best: %.4f", Value), ""),
    kernel_name = case_when(
      round(kernel_id) == 1 ~ "rectangular",
      round(kernel_id) == 2 ~ "triangular",
      round(kernel_id) == 3 ~ "epanechnikov",
      round(kernel_id) == 4 ~ "optimal"
    )
  )

# 1. 目标函数值变化趋势
p1 <- ggplot(history_df, aes(x = Iteration, y = Value)) +
  geom_line(color = "steelblue", alpha = 0.7) +
  geom_point(aes(color = Point_Type, size = Is_Best, shape = Point_Type), alpha = 0.9) +
  geom_text(aes(label = Label), vjust = -1.5, size = 3.5, color = "red") +
  geom_hline(yintercept = opt_result_kknn$Best_Value, 
             linetype = "dashed", color = "red", alpha = 0.7) +
  scale_size_manual(values = c(3, 5)) +
  scale_shape_manual(values = c(16, 17)) +
  scale_color_manual(values = c("darkorange", "purple")) +
  theme_minimal() +
  labs(title = "kknn贝叶斯优化过程",
       subtitle = sprintf("最佳参数: k=%d, kernel=%s, distance=%.3f", 
                          k_best, kernel_best, distance_best),
       x = "迭代轮次", y = "AUC值") +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5)) +
  scale_x_continuous(breaks = 1:max(history_df$Iteration))

# 2. k参数探索轨迹
p2 <- ggplot(history_df, aes(x = Iteration, y = k)) +
  geom_line(color = "steelblue", alpha = 0.5) +
  geom_point(aes(size = Value, color = Value), alpha = 0.8) +
  geom_hline(yintercept = k_best, linetype = "dashed", color = "red") +
  geom_text(aes(label = ifelse(Is_Best, sprintf("Optimal: %d", k), "")),
            vjust = -1.5, size = 3, color = "red") +
  scale_color_gradient(low = "blue", high = "red", name = "AUC") +
  scale_size_continuous(range = c(3, 8), name = "AUC") +
  labs(title = "kknn k参数探索轨迹",
       subtitle = "k: 邻居数量（影响偏差-方差权衡）",
       y = "k值") +
  theme_minimal()

# 3. kernel参数探索（按迭代显示核函数类型）
p3 <- ggplot(history_df, aes(x = Iteration, y = factor(kernel_name, levels = kernel_map))) +
  geom_point(aes(size = Value, color = Value), alpha = 0.8) +
  geom_hline(yintercept = kernel_best, linetype = "dashed", color = "red") +
  scale_color_gradient(low = "blue", high = "red", name = "AUC") +
  scale_size_continuous(range = c(3, 8), name = "AUC") +
  labs(title = "kknn kernel参数探索",
       subtitle = "核函数类型：影响样本权重分配",
       y = "kernel") +
  theme_minimal() +
  theme(axis.text.y = element_text(angle = 0))

# 4. distance参数探索轨迹
p4 <- ggplot(history_df, aes(x = Iteration, y = distance)) +
  geom_line(color = "steelblue", alpha = 0.5) +
  geom_point(aes(size = Value, color = Value), alpha = 0.8) +
  geom_hline(yintercept = distance_best, linetype = "dashed", color = "red") +
  geom_text(aes(label = ifelse(Is_Best, sprintf("Optimal: %.3f", distance), "")),
            vjust = -1.5, size = 3, color = "red") +
  scale_color_gradient(low = "blue", high = "red", name = "AUC") +
  scale_size_continuous(range = c(3, 8), name = "AUC") +
  labs(title = "kknn distance参数探索轨迹",
       subtitle = "distance: 距离幂参数（1=曼哈顿，2=欧氏）",
       y = "distance") +
  theme_minimal()

# 保存图形
ggsave(file.path(FIG_DIR, "kknn_optimization_process.pdf"), p1, width = 10, height = 8)
ggsave(file.path(FIG_DIR, "kknn_k_exploration.pdf"), p2, width = 10, height = 6)
ggsave(file.path(FIG_DIR, "kknn_kernel_exploration.pdf"), p3, width = 10, height = 6)
ggsave(file.path(FIG_DIR, "kknn_distance_exploration.pdf"), p4, width = 10, height = 6)

## 使用最佳参数训练最终模型
# 训练集完整模型（用于预测）
final_kknn <- kknn(
  formula = group ~ .,
  train = cbind(x_train, group = y_train),
  test = x_train,   # 训练集预测
  k = k_best,
  kernel = kernel_best,
  distance = distance_best
)

# 测试集预测
test_kknn <- kknn(
  formula = group ~ .,
  train = cbind(x_train, group = y_train),
  test = x_test,
  k = k_best,
  kernel = kernel_best,
  distance = distance_best
)

# 提取概率
train_pred <- final_kknn$prob[, "cancer"]
test_pred <- test_kknn$prob[, "cancer"]

# 性能评估
positive_class <- "cancer"
train_roc <- roc(response = as.numeric(y_train == positive_class), 
                 predictor = train_pred)
test_roc <- roc(response = as.numeric(y_test == positive_class), 
                predictor = test_pred)

cat("\n最终模型性能:\n")
cat(sprintf("训练集AUC: %.4f\n", auc(train_roc)))
cat(sprintf("测试集AUC: %.4f\n", auc(test_roc)))

# 寻找最佳阈值（基于测试集）
coords_test <- coords(test_roc, "best", ret = "threshold")
best_thresh <- coords_test[1, "threshold"]

# 混淆矩阵及其他指标
pred_class <- factor(ifelse(test_pred >= best_thresh, "cancer", "control"),
                     levels = c("control", "cancer"))
conf_matrix <- confusionMatrix(pred_class, y_test, positive = "cancer")

cat("\n测试集详细性能:\n")
cat(sprintf("  最佳阈值: %.4f\n", best_thresh))
cat(sprintf("  准确率: %.4f\n", conf_matrix$overall["Accuracy"]))
cat(sprintf("  灵敏度: %.4f\n", conf_matrix$byClass["Sensitivity"]))
cat(sprintf("  特异度: %.4f\n", conf_matrix$byClass["Specificity"]))
cat(sprintf("  F1分数: %.4f\n", conf_matrix$byClass["F1"]))

## 特征重要性（KNN无原生重要性，可基于变量对AUC的贡献评估，此处用caret的filterVarImp）
# 使用单变量逻辑回归评估每个特征的区分能力（仅供参考）
var_imp <- sapply(names(x_train), function(var) {
  model <- glm(y_train == "cancer" ~ x_train[[var]], family = binomial)
  coef_sum <- summary(model)$coefficients
  if (nrow(coef_sum) > 1) abs(coef_sum[2, 3]) else 0  # |t值|
})
importance_df <- data.frame(
  Feature = names(var_imp),
  Importance = var_imp
) %>% arrange(desc(Importance))

cat("\n=== Top 10 重要特征（基于单变量逻辑回归t值）===\n")
print(head(importance_df, 10))

# 可视化特征重要性
p_importance <- ggplot(importance_df %>% head(20), 
                       aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(title = "kknn特征重要性 (Top 20，基于单变量逻辑回归)",
       subtitle = sprintf("k=%d, kernel=%s, distance=%.3f", k_best, kernel_best, distance_best),
       x = "特征", y = "|t值|") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave(file.path(FIG_DIR, "kknn_feature_importance.pdf"), p_importance, width = 12, height = 8)

## ROC曲线绘制
png(file.path(FIG_DIR, "kknn_roc.png"), width = 8, height = 6, units = "in", res = 300)
plot(train_roc, col = "blue", lwd = 2, main = "kknn模型ROC曲线")
plot(test_roc, col = "red", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c(sprintf("训练集 (AUC=%.3f)", auc(train_roc)),
                  sprintf("测试集 (AUC=%.3f)", auc(test_roc))),
       col = c("blue", "red"), lwd = 2, cex = 0.8)
dev.off()

## 保存模型与结果
# 保存最终模型对象（注意kknn模型无法直接保存预测函数，故保存训练集和参数）
save(final_kknn, test_kknn, preprocess_params, opt_result_kknn,
     train_data, test_data,
     file = file.path(DATA_DIR, "final_kknn_model.rdata"))

# 保存特征重要性
write.csv(importance_df, file.path(DATA_DIR, "kknn_feature_importance.csv"), row.names = FALSE)

# 保存预测结果
train_results <- data.frame(
  Sample_ID = rownames(x_train),
  Group = y_train,
  Pred_Prob_Cancer = train_pred,
  Pred_Class = ifelse(train_pred >= best_thresh, "cancer", "control"),
  Set = "Training"
)

test_results <- data.frame(
  Sample_ID = rownames(x_test),
  Group = y_test,
  Pred_Prob_Cancer = test_pred,
  Pred_Class = ifelse(test_pred >= best_thresh, "cancer", "control"),
  Set = "Testing"
)

all_results <- rbind(train_results, test_results)
write.csv(all_results, file.path(DATA_DIR, "kknn_predictions.csv"), row.names = FALSE)

## 模型摘要
cat("\n========================================\n")
cat("kknn模型摘要\n")
cat("========================================\n")
cat(sprintf("模型类型: 加权K最近邻 (kknn)\n"))
cat(sprintf("邻居数 k: %d\n", k_best))
cat(sprintf("核函数: %s\n", kernel_best))
cat(sprintf("距离参数 distance: %.4f\n", distance_best))
cat(sprintf("训练集样本数: %d\n", nrow(x_train)))
cat(sprintf("测试集样本数: %d\n", nrow(x_test)))
cat(sprintf("特征数量: %d\n", ncol(x_train)))
cat(sprintf("训练集AUC: %.4f\n", auc(train_roc)))
cat(sprintf("测试集AUC: %.4f\n", auc(test_roc)))

# 性能对比表
performance_comparison <- data.frame(
  Dataset = c("Training", "Test"),
  AUC = c(auc(train_roc), auc(test_roc)),
  k = c(k_best, k_best),
  kernel = c(kernel_best, kernel_best),
  distance = c(distance_best, distance_best)
)

print(performance_comparison)
write.csv(performance_comparison, file.path(DATA_DIR, "kknn_performance.csv"), row.names = FALSE)

cat("\n=== kknn模型优化完成！ ===\n")
cat(sprintf("所有结果已保存至: %s 和 %s\n", FIG_DIR, DATA_DIR))