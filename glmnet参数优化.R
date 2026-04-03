rm(list = ls())
###############################################
## glmnet贝叶斯优化模型                      ##
## 功能：贝叶斯优化+多可视化+模型部署       ##
## 作者：基于Multinom代码改编               ##
## 版本：v1.0 (2024-05-25)                  ##
###############################################

set.seed(3456)
FIG_DIR <- "figures_glmnet/" 
DATA_DIR <- "data_glmnet/"

if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR)
if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR)

library(caret)
library(ggplot2)
library(dplyr)
library(tidyr)
library(pROC)
library(rBayesianOptimization)
library(glmnet)
library(glmnetUtils)  # 提供更友好的glmnet接口

# 加载数据
load(file = ".left_data.rdata")

## 数据预处理 - glmnet会在内部处理标准化，但这里我们为了保持一致性仍然预处理
preprocess_params <- preProcess(train_data[, -ncol(train_data)], method = c("center", "scale"))
train_data_processed <- predict(preprocess_params, train_data)
test_data_processed <- predict(preprocess_params, test_data)

# 将响应变量转换为因子（确保是二分类）
train_data_processed$group <- factor(train_data_processed$group, levels = c("control", "cancer"))
test_data_processed$group <- factor(test_data_processed$group, levels = c("control", "cancer"))

## 准备glmnet需要的矩阵格式
# caret的train函数会自动处理，但为了更高效的贝叶斯优化，我们直接使用glmnet
x_train <- as.matrix(train_data_processed[, -ncol(train_data_processed)])
y_train <- train_data_processed$group
x_test <- as.matrix(test_data_processed[, -ncol(test_data_processed)])
y_test <- test_data_processed$group

## 贝叶斯优化函数 for glmnet
bayes_opt_glmnet <- function(alpha, lambda) {
  # 参数转换和约束
  alpha_val <- max(min(alpha, 1), 0)  # alpha在[0,1]之间
  lambda_val <- max(lambda, 1e-6)      # lambda需要为正数
  
  # 使用交叉验证评估模型
  tryCatch({
    # 使用glmnet的cv.glmnet进行交叉验证
    cv_fit <- cv.glmnet(
      x = x_train,
      y = y_train,
      family = "binomial",      # 二分类问题
      alpha = alpha_val,
      lambda = exp(seq(log(0.001), log(1), length.out = 50)),  # lambda范围
      nfolds = 5,
      type.measure = "auc",     # 使用AUC作为评估指标
      standardize = FALSE       # 数据已经标准化过
    )
    
    # 获取最佳lambda对应的AUC
    auc_value <- max(cv_fit$cvm, na.rm = TRUE)
    
    list(Score = auc_value, Pred = 0)
  }, error = function(e) {
    # 如果出错，返回一个较低的分数
    list(Score = 0.5, Pred = 0)
  })
}

## 执行贝叶斯优化
bounds_glmnet <- list(
  alpha = c(0, 1),        # alpha: 0=岭回归, 1=lasso, 中间值=弹性网
  lambda = c(-6, 0)       # log(lambda)的范围: exp(-6)到exp(0) = 0.0025到1
)

set.seed(123)
opt_result_glmnet <- BayesianOptimization(
  FUN = bayes_opt_glmnet,
  bounds = bounds_glmnet,
  init_points = 8,
  n_iter = 12,
  acq = "ucb",
  kappa = 2.576,          # 探索参数
  verbose = TRUE
)

## 输出结果
cat("最佳参数组合:\n")
cat(sprintf("alpha: %.4f\n", opt_result_glmnet$Best_Par[["alpha"]]))
cat(sprintf("lambda: %.6f (log-scale: %.3f)\n", 
            exp(opt_result_glmnet$Best_Par[["lambda"]]),
            opt_result_glmnet$Best_Par[["lambda"]]))
cat(sprintf("最佳AUC: %.4f\n", opt_result_glmnet$Best_Value))

# 提取优化历史数据
history_df <- as.data.frame(opt_result_glmnet$History)

# 添加计算列
history_df <- history_df %>%
  mutate(
    Iteration = Round,
    Is_Best = Value == max(Value),
    Point_Type = ifelse(Round <= 8, "Initial Design", "Bayesian Optimization"),
    Label = ifelse(Is_Best, sprintf("Best: %.4f", Value), ""),
    lambda_exp = exp(lambda)  # 将log-lambda转换回原始尺度
  )

# 1. 目标函数值变化趋势
p1 <- ggplot(history_df, aes(x = Iteration, y = Value)) +
  geom_line(color = "steelblue", alpha = 0.7) +
  geom_point(aes(color = Point_Type, size = Is_Best, shape = Point_Type), alpha = 0.9) +
  geom_text(aes(label = Label), vjust = -1.5, size = 3.5, color = "red") +
  geom_hline(yintercept = opt_result_glmnet$Best_Value, 
             linetype = "dashed", color = "red", alpha = 0.7) +
  scale_size_manual(values = c(3, 5)) +
  scale_shape_manual(values = c(16, 17)) +
  scale_color_manual(values = c("darkorange", "purple")) +
  theme_minimal() +
  labs(title = "glmnet贝叶斯优化过程",
       subtitle = sprintf("最佳参数: alpha=%.3f, lambda=%.4f", 
                          opt_result_glmnet$Best_Par[["alpha"]],
                          exp(opt_result_glmnet$Best_Par[["lambda"]])),
       x = "迭代轮次", y = "AUC值") +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5)) +
  scale_x_continuous(breaks = 1:max(history_df$Iteration)) +
  ylim(min(history_df$Value) * 0.995, max(history_df$Value) * 1.005)

# 2. 参数空间探索轨迹 (二维)
p2 <- ggplot(history_df, aes(x = alpha, y = lambda_exp)) +
  geom_point(aes(size = Value, color = Value), alpha = 0.8) +
  geom_path(alpha = 0.3, color = "gray") +
  geom_point(data = history_df %>% filter(Is_Best), 
             aes(x = alpha, y = lambda_exp),
             color = "red", shape = 1, size = 8, stroke = 1.5) +
  geom_text(data = history_df %>% filter(Is_Best),
            aes(label = "最佳点"), 
            vjust = -1.5, size = 3.5, color = "red") +
  scale_color_gradient(low = "blue", high = "red", name = "AUC") +
  scale_size_continuous(range = c(3, 8), name = "AUC") +
  scale_y_log10() +  # lambda使用对数尺度
  labs(title = "glmnet参数空间探索",
       x = "alpha (L1/L2混合参数)",
       y = "lambda (正则化强度，log10尺度)") +
  theme_minimal()

# 3. alpha参数探索轨迹
p3 <- ggplot(history_df, aes(x = Iteration, y = alpha)) +
  geom_line(color = "steelblue", alpha = 0.5) +
  geom_point(aes(size = Value, color = Value), alpha = 0.8) +
  geom_hline(yintercept = opt_result_glmnet$Best_Par[["alpha"]], 
             linetype = "dashed", color = "red") +
  geom_text(aes(label = ifelse(Is_Best, sprintf("Optimal: %.3f", alpha), "")),
            vjust = -1.5, size = 3, color = "red") +
  scale_color_gradient(low = "blue", high = "red", name = "AUC") +
  scale_size_continuous(range = c(3, 8), name = "AUC") +
  labs(title = "glmnet alpha参数探索轨迹",
       subtitle = "alpha=0: 岭回归, alpha=1: Lasso, 中间值: 弹性网",
       y = "alpha") +
  theme_minimal()

# 4. lambda参数探索轨迹
p4 <- ggplot(history_df, aes(x = Iteration, y = lambda_exp)) +
  geom_line(color = "steelblue", alpha = 0.5) +
  geom_point(aes(size = Value, color = Value), alpha = 0.8) +
  geom_hline(yintercept = exp(opt_result_glmnet$Best_Par[["lambda"]]), 
             linetype = "dashed", color = "red") +
  geom_text(aes(label = ifelse(Is_Best, 
                               sprintf("Optimal: %.4f", lambda_exp), "")),
            vjust = -1.5, size = 3, color = "red") +
  scale_color_gradient(low = "blue", high = "red", name = "AUC") +
  scale_size_continuous(range = c(3, 8), name = "AUC") +
  scale_y_log10() +
  labs(title = "glmnet lambda参数探索轨迹 (log10尺度)",
       y = "lambda") +
  theme_minimal()

# 保存所有图形
ggsave(file.path(FIG_DIR, "glmnet_optimization_process.pdf"), p1, width = 10, height = 8)
ggsave(file.path(FIG_DIR, "glmnet_parameter_space_exploration.pdf"), p2, width = 10, height = 8)
ggsave(file.path(FIG_DIR, "glmnet_alpha_exploration.pdf"), p3, width = 10, height = 8)
ggsave(file.path(FIG_DIR, "glmnet_lambda_exploration.pdf"), p4, width = 10, height = 8)

## 使用最佳参数训练最终模型
alpha_best <- opt_result_glmnet$Best_Par[["alpha"]]
lambda_best <- exp(opt_result_glmnet$Best_Par[["lambda"]])

# 使用caret的train函数进行训练（为了保持与之前代码的一致性）
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

tune_grid <- expand.grid(
  alpha = alpha_best,
  lambda = lambda_best
)

final_model <- train(
  x = x_train,
  y = y_train,
  method = "glmnet",
  family = "binomial",
  trControl = ctrl,
  tuneGrid = tune_grid,
  metric = "ROC",
  standardize = FALSE  # 数据已经标准化
)

# 或者直接使用glmnet训练（更高效）
final_glmnet <- glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = alpha_best,
  lambda = lambda_best,
  standardize = FALSE
)

# 模型预测
train_pred <- predict(final_glmnet, newx = x_train, type = "response")[, 1]
test_pred <- predict(final_glmnet, newx = x_test, type = "response")[, 1]

# 注意：predict返回的是癌症类别的概率
# 需要根据因子的水平确定哪个是正类
positive_class <- "cancer"

# 性能评估
train_roc <- roc(response = as.numeric(y_train == positive_class), 
                 predictor = train_pred)
test_roc <- roc(response = as.numeric(y_test == positive_class), 
                predictor = test_pred)

cat("最终模型性能:\n")
cat(sprintf("训练集AUC: %.4f\n", auc(train_roc)))
cat(sprintf("测试集AUC: %.4f\n", auc(test_roc)))

# 保存模型
save(final_model, final_glmnet, preprocess_params, opt_result_glmnet,
     file = file.path(DATA_DIR, "final_glmnet_model.rdata"))

## 特征重要性分析
# 提取系数
coefficients <- coef(final_glmnet, s = lambda_best)
coef_df <- data.frame(
  Feature = rownames(coefficients),
  Coefficient = as.numeric(coefficients)
) %>%
  filter(Feature != "(Intercept)") %>%
  arrange(desc(abs(Coefficient)))

# 可视化特征重要性（基于系数绝对值）
p_importance <- ggplot(coef_df %>% head(20), 
                       aes(x = reorder(Feature, abs(Coefficient)), y = abs(Coefficient))) +
  geom_col(aes(fill = Coefficient > 0), alpha = 0.8) +
  coord_flip() +
  scale_fill_manual(values = c("red", "blue"), 
                    name = "系数方向",
                    labels = c("负向", "正向")) +
  labs(title = "glmnet特征重要性 (Top 20，基于系数绝对值)",
       subtitle = sprintf("模型类型: %s (alpha=%.3f)", 
                          ifelse(alpha_best == 0, "岭回归",
                                 ifelse(alpha_best == 1, "Lasso", "弹性网")),
                          alpha_best),
       x = "特征",
       y = "系数绝对值") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave(file.path(FIG_DIR, "glmnet_feature_importance.pdf"), p_importance, width = 12, height = 8)

## 系数可视化
p_coef <- ggplot(coef_df %>% head(20), 
                 aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_col(aes(fill = Coefficient > 0), alpha = 0.8) +
  coord_flip() +
  scale_fill_manual(values = c("red", "blue"), 
                    name = "系数方向",
                    labels = c("负向", "正向")) +
  labs(title = "glmnet模型系数 (Top 20)",
       x = "特征",
       y = "系数值") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave(file.path(FIG_DIR, "glmnet_coefficients.pdf"), p_coef, width = 12, height = 8)

## 稀疏性分析
# 统计非零系数的数量
non_zero_coefs <- sum(coef_df$Coefficient != 0)
total_features <- nrow(coef_df)

cat("\n=== 模型稀疏性分析 ===\n")
cat(sprintf("总特征数: %d\n", total_features))
cat(sprintf("非零系数特征数: %d\n", non_zero_coefs))
cat(sprintf("稀疏性: %.1f%%\n", 100 * (total_features - non_zero_coefs) / total_features))
cat(sprintf("alpha = %.3f (0=岭回归, 1=Lasso)\n", alpha_best))
cat(sprintf("lambda = %.6f\n", lambda_best))

## 正则化路径可视化（选做）
if (non_zero_coefs > 0) {
  # 计算完整正则化路径
  glmnet_path <- glmnet(
    x = x_train,
    y = y_train,
    family = "binomial",
    alpha = alpha_best,
    standardize = FALSE
  )
  
  # 保存路径图
  pdf(file.path(FIG_DIR, "glmnet_regularization_path.pdf"), width = 10, height = 8)
  plot(glmnet_path, xvar = "lambda", label = TRUE, main = "glmnet正则化路径")
  abline(v = log(lambda_best), col = "red", lty = 2, lwd = 2)
  text(x = log(lambda_best), y = max(glmnet_path$beta) * 0.9, 
       "最佳lambda", col = "red", pos = 4)
  dev.off()
}

## 模型部署（如果需要的函数存在）
if (file.exists("smart_caret_deploy.R")) {
  source("smart_caret_deploy.R")
  export_app(final_model, output_dir = "glmnet_app", app_title = "glmnet分类器")
  deploy_model(final_model, title = 'glmnet分类器', port = 8891)
}

cat("\n=== glmnet模型优化完成！ ===\n")
cat("最佳参数:\n")
cat(sprintf("alpha: %.4f\n", alpha_best))
cat(sprintf("lambda: %.6f\n", lambda_best))
cat(sprintf("测试集AUC: %.4f\n", auc(test_roc)))

# 保存工作空间
save.image(file = file.path(DATA_DIR, "glmnet_complete_workspace.rdata"))

# 输出模型摘要
cat("\n=== 模型摘要 ===\n")
cat(sprintf("模型类型: glmnet (family=binomial)\n"))
cat(sprintf("正则化类型: %s\n", 
            ifelse(alpha_best == 0, "岭回归 (L2)",
                   ifelse(alpha_best == 1, "Lasso (L1)", "弹性网 (L1+L2)"))))
cat(sprintf("alpha参数: %.4f\n", alpha_best))
cat(sprintf("lambda参数: %.6f\n", lambda_best))
cat(sprintf("训练集样本数: %d\n", nrow(train_data_processed)))
cat(sprintf("测试集样本数: %d\n", nrow(test_data_processed)))
cat(sprintf("特征数量: %d\n", ncol(train_data_processed) - 1))
cat(sprintf("非零系数特征数: %d\n", non_zero_coefs))

# 创建性能对比表
performance_comparison <- data.frame(
  Dataset = c("Training", "Test"),
  AUC = c(auc(train_roc), auc(test_roc)),
  Alpha = c(alpha_best, alpha_best),
  Lambda = c(lambda_best, lambda_best)
)

print(performance_comparison)

# 可选：输出非零系数特征
if (non_zero_coefs > 0) {
  cat("\n=== 非零系数特征（按重要性排序）===\n")
  print(coef_df %>% filter(Coefficient != 0) %>% head(20))
}

# 可选：创建特征选择报告
feature_report <- coef_df %>%
  mutate(
    Importance_Rank = rank(-abs(Coefficient)),
    Is_Selected = Coefficient != 0,
    Selection_Type = case_when(
      Is_Selected & Coefficient > 0 ~ "正向选择",
      Is_Selected & Coefficient < 0 ~ "负向选择",
      TRUE ~ "未选择"
    )
  )

write.csv(feature_report, file.path(DATA_DIR, "glmnet_feature_selection_report.csv"), row.names = FALSE)
cat(sprintf("\n特征选择报告已保存至: %s\n", file.path(DATA_DIR, "glmnet_feature_selection_report.csv")))

