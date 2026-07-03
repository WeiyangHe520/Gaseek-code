rm(list = ls())
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
library(glmnetUtils)

# 加载数据
load(file = ".left_data.rdata")

## ========== 统一的数据预处理 ==========
# 直接使用您已经预处理好的数据（假设它们已经存在）
# 如果您还没有执行Yeo-Johnson变换，请取消下面的注释
if (!exists("train_data") || !exists("test_data")) {
  # 使用Yeo-Johnson变换
  train_pre <- preProcess(train_data[, -ncol(train_data)], 
                          method = c("YeoJohnson", "center", "scale"))
  train_data <- predict(train_pre, train_data)
  
  test_pre <- preProcess(test_data[, -ncol(test_data)], 
                         method = c("YeoJohnson", "center", "scale"))
  test_data <- predict(test_pre, test_data)
  
  # 保存预处理参数
  preprocess_params <- train_pre
  save(preprocess_params, file = file.path(DATA_DIR, "preprocess_params.rdata"))
} else {
  # 如果已经预处理过，创建一个虚拟的preprocess_params用于保存
  preprocess_params <- list(method = "YeoJohnson_center_scale", 
                            already_applied = TRUE)
  class(preprocess_params) <- "preProcess"
}

train_data_processed <- train_data
test_data_processed <- test_data

# 确保响应变量是因子（关键！）
train_data_processed$group <- factor(train_data_processed$group, 
                                     levels = c("control", "cancer"))
test_data_processed$group <- factor(test_data_processed$group, 
                                    levels = c("control", "cancer"))

# 检查类别分布
cat("训练集类别分布:\n")
print(table(train_data_processed$group))
cat("\n测试集类别分布:\n")
print(table(test_data_processed$group))

## 准备glmnet需要的矩阵格式
x_train <- as.matrix(train_data_processed[, -ncol(train_data_processed)])
y_train <- train_data_processed$group
x_test <- as.matrix(test_data_processed[, -ncol(test_data_processed)])
y_test <- test_data_processed$group

# 检查数据是否有变化
cat(sprintf("\n训练集维度: %d x %d\n", nrow(x_train), ncol(x_train)))
cat(sprintf("测试集维度: %d x %d\n", nrow(x_test), ncol(x_test)))

## ========== 修复的贝叶斯优化函数 ==========
bayes_opt_glmnet <- function(alpha, lambda_log) {
  alpha_val <- max(min(alpha, 1), 0)
  lambda_val <- max(exp(lambda_log), 1e-6)
  
  tryCatch({
    # 使用5折交叉验证
    cv_fit <- cv.glmnet(
      x = x_train,
      y = y_train,
      family = "binomial",
      alpha = alpha_val,
      lambda = exp(seq(log(0.001), log(10), length.out = 50)),
      nfolds = 5,
      type.measure = "auc",
      standardize = FALSE  # 数据已经标准化
    )
    
    auc_value <- max(cv_fit$cvm, na.rm = TRUE)
    
    # 调试输出
    if (runif(1) < 0.1) {
      cat(sprintf("  alpha=%.3f, lambda=%.6f, AUC=%.4f\n", 
                  alpha_val, lambda_val, auc_value))
    }
    
    list(Score = auc_value, Pred = 0)
  }, error = function(e) {
    cat(sprintf("  错误: alpha=%.3f, lambda=%.6f, %s\n", 
                alpha_val, lambda_val, e$message))
    list(Score = 0.5, Pred = 0)
  })
}

## 执行贝叶斯优化
bounds_glmnet <- list(
  alpha = c(0, 1),
  lambda_log = c(-8, 2)  # exp(-8)到exp(2) = 0.0003到7.39
)

set.seed(123)
cat("\n开始贝叶斯优化...\n")
opt_result_glmnet <- BayesianOptimization(
  FUN = bayes_opt_glmnet,
  bounds = bounds_glmnet,
  init_points = 10,  # 增加初始点
  n_iter = 15,       # 增加迭代次数
  acq = "ucb",
  kappa = 2.576,
  verbose = TRUE
)

## 输出最佳参数
alpha_best <- opt_result_glmnet$Best_Par[["alpha"]]
lambda_best <- exp(opt_result_glmnet$Best_Par[["lambda_log"]])

cat("\n最佳参数组合:\n")
cat(sprintf("alpha: %.4f\n", alpha_best))
cat(sprintf("lambda: %.6f (log-scale: %.3f)\n", 
            lambda_best, opt_result_glmnet$Best_Par[["lambda_log"]]))
cat(sprintf("最佳AUC: %.4f\n", opt_result_glmnet$Best_Value))

## ========== 训练最终模型 ==========
# 方法1：使用glmnet直接训练
final_glmnet <- glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = alpha_best,
  lambda = lambda_best,
  standardize = FALSE
)

# 方法2：使用caret训练（可选）
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

final_model <- tryCatch({
  train(
    x = x_train,
    y = y_train,
    method = "glmnet",
    family = "binomial",
    trControl = ctrl,
    tuneGrid = tune_grid,
    metric = "ROC",
    standardize = FALSE
  )
}, error = function(e) {
  cat("caret训练失败，使用glmnet模型\n")
  return(NULL)
})

## ========== 模型预测和评估 ==========
# 预测概率（注意：返回的是cancer类别的概率）
train_pred <- predict(final_glmnet, newx = x_train, type = "response")[, 1]
test_pred <- predict(final_glmnet, newx = x_test, type = "response")[, 1]

# 确保预测概率不是常数
cat(sprintf("\n训练集预测概率范围: [%.4f, %.4f]\n", 
            min(train_pred), max(train_pred)))
cat(sprintf("测试集预测概率范围: [%.4f, %.4f]\n", 
            min(test_pred), max(test_pred)))

# 如果预测概率都是0.5，说明模型有问题
if (sd(train_pred) < 0.01) {
  cat("\n警告：预测概率几乎没有变化！可能原因：\n")
  cat("1. lambda太大导致模型过于稀疏\n")
  cat("2. 数据中可能没有信号\n")
  cat("3. 类别不平衡严重\n")
  
  # 尝试使用更小的lambda重新训练
  cat("\n尝试使用更小的lambda重新训练...\n")
  final_glmnet <- glmnet(
    x = x_train,
    y = y_train,
    family = "binomial",
    alpha = alpha_best,
    lambda = lambda_best / 100,  # 减小lambda
    standardize = FALSE
  )
  train_pred <- predict(final_glmnet, newx = x_train, type = "response")[, 1]
  test_pred <- predict(final_glmnet, newx = x_test, type = "response")[, 1]
  cat(sprintf("新训练集预测概率范围: [%.4f, %.4f]\n", 
              min(train_pred), max(train_pred)))
}

# ROC评估
positive_class <- "cancer"
train_roc <- roc(response = as.numeric(y_train == positive_class), 
                 predictor = train_pred)
test_roc <- roc(response = as.numeric(y_test == positive_class), 
                predictor = test_pred)

cat("\n最终模型性能:\n")
cat(sprintf("训练集AUC: %.4f (95%% CI: %.4f-%.4f)\n", 
            auc(train_roc),
            ci.auc(train_roc)[1], ci.auc(train_roc)[3]))
cat(sprintf("测试集AUC: %.4f (95%% CI: %.4f-%.4f)\n", 
            auc(test_roc),
            ci.auc(test_roc)[1], ci.auc(test_roc)[3]))

# 如果AUC仍然是0.5，检查是否是标签顺序问题
if (auc(test_roc) < 0.51) {
  cat("\n=== 诊断信息 ===\n")
  cat("AUC接近0.5，可能是标签顺序反了\n")
  # 尝试反向预测
  test_roc_rev <- roc(response = as.numeric(y_test == positive_class), 
                      predictor = 1 - test_pred)
  cat(sprintf("反向预测测试集AUC: %.4f\n", auc(test_roc_rev)))
}

## ========== 保存模型 ==========
save(final_model, final_glmnet, preprocess_params, opt_result_glmnet,
     train_data_processed, test_data_processed,
     file = file.path(DATA_DIR, "final_glmnet_model.rdata"))

## ========== 特征重要性 ==========
coefficients <- coef(final_glmnet)
coef_df <- data.frame(
  Feature = rownames(coefficients),
  Coefficient = as.numeric(coefficients[, 1])
) %>%
  filter(Feature != "(Intercept)") %>%
  arrange(desc(abs(Coefficient)))

non_zero_coefs <- sum(coef_df$Coefficient != 0)
cat(sprintf("\n非零系数特征数: %d / %d (%.1f%%)\n", 
            non_zero_coefs, nrow(coef_df), 
            100 * non_zero_coefs / nrow(coef_df)))

# 输出top特征
if (non_zero_coefs > 0) {
  cat("\nTop 10 重要特征:\n")
  print(head(coef_df, 10))
}

cat("\n=== glmnet模型优化完成！ ===\n")