rm(list = ls())
###############################################
## glmnet模型简化代码（固定参数版）         ##
## 功能：使用指定参数训练+可视化+模型部署  ##
## 作者：基于glmStepAIC代码改编             ##
## 版本：v1.0 (2024-05-25)                  ##
###############################################

set.seed(3456)
FIG_DIR <- "figures_glmnet_fixed/" 
DATA_DIR <- "data_glmnet_fixed/"

if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR)
if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR)

library(caret)
library(glmnet)  # glmnet包
library(ggplot2)
library(dplyr)
library(tidyr)
library(pROC)
library(Matrix)  # 用于稀疏矩阵

# 加载数据
load(file = ".left_data.rdata")

## 设置最佳参数（直接从您提供的参数）
BEST_ALPHA <- 0.5281
BEST_LAMBDA <- 0.004597

cat("使用固定参数训练glmnet模型:\n")
cat(sprintf("Alpha = %.4f (Elastic Net混合参数)\n", BEST_ALPHA))
cat(sprintf("Lambda = %.6f (正则化强度)\n", BEST_LAMBDA))

## 数据预处理
preprocess_params <- preProcess(train_data[, -ncol(train_data)], method = c("center", "scale"))
train_data_processed <- predict(preprocess_params, train_data)
test_data_processed <- predict(preprocess_params, test_data)

## 准备glmnet所需的数据格式
x_train <- model.matrix(~ . - 1, train_data_processed[, -ncol(train_data_processed)])
x_test <- model.matrix(~ . - 1, test_data_processed[, -ncol(test_data_processed)])

# 将因子转换为数值
y_train <- ifelse(train_data_processed$group == "cancer", 1, 0)
y_test <- ifelse(test_data_processed$group == "cancer", 1, 0)

## 使用固定参数直接训练glmnet模型
cat("\n开始训练glmnet模型（使用固定参数）...\n")

# 方法1：直接使用glmnet函数（更简单直接）
glmnet_fit <- glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = BEST_ALPHA,
  lambda = BEST_LAMBDA,
  standardize = FALSE  # 因为我们已经标准化了
)

# 方法2：使用caret训练（为了保持一致性）
ctrl_fixed <- trainControl(
  method = "none",  # 不需要交叉验证，因为参数固定
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# 固定参数网格
fixed_grid <- data.frame(
  alpha = BEST_ALPHA,
  lambda = BEST_LAMBDA
)

glmnet_model <- train(
  x = x_train,
  y = as.factor(ifelse(y_train == 1, "cancer", "normal")),
  method = "glmnet",
  trControl = ctrl_fixed,
  tuneGrid = fixed_grid,
  metric = "ROC",
  family = "binomial"
)

# 提取最终模型
final_model <- glmnet_model$finalModel

# 获取系数
coef_values <- as.matrix(coef(final_model, s = BEST_LAMBDA))

# 模型预测
train_pred_prob <- predict(glmnet_model, newdata = as.data.frame(x_train), type = "prob")[, "cancer"]
test_pred_prob <- predict(glmnet_model, newdata = as.data.frame(x_test), type = "prob")[, "cancer"]

train_pred_class <- predict(glmnet_model, newdata = as.data.frame(x_train))
test_pred_class <- predict(glmnet_model, newdata = as.data.frame(x_test))

# 性能评估
train_roc <- roc(y_train, train_pred_prob)
test_roc <- roc(y_test, test_pred_prob)

# 计算准确率等其他指标
train_cm <- confusionMatrix(train_pred_class, as.factor(ifelse(y_train == 1, "cancer", "normal")))
test_cm <- confusionMatrix(test_pred_class, as.factor(ifelse(y_test == 1, "cancer", "normal")))

cat("\n最终模型性能:\n")
cat(sprintf("训练集AUC: %.4f\n", auc(train_roc)))
cat(sprintf("测试集AUC: %.4f\n", auc(test_roc)))
cat(sprintf("训练集准确率: %.4f\n", train_cm$overall["Accuracy"]))
cat(sprintf("测试集准确率: %.4f\n", test_cm$overall["Accuracy"]))
cat(sprintf("训练集灵敏度: %.4f\n", train_cm$byClass["Sensitivity"]))
cat(sprintf("训练集特异度: %.4f\n", train_cm$byClass["Specificity"]))
cat(sprintf("测试集灵敏度: %.4f\n", test_cm$byClass["Sensitivity"]))
cat(sprintf("测试集特异度: %.4f\n", test_cm$byClass["Specificity"]))

# 显示选择的变量
non_zero_coef <- coef_values[coef_values != 0, ]
selected_vars <- names(non_zero_coef)[-1]  # 排除截距项

cat("\n变量选择结果:\n")
cat(sprintf("从 %d 个初始变量中选择了 %d 个非零系数变量\n", 
            ncol(x_train), length(selected_vars)))

if (length(selected_vars) > 0) {
  # 显示前20个最重要的变量
  coef_df <- data.frame(
    Variable = names(non_zero_coef)[-1],
    Coefficient = non_zero_coef[-1]
  ) %>% 
    mutate(Absolute = abs(Coefficient)) %>%
    arrange(desc(Absolute))
  
  cat("\n变量重要性排序（前20个）:\n")
  print(head(coef_df, 20))
  
  cat("\n所有选择的变量:\n")
  cat(paste(selected_vars, collapse = ", "), "\n")
  
  # 系数统计
  cat("\n系数统计:\n")
  cat(sprintf("正系数数量: %d\n", sum(coef_df$Coefficient > 0)))
  cat(sprintf("负系数数量: %d\n", sum(coef_df$Coefficient < 0)))
  cat(sprintf("最大系数绝对值: %.4f\n", max(abs(coef_df$Coefficient))))
  cat(sprintf("平均系数绝对值: %.4f\n", mean(abs(coef_df$Coefficient))))
} else {
  cat("警告：没有选择任何变量！\n")
}

# 保存完整模型对象
save(glmnet_model, preprocess_params, fixed_grid, glmnet_fit,
     file = file.path(DATA_DIR, "final_glmnet_fixed_model.rdata"))

## 性能可视化
# 1. ROC曲线
roc_data <- data.frame(
  Dataset = c(rep("Training", length(train_roc$sensitivities)), 
              rep("Test", length(test_roc$sensitivities))),
  Sensitivity = c(train_roc$sensitivities, test_roc$sensitivities),
  Specificity = c(1 - train_roc$specificities, 1 - test_roc$specificities)
)

p_roc <- ggplot(roc_data, aes(x = Specificity, y = Sensitivity, color = Dataset)) +
  geom_line(size = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(title = sprintf("glmnet ROC曲线 (alpha=%.4f, lambda=%.6f)", BEST_ALPHA, BEST_LAMBDA),
       subtitle = sprintf("训练集AUC: %.4f, 测试集AUC: %.4f", 
                          auc(train_roc), auc(test_roc))) +
  theme_minimal() +
  scale_color_manual(values = c("Training" = "blue", "Test" = "red")) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave(file.path(FIG_DIR, "glmnet_roc_curves.pdf"), p_roc, width = 10, height = 8)
ggsave(file.path(FIG_DIR, "glmnet_roc_curves.png"), p_roc, width = 10, height = 8, dpi = 300)

# 2. 预测概率分布
prob_data <- data.frame(
  Dataset = c(rep("Training", length(train_pred_prob)), 
              rep("Test", length(test_pred_prob))),
  Probability = c(train_pred_prob, test_pred_prob),
  TrueLabel = c(ifelse(y_train == 1, "cancer", "normal"), 
                ifelse(y_test == 1, "cancer", "normal"))
)

p_prob <- ggplot(prob_data, aes(x = Probability, fill = TrueLabel)) +
  geom_histogram(alpha = 0.7, position = "identity", bins = 30) +
  facet_wrap(~ Dataset, ncol = 2) +
  labs(title = sprintf("glmnet预测概率分布 (alpha=%.4f)", BEST_ALPHA),
       x = "预测概率 (cancer)",
       y = "频数") +
  theme_minimal() +
  scale_fill_manual(values = c("cancer" = "red", "normal" = "blue")) +
  theme(plot.title = element_text(hjust = 0.5))

ggsave(file.path(FIG_DIR, "glmnet_probability_distribution.pdf"), p_prob, width = 12, height = 6)
ggsave(file.path(FIG_DIR, "glmnet_probability_distribution.png"), p_prob, width = 12, height = 6, dpi = 300)

# 3. 变量系数图（按重要性排序）
if (length(selected_vars) > 0) {
  # 准备系数数据
  coef_plot_data <- coef_df %>%
    arrange(desc(Absolute)) %>%
    head(30)  # 只显示前30个最重要的变量
  
  p_coef <- ggplot(coef_plot_data, aes(x = reorder(Variable, Coefficient), y = Coefficient, 
                                       fill = ifelse(Coefficient > 0, "Positive", "Negative"))) +
    geom_col() +
    coord_flip() +
    labs(title = sprintf("glmnet模型系数 (alpha=%.4f, lambda=%.6f)", BEST_ALPHA, BEST_LAMBDA),
         x = "变量",
         y = "系数值",
         subtitle = sprintf("共选择 %d 个变量，显示前 %d 个", nrow(coef_df), nrow(coef_plot_data))) +
    theme_minimal() +
    scale_fill_manual(name = "系数符号", 
                      values = c("Positive" = "red", "Negative" = "blue")) +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5),
          legend.position = "top")
  
  # 动态调整图形高度
  plot_height <- max(6, nrow(coef_plot_data) * 0.25)
  ggsave(file.path(FIG_DIR, "glmnet_coefficients.pdf"), p_coef, width = 10, height = plot_height)
  ggsave(file.path(FIG_DIR, "glmnet_coefficients.png"), p_coef, width = 10, height = plot_height, dpi = 300)
  
  # 4. 系数绝对值条形图
  p_abs_coef <- ggplot(coef_plot_data, aes(x = reorder(Variable, Absolute), y = Absolute)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = "glmnet变量重要性（系数绝对值）",
         x = "变量",
         y = "系数绝对值") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  ggsave(file.path(FIG_DIR, "glmnet_variable_importance.pdf"), p_abs_coef, width = 10, height = plot_height)
  ggsave(file.path(FIG_DIR, "glmnet_variable_importance.png"), p_abs_coef, width = 10, height = plot_height, dpi = 300)
}

# 5. 混淆矩阵热图
create_confusion_matrix_plot <- function(cm, title) {
  cm_df <- as.data.frame(cm$table)
  
  ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "white", size = 6) +
    scale_fill_gradient(low = "blue", high = "red") +
    labs(title = title,
         x = "真实类别",
         y = "预测类别",
         fill = "频数") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}

p_train_cm <- create_confusion_matrix_plot(train_cm, "训练集混淆矩阵")
p_test_cm <- create_confusion_matrix_plot(test_cm, "测试集混淆矩阵")

ggsave(file.path(FIG_DIR, "glmnet_train_confusion_matrix.pdf"), p_train_cm, width = 8, height = 6)
ggsave(file.path(FIG_DIR, "glmnet_test_confusion_matrix.pdf"), p_test_cm, width = 8, height = 6)
ggsave(file.path(FIG_DIR, "glmnet_train_confusion_matrix.png"), p_train_cm, width = 8, height = 6, dpi = 300)
ggsave(file.path(FIG_DIR, "glmnet_test_confusion_matrix.png"), p_test_cm, width = 8, height = 6, dpi = 300)

# 6. 性能摘要表格（保存为CSV）
performance_summary <- data.frame(
  Metric = c("Alpha", "Lambda", 
             "Train_AUC", "Test_AUC",
             "Train_Accuracy", "Test_Accuracy",
             "Train_Sensitivity", "Test_Sensitivity",
             "Train_Specificity", "Test_Specificity",
             "Selected_Variables", "Total_Variables",
             "Positive_Coefficients", "Negative_Coefficients"),
  Value = c(BEST_ALPHA, BEST_LAMBDA,
            auc(train_roc), auc(test_roc),
            train_cm$overall["Accuracy"], test_cm$overall["Accuracy"],
            train_cm$byClass["Sensitivity"], test_cm$byClass["Sensitivity"],
            train_cm$byClass["Specificity"], test_cm$byClass["Specificity"],
            length(selected_vars), ncol(x_train),
            sum(coef_df$Coefficient > 0), sum(coef_df$Coefficient < 0))
)

write.csv(performance_summary, file.path(DATA_DIR, "glmnet_performance_summary.csv"), row.names = FALSE)

# 保存系数表格
if (exists("coef_df")) {
  write.csv(coef_df, file.path(DATA_DIR, "glmnet_coefficients.csv"), row.names = FALSE)
}

## 模型部署函数
create_glmnet_predictor <- function(model, preprocessor, lambda = BEST_LAMBDA) {
  function(new_data) {
    # 预处理新数据
    new_data_processed <- predict(preprocessor, new_data)
    
    # 转换为矩阵格式
    x_new <- model.matrix(~ . - 1, new_data_processed)
    
    # 预测概率
    predictions <- predict(model, newdata = as.data.frame(x_new), type = "prob")
    
    # 返回癌症概率
    return(predictions[, "cancer"])
  }
}

# 创建预测函数
glmnet_predictor <- create_glmnet_predictor(glmnet_model, preprocess_params)

# 测试预测函数
cat("\n测试预测函数...\n")
sample_pred <- glmnet_predictor(test_data[1:5, -ncol(test_data)])
cat("前5个样本的预测概率:\n")
print(sample_pred)

# 保存预测函数
save(glmnet_predictor, file = file.path(DATA_DIR, "glmnet_predictor.rdata"))

## 模型部署（如果需要的函数存在）
if (file.exists("smart_caret_deploy.R")) {
  source("smart_caret_deploy.R")
  export_app(glmnet_model, output_dir = "glmnet_fixed_app", 
             app_title = sprintf("glmnet分类器 (alpha=%.4f)", BEST_ALPHA))
  deploy_model(glmnet_model, 
               title = sprintf('glmnet分类器 (alpha=%.4f, lambda=%.6f)', BEST_ALPHA, BEST_LAMBDA), 
               port = 8892)
} else {
  cat("注意: smart_caret_deploy.R 文件不存在，跳过模型部署部分。\n")
  cat("如需部署，请确保 smart_caret_deploy.R 文件在当前工作目录中。\n")
}

# 创建简单的部署函数
create_simple_app <- function() {
  cat("\n创建简单的预测脚本...\n")
  
  app_code <- sprintf('
# glmnet预测脚本
# 使用方法：source("glmnet_predict.R")
# 然后使用 predict_glmnet(new_data) 进行预测

# 加载必要的库
library(caret)
library(glmnet)

# 加载模型和预处理参数
load("%s")
load("%s")

# 预测函数
predict_glmnet <- function(new_data) {
  # 预处理
  new_data_processed <- predict(preprocess_params, new_data)
  
  # 转换为矩阵
  x_new <- model.matrix(~ . - 1, new_data_processed)
  
  # 预测
  predictions <- predict(glmnet_model, newdata = as.data.frame(x_new), type = "prob")
  
  # 返回结果
  result <- data.frame(
    Sample = rownames(new_data),
    Probability_cancer = predictions[, "cancer"],
    Probability_normal = predictions[, "normal"],
    Prediction = ifelse(predictions[, "cancer"] > 0.5, "cancer", "normal")
  )
  
  return(result)
}

cat("glmnet预测函数已加载。\\n")
cat("参数：alpha=%.4f, lambda=%.6f\\n")
cat("使用 predict_glmnet(new_data) 进行预测\\n")
',
file.path(DATA_DIR, "final_glmnet_fixed_model.rdata"),
file.path(DATA_DIR, "glmnet_predictor.rdata"),
BEST_ALPHA, BEST_LAMBDA)
  
  writeLines(app_code, file.path(DATA_DIR, "glmnet_predict.R"))
  cat(sprintf("预测脚本已保存至: %s\n", file.path(DATA_DIR, "glmnet_predict.R")))
}

create_simple_app()

cat("\n")
cat(rep("=", 60), "\n")
cat("glmnet模型训练完成！\n")
cat(rep("=", 60), "\n")
cat(sprintf("模型类型: Elastic Net回归 (alpha=%.4f)\n", BEST_ALPHA))
cat(sprintf("正则化参数: lambda=%.6f\n", BEST_LAMBDA))
cat(sprintf("测试集AUC: %.4f\n", auc(test_roc)))
cat(sprintf("测试集准确率: %.4f\n", test_cm$overall["Accuracy"]))
cat(sprintf("变量选择: 从 %d 个变量中选择了 %d 个\n", 
            ncol(x_train), length(selected_vars)))
cat(sprintf("模型已保存至: %s\n", file.path(DATA_DIR, "final_glmnet_fixed_model.rdata")))
cat(sprintf("预测函数已保存至: %s\n", file.path(DATA_DIR, "glmnet_predictor.rdata")))
cat(sprintf("预测脚本已保存至: %s\n", file.path(DATA_DIR, "glmnet_predict.R")))
cat(sprintf("性能摘要已保存至: %s\n", file.path(DATA_DIR, "glmnet_performance_summary.csv")))
cat(sprintf("系数表格已保存至: %s\n", file.path(DATA_DIR, "glmnet_coefficients.csv")))
cat(sprintf("图形已保存至: %s\n", FIG_DIR))
cat(rep("=", 60), "\n")

# 保存工作空间
save.image(file = file.path(DATA_DIR, "glmnet_fixed_complete_workspace.rdata"))

# 输出完成信息
cat("所有任务已完成！\n")

