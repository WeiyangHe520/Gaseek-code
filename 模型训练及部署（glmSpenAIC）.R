rm(list = ls())
###############################################
## glmStepAIC模型简化代码                    ##
## 功能：默认参数训练+可视化+模型部署       ##
## 作者：基于XGBoost代码改编                ##
## 版本：v1.0 (2024-05-25)                  ##
###############################################

set.seed(3456)
FIG_DIR <- "figures_glmStepAIC/" 
DATA_DIR <- "data_glmStepAIC/"

if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR)
if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR)

library(caret)
library(MASS)  # 包含stepAIC函数
library(ggplot2)
library(dplyr)
library(tidyr)
library(pROC)

# 加载数据
load(file = ".left_data(武汉).rdata")

## 数据预处理
preprocess_params <- preProcess(train_data[, -ncol(train_data)], method = c("center", "scale"))
train_data_processed <- predict(preprocess_params, train_data)
test_data_processed <- predict(preprocess_params, test_data)

## 使用默认参数训练glmStepAIC模型
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# glmStepAIC的默认参数
default_params <- data.frame(
  parameter = "none"  # 使用默认参数，无需调优网格
)

cat("使用glmStepAIC默认参数:\n")
cat("使用stepAIC算法进行变量选择，方向为双向\n")
cat("使用AIC准则进行模型选择\n")

final_model <- train(
  group ~ .,
  data = train_data_processed,
  method = "glmStepAIC",
  trControl = ctrl,
  tuneGrid = default_params,
  metric = "ROC",
  trace = FALSE  # 不显示逐步回归的详细过程
)

# 模型预测
train_pred <- predict(final_model, train_data_processed, type = "prob")[, "cancer"]
test_pred <- predict(final_model, test_data_processed, type = "prob")[, "cancer"]

# 性能评估
train_roc <- roc(train_data_processed$group, train_pred)
test_roc <- roc(test_data_processed$group, test_pred)

cat("\n最终模型性能:\n")
cat(sprintf("训练集AUC: %.4f\n", auc(train_roc)))
cat(sprintf("测试集AUC: %.4f\n", auc(test_roc)))

# 显示最终选择的变量
cat("\n最终模型选择的变量数量:\n")
final_vars <- names(coef(final_model$finalModel))
cat(sprintf("从 %d 个初始变量中选择了 %d 个变量\n", 
            ncol(train_data_processed) - 1, length(final_vars) - 1))
cat("选择的变量:", paste(final_vars[-1], collapse = ", "), "\n")

# 保存模型
save(final_model, preprocess_params, default_params,
     file = file.path(DATA_DIR, "final_glmStepAIC_model.rdata"))

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
  labs(title = "glmStepAIC ROC曲线 (默认参数)",
       subtitle = sprintf("训练集AUC: %.4f, 测试集AUC: %.4f", 
                          auc(train_roc), auc(test_roc))) +
  theme_minimal() +
  scale_color_manual(values = c("Training" = "blue", "Test" = "red"))

ggsave(file.path(FIG_DIR, "glmStepAIC_roc_curves.pdf"), p_roc, width = 10, height = 8)

# 2. 预测概率分布
prob_data <- data.frame(
  Dataset = c(rep("Training", length(train_pred)), 
              rep("Test", length(test_pred))),
  Probability = c(train_pred, test_pred),
  TrueLabel = c(as.character(train_data_processed$group), 
                as.character(test_data_processed$group))
)

p_prob <- ggplot(prob_data, aes(x = Probability, fill = TrueLabel)) +
  geom_histogram(alpha = 0.7, position = "identity", bins = 30) +
  facet_wrap(~ Dataset, ncol = 2) +
  labs(title = "glmStepAIC预测概率分布 (默认参数)",
       x = "预测概率 (cancer)",
       y = "频数") +
  theme_minimal() +
  scale_fill_manual(values = c("cancer" = "red", "normal" = "blue"))

ggsave(file.path(FIG_DIR, "glmStepAIC_probability_distribution.pdf"), p_prob, width = 12, height = 6)

# 3. 变量重要性图（修复版本）
tryCatch({
  var_imp <- varImp(final_model, scale = FALSE)
  
  # 检查var_imp对象的结构
  if (!is.null(var_imp)) {
    # 提取变量重要性数据
    if ("importance" %in% names(var_imp)) {
      var_imp_df <- as.data.frame(var_imp$importance)
    } else {
      var_imp_df <- as.data.frame(var_imp)
    }
    
    # 添加变量名
    var_imp_df$Variable <- rownames(var_imp_df)
    
    # 检查列名并选择正确的列
    col_names <- names(var_imp_df)
    importance_col <- ifelse("Overall" %in% col_names, "Overall", 
                             ifelse("cancer" %in% col_names, "cancer", col_names[1]))
    
    # 重命名列以便于使用
    names(var_imp_df)[names(var_imp_df) == importance_col] <- "Importance"
    
    # 绘制变量重要性图
    p_varimp <- ggplot(var_imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(title = "glmStepAIC变量重要性",
           x = "变量",
           y = "重要性") +
      theme_minimal()
    
    ggsave(file.path(FIG_DIR, "glmStepAIC_variable_importance.pdf"), p_varimp, width = 10, height = 8)
    cat("变量重要性图已生成并保存。\n")
  }
}, error = function(e) {
  cat("变量重要性图生成失败:", e$message, "\n")
  # 备选方案：使用系数绝对值作为重要性
  try({
    coef_values <- coef(final_model$finalModel)
    coef_values <- coef_values[-1]  # 移除截距项
    var_imp_df <- data.frame(
      Variable = names(coef_values),
      Importance = abs(coef_values)
    )
    
    p_varimp <- ggplot(var_imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(title = "glmStepAIC变量重要性 (基于系数绝对值)",
           x = "变量", 
           y = "系数绝对值") +
      theme_minimal()
    
    ggsave(file.path(FIG_DIR, "glmStepAIC_variable_importance.pdf"), p_varimp, width = 10, height = 8)
    cat("使用系数绝对值生成了变量重要性图。\n")
  })
})

## 模型部署（如果需要的函数存在）
if (file.exists("smart_caret_deploy.R")) {
  source("smart_caret_deploy.R")
  export_app(final_model, output_dir = "glmStepAIC_app", app_title = "glmStepAIC分类器 (默认参数)")
  deploy_model(final_model, title = 'glmStepAIC分类器 (默认参数)', port = 8890)
} else {
  cat("注意: smart_caret_deploy.R 文件不存在，跳过模型部署部分。\n")
  cat("如需部署，请确保 smart_caret_deploy.R 文件在当前工作目录中。\n")
}

cat("\nglmStepAIC模型训练完成！\n")
cat("使用的参数: 默认stepAIC算法（双向选择）\n")
cat(sprintf("测试集AUC: %.4f\n", auc(test_roc)))
cat(sprintf("从 %d 个变量中选择了 %d 个重要变量\n", 
            ncol(train_data_processed) - 1, length(coef(final_model$finalModel)) - 1))
cat(sprintf("模型已保存至: %s\n", file.path(DATA_DIR, "final_glmStepAIC_model.rdata")))
cat(sprintf("图形已保存至: %s\n", FIG_DIR))

# 保存工作空间
save.image(file = file.path(DATA_DIR, "glmStepAIC_complete_workspace.rdata"))