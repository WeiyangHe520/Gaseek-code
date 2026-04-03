###############################################
## 临床预测模型开发教学代码 - 多模型输出版   ##
## 完整版本（含AUC置信区间）               ##
###############################################

rm(list = ls())
set.seed(278)
FIG_DIR <- "figures_glmnet_fixed/"
DATA_DIR <- "data_glmnet_fixed/"
dir.create(FIG_DIR, showWarnings = FALSE)
dir.create(DATA_DIR, showWarnings = FALSE)

# 加载必要的包
library(caret)
library(tidyverse)
library(viridis)
library(ggprism)
library(pROC)
library(ggplot2)

# 检查并安装缺失的包
required_packages <- c("arm", "klaR", "mboost")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

load(file = ".left_data.rdata")

# 数据预处理和检查
cat("=== 数据预处理 ===\n")

# 确保分组变量名称一致且为因子
colnames(train_data)[ncol(train_data)] <- "class"
colnames(test_data)[ncol(test_data)] <- "class"

# 检查并设置因子水平
cat("训练数据类别分布:\n")
print(table(train_data$class))
cat("测试数据类别分布:\n")
print(table(test_data$class))

# 确保类别为因子且水平正确
train_data$class <- factor(train_data$class, levels = c("control", "cancer"))
test_data$class <- factor(test_data$class, levels = c("control", "cancer"))

cat("处理后的训练数据类别分布:\n")
print(table(train_data$class))
cat("处理后的测试数据类别分布:\n")
print(table(test_data$class))

# 检查特征数据
cat("训练数据特征数量:", ncol(train_data) - 1, "\n")
cat("测试数据特征数量:", ncol(test_data) - 1, "\n")

# 增强的性能指标计算函数（含AUC置信区间）
calculate_simple_metrics <- function(predictions, actual, model_name = "") {
  
  # 确保输入正确
  if (length(predictions) == 0 || length(actual) == 0) {
    cat("错误：预测值或实际值为空 -", model_name, "\n")
    return(NULL)
  }
  
  # 检查预测值范围
  cat(model_name, "预测概率范围:", range(predictions), "\n")
  
  # 预测类别
  pred_class <- ifelse(predictions > 0.5, "cancer", "control")
  pred_class <- factor(pred_class, levels = c("control", "cancer"))
  actual <- factor(actual, levels = c("control", "cancer"))
  
  # 混淆矩阵
  conf_matrix <- table(Predicted = pred_class, Actual = actual)
  cat(model_name, "混淆矩阵:\n")
  print(conf_matrix)
  
  # 提取TP, TN, FP, FN
  TP <- ifelse("cancer" %in% rownames(conf_matrix) && "cancer" %in% colnames(conf_matrix), 
               conf_matrix["cancer", "cancer"], 0)
  TN <- ifelse("control" %in% rownames(conf_matrix) && "control" %in% colnames(conf_matrix), 
               conf_matrix["control", "control"], 0)
  FP <- ifelse("cancer" %in% rownames(conf_matrix) && "control" %in% colnames(conf_matrix), 
               conf_matrix["cancer", "control"], 0)
  FN <- ifelse("control" %in% rownames(conf_matrix) && "cancer" %in% colnames(conf_matrix), 
               conf_matrix["control", "cancer"], 0)
  
  # 计算基本指标
  accuracy <- (TP + TN) / sum(conf_matrix)
  sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
  ppv <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  npv <- ifelse((TN + FN) > 0, TN / (TN + FN), 0)
  
  # AUC 和 95% 置信区间计算
  auc_value <- 0
  auc_ci_lower <- 0
  auc_ci_upper <- 0
  auc_95ci <- "0 (0-0)"
  
  roc_obj <- tryCatch({
    roc(actual, predictions)
  }, error = function(e) {
    cat("计算AUC时出错 -", model_name, ":", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(roc_obj)) {
    auc_value <- auc(roc_obj)
    
    # 计算AUC的95%置信区间 - 使用DeLong方法
    auc_ci <- tryCatch({
      ci.auc(roc_obj, method = "delong")
    }, error = function(e) {
      cat("计算AUC CI时出错 -", model_name, ":", e$message, "\n")
      # 如果DeLong方法失败，尝试bootstrap方法
      cat("尝试使用bootstrap方法计算CI...\n")
      calculate_auc_ci_bootstrap(predictions, actual)
    })
    
    if (length(auc_ci) == 3 && all(!is.na(auc_ci))) {
      auc_ci_lower <- auc_ci[1]
      auc_ci_upper <- auc_ci[3]
      auc_95ci <- paste0(round(auc_value, 3), " (", 
                        round(auc_ci_lower, 3), "-", 
                        round(auc_ci_upper, 3), ")")
    } else {
      # 如果CI计算失败，使用点估计值
      auc_ci_lower <- auc_value
      auc_ci_upper <- auc_value
      auc_95ci <- paste0(round(auc_value, 3), " (NA)")
    }
  }
  
  # Brier score
  actual_binary <- ifelse(actual == "cancer", 1, 0)
  brier_score <- mean((predictions - actual_binary)^2)
  
  # Youden指数
  youden_index <- sensitivity + specificity - 1
  
  # F1分数
  f1_score <- ifelse((sensitivity + ppv) > 0, 2 * (sensitivity * ppv) / (sensitivity + ppv), 0)
  
  result_df <- data.frame(
    AUC = round(auc_value, 4),
    AUC_CI_lower = round(auc_ci_lower, 4),
    AUC_CI_upper = round(auc_ci_upper, 4),
    AUC_95CI = auc_95ci,
    ACC = round(accuracy, 4),
    SENS = round(sensitivity, 4),
    SPEC = round(specificity, 4),
    PPV = round(ppv, 4),
    NPV = round(npv, 4),
    YDI = round(youden_index, 4),
    F1 = round(f1_score, 4),
    BRIER = round(brier_score, 4),
    stringsAsFactors = FALSE
  )
  
  cat(model_name, "性能指标:\n")
  print(result_df)
  cat("AUC 95% CI:", auc_95ci, "\n\n")
  
  return(result_df)
}

# Bootstrap方法计算AUC置信区间（备用方法）
calculate_auc_ci_bootstrap <- function(predictions, actual, n_bootstraps = 1000) {
  cat("使用Bootstrap方法计算AUC置信区间...\n")
  auc_values <- numeric(n_bootstraps)
  n <- length(actual)
  valid_boots <- 0
  
  for (i in 1:n_bootstraps) {
    # 有放回抽样
    indices <- sample(1:n, n, replace = TRUE)
    tryCatch({
      roc_boot <- roc(actual[indices], predictions[indices])
      auc_values[i] <- auc(roc_boot)
      valid_boots <- valid_boots + 1
    }, error = function(e) {
      auc_values[i] <- NA
    })
  }
  
  # 移除NA值
  auc_values <- auc_values[!is.na(auc_values)]
  
  if (length(auc_values) >= 100) {  # 至少100个有效的bootstrap样本
    ci <- quantile(auc_values, c(0.025, 0.975), na.rm = TRUE)
    cat(sprintf("Bootstrap完成: %d/%d 有效样本\n", length(auc_values), n_bootstraps))
    return(c(ci[1], mean(auc_values), ci[2]))  # 返回与ci.auc相同的格式
  } else {
    cat("Bootstrap失败: 有效样本不足\n")
    return(c(NA, NA, NA))
  }
}

# 修复的模型训练和评估函数
evaluate_all_models <- function(train_data, test_data, method_list = c("multinom","gamSpline","lda2","glmStepAIC","mlp","svmRadial","xgbTree","rf","xgbLinear","glmnet")) {
  
  # 设置训练控制参数 - 简化版本
  fit_control <- trainControl(
    method = "cv",
    number = 3,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    verboseIter = FALSE,
    allowParallel = FALSE
  )
  
  all_performance <- list()
  all_predictions <- list()
  
  cat("=== 开始训练和评估所有模型 ===\n")
  
  for (method in method_list) {
    cat("\n--- 训练模型:", method, "---\n")
    
    tryCatch({
      # 根据模型类型设置不同的参数
      if (method == "bayesglm") {
        # 对于bayesglm，使用更简单的配置
        model <- train(
          class ~ .,
          data = train_data,
          method = "bayesglm",
          trControl = fit_control,
          metric = "ROC"
        )
      } else if (method == "nb") {
        # 对于朴素贝叶斯，使用固定参数
        model <- train(
          class ~ .,
          data = train_data,
          method = "nb",
          trControl = fit_control,
          metric = "ROC",
          tuneGrid = data.frame(fL = 0, usekernel = TRUE, adjust = 1)
        )
      } else {
        # 其他模型使用默认配置
        model <- train(
          class ~ .,
          data = train_data,
          method = method,
          trControl = fit_control,
          metric = "ROC",
          tuneLength = 2,
          preProcess = c("center", "scale"),
          verbose = FALSE
        )
      }
      
      # 训练集预测和评估
      cat("计算训练集性能...\n")
      train_pred <- predict(model, train_data, type = "prob")[, "cancer"]
      train_metrics <- calculate_simple_metrics(train_pred, train_data$class, paste(method, "train"))
      
      # 测试集预测和评估
      cat("计算测试集性能...\n")
      test_pred <- predict(model, test_data, type = "prob")[, "cancer"]
      test_metrics <- calculate_simple_metrics(test_pred, test_data$class, paste(method, "test"))
      
      # 存储结果
      all_performance[[method]] <- list(
        model = model,
        train_metrics = train_metrics,
        test_metrics = test_metrics,
        train_predictions = train_pred,
        test_predictions = test_pred
      )
      
      cat("✅", method, "模型完成\n")
      
    }, error = function(e) {
      cat("❌ 训练", method, "模型时出错:", e$message, "\n")
      # 尝试更简单的配置
      tryCatch({
        cat("尝试简化配置训练", method, "...\n")
        simple_control <- trainControl(
          method = "none",
          classProbs = TRUE,
          summaryFunction = twoClassSummary,
          verboseIter = FALSE
        )
        
        model <- train(
          class ~ .,
          data = train_data,
          method = method,
          trControl = simple_control,
          metric = "ROC",
          tuneLength = 1
        )
        
        # 训练集预测和评估
        train_pred <- predict(model, train_data, type = "prob")[, "cancer"]
        train_metrics <- calculate_simple_metrics(train_pred, train_data$class, paste(method, "train"))
        
        # 测试集预测和评估
        test_pred <- predict(model, test_data, type = "prob")[, "cancer"]
        test_metrics <- calculate_simple_metrics(test_pred, test_data$class, paste(method, "test"))
        
        all_performance[[method]] <- list(
          model = model,
          train_metrics = train_metrics,
          test_metrics = test_metrics,
          train_predictions = train_pred,
          test_predictions = test_pred
        )
        
        cat("✅", method, "模型（简化配置）完成\n")
        
      }, error = function(e2) {
        cat("❌ 简化配置也失败:", e2$message, "\n")
        all_performance[[method]] <- list(error = e2$message)
      })
    })
  }
  
  return(all_performance)
}

# 创建综合性能表格
create_comprehensive_performance_table <- function(all_performance) {
  performance_table <- data.frame()
  
  for (model_name in names(all_performance)) {
    model_result <- all_performance[[model_name]]
    
    # 跳过出错的结果
    if (!is.null(model_result$error)) {
      next
    }
    
    # 训练集结果
    if (!is.null(model_result$train_metrics)) {
      train_row <- model_result$train_metrics
      train_row$Model <- model_name
      train_row$Dataset <- "train"
      performance_table <- rbind(performance_table, train_row)
    }
    
    # 测试集结果
    if (!is.null(model_result$test_metrics)) {
      test_row <- model_result$test_metrics
      test_row$Model <- model_name
      test_row$Dataset <- "test"
      performance_table <- rbind(performance_table, test_row)
    }
  }
  
  # 重新排列列顺序
  if (nrow(performance_table) > 0) {
    performance_table <- performance_table %>%
      select(Model, Dataset, everything())
  }
  
  return(performance_table)
}

# 可视化所有模型性能（含置信区间）
plot_all_models_performance <- function(performance_table) {
  plots <- list()
  
  if (nrow(performance_table) == 0) {
    cat("没有可用的性能数据用于绘图\n")
    return(plots)
  }
  
  # AUC比较图（含误差线表示置信区间）
  auc_plot <- performance_table %>%
    ggplot(aes(x = Model, y = AUC, fill = Dataset)) +
    geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
    geom_errorbar(aes(ymin = AUC_CI_lower, ymax = AUC_CI_upper), 
                  position = position_dodge(0.8), width = 0.2) +
    scale_fill_manual(values = c("train" = "#1f77b4", "test" = "#ff7f0e")) +
    labs(title = "所有模型的AUC性能比较（含95%置信区间）",
         x = "模型", y = "AUC") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    geom_text(aes(label = sprintf("%.3f", AUC)), 
              position = position_dodge(0.8), vjust = -0.3, size = 3)
  
  # 精度比较图
  accuracy_plot <- performance_table %>%
    ggplot(aes(x = Model, y = ACC, fill = Dataset)) +
    geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
    scale_fill_manual(values = c("train" = "#1f77b4", "test" = "#ff7f0e")) +
    labs(title = "所有模型的准确率比较",
         x = "模型", y = "准确率") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    geom_text(aes(label = sprintf("%.3f", ACC)), 
              position = position_dodge(0.8), vjust = -0.3, size = 3)
  
  plots$auc_plot <- auc_plot
  plots$accuracy_plot <- accuracy_plot
  
  return(plots)
}

# 生成AUC置信区间汇总表格
create_auc_ci_summary <- function(performance_table) {
  auc_summary <- performance_table %>%
    select(Model, Dataset, AUC, AUC_CI_lower, AUC_CI_upper, AUC_95CI) %>%
    arrange(Dataset, desc(AUC))
  
  return(auc_summary)
}

# 主执行流程
cat("=== 开始多模型比较分析（含AUC置信区间） ===\n")

# 使用更可靠的模型列表
reliable_methods <- c("multinom","gamSpline","lda2","glmStepAIC","mlp","svmRadial","xgbTree","rf","xgbLinear","glmnet")

# 评估所有模型
all_model_results <- evaluate_all_models(
  train_data = train_data,
  test_data = test_data,
  method_list = reliable_methods
)

# 创建综合性能表格
performance_table <- create_comprehensive_performance_table(all_model_results)

cat("=== 所有模型性能总结 ===\n")
if (nrow(performance_table) > 0) {
  print(performance_table)
  
  # 创建AUC置信区间汇总
  auc_ci_summary <- create_auc_ci_summary(performance_table)
  cat("\n=== AUC及95%置信区间汇总 ===\n")
  print(auc_ci_summary)
  
  # 保存性能表格
  write.csv(performance_table, 
            file.path(DATA_DIR, "all_models_performance_with_CI.csv"), 
            row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(auc_ci_summary, 
            file.path(DATA_DIR, "auc_ci_summary.csv"), 
            row.names = FALSE, fileEncoding = "UTF-8")
  
  # 生成图表
  plots <- plot_all_models_performance(performance_table)
  
  # 保存图表
  if (length(plots) > 0) {
    ggsave(file.path(FIG_DIR, "auc_comparison_with_CI.png"), 
           plots$auc_plot, width = 10, height = 6, dpi = 300)
    ggsave(file.path(FIG_DIR, "accuracy_comparison.png"), 
           plots$accuracy_plot, width = 8, height = 6, dpi = 300)
    cat("图表已保存到", FIG_DIR, "目录\n")
  }
} else {
  cat("没有成功的模型训练结果\n")
}

# 显示成功的模型
successful_models <- names(all_model_results)[sapply(all_model_results, function(x) !is.null(x$model))]
cat("成功训练的模型:", successful_models, "\n")

# 输出最佳模型（基于测试集AUC）
if (nrow(performance_table) > 0) {
  test_performance <- performance_table %>% 
    filter(Dataset == "test") %>%
    arrange(desc(AUC))
  
  best_model <- test_performance[1, ]
  cat("\n=== 最佳模型（基于测试集AUC） ===\n")
  cat("模型:", best_model$Model, "\n")
  cat("AUC:", best_model$AUC, "\n")
  cat("AUC 95% CI:", best_model$AUC_95CI, "\n")
  cat("准确率:", best_model$ACC, "\n")
}

cat("\n=== 分析完成 ===\n")

