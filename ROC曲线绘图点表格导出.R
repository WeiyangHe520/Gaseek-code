###############################################
## 临床预测模型开发教学代码 - 多模型输出版   ##
## 简化版本 - 只保留glmnet模型             ##
###############################################

rm(list = ls())
set.seed(278)
FIG_DIR <- "figures（肿瘤标志物）/"
DATA_DIR <- "data（肿瘤标志物）/"
dir.create(FIG_DIR, showWarnings = FALSE)
dir.create(DATA_DIR, showWarnings = FALSE)

# 加载必要的包
library(caret)
library(tidyverse)
library(viridis)
library(ggprism)
library(pROC)
library(ggplot2)
library(gridExtra)
library(glmnet)
library(pracma)  # 添加pracma包用于数值积分

# 检查并安装缺失的包
required_packages <- c("arm", "klaR", "mboost", "pracma")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# 加载数据
load(file = ".left_data.rdata")

# 数据预处理和检查
cat("=== 数据预处理 ===\n")

# 重命名分组变量为label（为了与函数兼容）
colnames(train_data)[ncol(train_data)] <- "label"
colnames(test_data)[ncol(test_data)] <- "label"

# 检查并设置因子水平
cat("训练数据类别分布:\n")
print(table(train_data$label))
cat("测试数据类别分布:\n")
print(table(test_data$label))

# 确保类别为因子且水平正确（根据你的数据实际情况调整）
# 这里假设你的分组是"control"和"cancer"，如果不是请修改
train_data$label <- factor(train_data$label, levels = c("control", "cancer"))
test_data$label <- factor(test_data$label, levels = c("control", "cancer"))

cat("处理后的训练数据类别分布:\n")
print(table(train_data$label))
cat("处理后的测试数据类别分布:\n")
print(table(test_data$label))

# 检查特征数据
cat("训练数据特征数量:", ncol(train_data) - 1, "\n")
cat("测试数据特征数量:", ncol(test_data) - 1, "\n")

# 简化的性能指标计算函数
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
  
  # AUC
  roc_obj <- tryCatch({
    roc(actual, predictions)
  }, error = function(e) {
    cat("计算AUC时出错 -", model_name, ":", e$message, "\n")
    return(NULL)
  })
  
  auc_value <- ifelse(!is.null(roc_obj), auc(roc_obj), 0)
  
  # Brier score
  actual_binary <- ifelse(actual == "cancer", 1, 0)
  brier_score <- mean((predictions - actual_binary)^2)
  
  # Youden指数
  youden_index <- sensitivity + specificity - 1
  
  # F1分数
  f1_score <- ifelse((sensitivity + ppv) > 0, 2 * (sensitivity * ppv) / (sensitivity + ppv), 0)
  
  result_df <- data.frame(
    AUC = round(auc_value, 4),
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
  cat("\n")
  
  return(list(metrics = result_df, roc_obj = roc_obj, conf_matrix = conf_matrix))
}

# 修复的模型训练和评估函数 - 支持自定义glmnet超参数
train_and_evaluate_glmnet_custom <- function(train_data, test_data, 
                                             alpha_value = 0.5281, 
                                             lambda_value = 0.004597,
                                             standardize = FALSE,
                                             intercept = TRUE,
                                             thresh = 1e-7,
                                             maxit = 1000,
                                             family = "binomial") {
  
  cat("=== 开始glmnet模型分析 ===\n")
  
  # 显示参数配置
  cat("glmnet参数配置:\n")
  cat("alpha =", alpha_value, "(L1/L2混合参数，0=ridge，1=lasso)\n")
  cat("lambda =", lambda_value, "(正则化强度)\n")
  cat("standardize =", standardize, "(是否标准化特征)\n")
  cat("intercept =", intercept, "(是否包含截距项)\n")
  cat("thresh =", thresh, "(收敛阈值)\n")
  cat("maxit =", maxit, "(最大迭代次数)\n")
  cat("family =", family, "(分布类型)\n\n")
  
  # 1. 准备数据矩阵
  cat("1. 准备数据矩阵...\n")
  
  # 提取特征和目标变量
  x_train <- as.matrix(train_data[, -which(colnames(train_data) == "label")])
  y_train <- train_data$label
  
  x_test <- as.matrix(test_data[, -which(colnames(test_data) == "label")])
  y_test <- test_data$label
  
  # 将标签转换为二元数值 (0/1)
  y_train_binary <- ifelse(y_train == "cancer", 1, 0)
  y_test_binary <- ifelse(y_test == "cancer", 1, 0)
  
  # 2. 训练glmnet模型（直接使用glmnet包）
  cat("2. 训练glmnet模型...\n")
  set.seed(123)
  
  glmnet_model <- glmnet(
    x = x_train,
    y = y_train_binary,
    family = family,
    alpha = alpha_value,
    lambda = lambda_value,
    standardize = standardize,
    intercept = intercept,
    thresh = thresh,
    maxit = maxit
  )
  
  # 3. 进行预测
  cat("3. 进行预测...\n")
  
  # 训练集预测
  train_pred_prob_raw <- predict(glmnet_model, newx = x_train, type = "response")
  train_pred_prob <- as.numeric(train_pred_prob_raw)
  
  # 测试集预测
  test_pred_prob_raw <- predict(glmnet_model, newx = x_test, type = "response")
  test_pred_prob <- as.numeric(test_pred_prob_raw)
  
  # 预测类别
  train_pred_class <- ifelse(train_pred_prob > 0.5, "cancer", "control")
  test_pred_class <- ifelse(test_pred_prob > 0.5, "cancer", "control")
  
  # 4. 计算性能指标 - 使用增强的ROC计算
  cat("4. 计算性能指标...\n")
  
  # 增强的ROC计算函数
  calculate_roc_with_more_points <- function(predictions, actual, model_name) {
    # 确保输入正确
    if (length(predictions) == 0 || length(actual) == 0) {
      cat("错误：预测值或实际值为空 -", model_name, "\n")
      return(NULL)
    }
    
    # 将actual转换为因子
    actual <- factor(actual, levels = c("control", "cancer"))
    
    # 计算ROC - 使用更多点
    roc_obj <- tryCatch({
      roc(actual, predictions, 
          algorithm = 2,  # 使用更精确的算法
          percent = FALSE,
          smooth = FALSE,  # 先不使用平滑
          auc = TRUE,
          ci = FALSE)
    }, error = function(e) {
      cat("计算ROC时出错 -", model_name, ":", e$message, "\n")
      return(NULL)
    })
    
    return(roc_obj)
  }
  
  # 计算训练集ROC
  train_roc_obj <- calculate_roc_with_more_points(train_pred_prob, y_train, "训练集")
  
  # 计算测试集ROC
  test_roc_obj <- calculate_roc_with_more_points(test_pred_prob, y_test, "测试集")
  
  # 检查测试集ROC是否有足够的数据点
  if (!is.null(test_roc_obj) && length(test_roc_obj$thresholds) <= 5) {
    cat("测试集ROC曲线点太少 (", length(test_roc_obj$thresholds), "个点)，尝试手动生成更多点...\n")
    
    # 创建一个更密集的阈值序列
    thresholds <- seq(0, 1, length.out = 101)
    
    # 手动计算每个阈值下的敏感性和特异性
    sensitivities <- numeric(length(thresholds))
    specificities <- numeric(length(thresholds))
    
    actual_binary <- ifelse(y_test == "cancer", 1, 0)
    
    for (i in seq_along(thresholds)) {
      pred_class <- ifelse(test_pred_prob > thresholds[i], 1, 0)
      
      # 计算混淆矩阵
      TP <- sum(pred_class == 1 & actual_binary == 1)
      TN <- sum(pred_class == 0 & actual_binary == 0)
      FP <- sum(pred_class == 1 & actual_binary == 0)
      FN <- sum(pred_class == 0 & actual_binary == 1)
      
      sensitivities[i] <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
      specificities[i] <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
    }
    
    # 创建新的ROC对象
    test_roc_obj <- list(
      sensitivities = sensitivities,
      specificities = specificities,
      thresholds = thresholds,
      auc = pracma::trapz(1 - specificities, sensitivities)  # 使用数值积分计算AUC
    )
    
    cat("手动生成的测试集ROC曲线有", length(thresholds), "个点\n")
  }
  
  # 5. 计算性能指标（使用原有的calculate_simple_metrics函数）
  train_metrics_result <- calculate_simple_metrics(
    predictions = train_pred_prob, 
    actual = y_train, 
    model_name = "训练集"
  )
  
  test_metrics_result <- calculate_simple_metrics(
    predictions = test_pred_prob, 
    actual = y_test, 
    model_name = "测试集"
  )
  
  # 6. 创建性能表格
  performance_table <- data.frame(
    Dataset = c("Training", "Test"),
    AUC = c(train_metrics_result$metrics$AUC, test_metrics_result$metrics$AUC),
    Accuracy = c(train_metrics_result$metrics$ACC, test_metrics_result$metrics$ACC),
    Sensitivity = c(train_metrics_result$metrics$SENS, test_metrics_result$metrics$SENS),
    Specificity = c(train_metrics_result$metrics$SPEC, test_metrics_result$metrics$SPEC),
    PPV = c(train_metrics_result$metrics$PPV, test_metrics_result$metrics$PPV),
    NPV = c(train_metrics_result$metrics$NPV, test_metrics_result$metrics$NPV),
    Youden_Index = c(train_metrics_result$metrics$YDI, test_metrics_result$metrics$YDI),
    F1_Score = c(train_metrics_result$metrics$F1, test_metrics_result$metrics$F1),
    Brier_Score = c(train_metrics_result$metrics$BRIER, test_metrics_result$metrics$BRIER),
    stringsAsFactors = FALSE
  )
  
  # 7. 准备ROC数据
  prepare_roc_data <- function(roc_obj, dataset_name) {
    if (is.null(roc_obj)) return(NULL)
    
    # 检查是否有sensitivities和specificities
    if (is.null(roc_obj$sensitivities) || is.null(roc_obj$specificities)) {
      cat("警告：", dataset_name, "ROC对象缺少必要数据\n")
      return(NULL)
    }
    
    # 确保长度一致
    n_points <- min(length(roc_obj$sensitivities), 
                    length(roc_obj$specificities),
                    if(!is.null(roc_obj$thresholds)) length(roc_obj$thresholds) else length(roc_obj$sensitivities))
    
    if (n_points <= 1) {
      cat("警告：", dataset_name, "ROC曲线只有", n_points, "个点\n")
      return(NULL)
    }
    
    curve_data <- data.frame(
      FPR = 1 - roc_obj$specificities[1:n_points],
      TPR = roc_obj$sensitivities[1:n_points]
    )
    
    # 添加阈值列（如果有）
    if (!is.null(roc_obj$thresholds) && length(roc_obj$thresholds) >= n_points) {
      curve_data$Threshold <- roc_obj$thresholds[1:n_points]
    }
    
    # 计算AUC
    auc_value <- if(!is.null(roc_obj$auc)) {
      roc_obj$auc
    } else {
      # 手动计算AUC
      sorted_idx <- order(curve_data$FPR)
      x <- curve_data$FPR[sorted_idx]
      y <- curve_data$TPR[sorted_idx]
      sum(diff(x) * (y[-1] + y[-length(y)]) / 2)
    }
    
    return(list(
      curve_data = curve_data,
      auc_value = auc_value
    ))
  }
  
  # 准备ROC数据
  train_roc_data <- prepare_roc_data(train_roc_obj, "train")
  test_roc_data <- prepare_roc_data(test_roc_obj, "test")
  
  # 8. 创建ROC曲线图的函数
  plot_single_roc <- function(roc_data, title, color = "#1f77b4") {
    if (is.null(roc_data)) return(NULL)
    
    p <- ggplot(roc_data$curve_data, aes(x = FPR, y = TPR)) +
      geom_line(color = color, size = 1.5) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", 
                  color = "gray50", size = 0.8) +
      
      scale_x_continuous(
        name = "1 - Specificity",
        limits = c(0, 1),
        breaks = seq(0, 1, 0.2),
        expand = expansion(mult = c(0.02, 0.02))
      ) +
      scale_y_continuous(
        name = "Sensitivity",
        limits = c(0, 1),
        breaks = seq(0, 1, 0.2),
        expand = expansion(mult = c(0.02, 0.02))
      ) +
      
      labs(
        title = title,
        subtitle = paste0("AUC = ", round(roc_data$auc_value, 3))
      ) +
      
      theme_minimal(base_size = 12) +
      theme(
        legend.position = "none",
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 10),
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey30"),
        plot.margin = margin(15, 15, 15, 15),
        aspect.ratio = 1
      ) +
      
      annotate("text", x = 0.7, y = 0.2, 
               label = paste0("AUC = ", round(roc_data$auc_value, 3)),
               size = 5, fontface = "bold", color = color) +
      
      coord_fixed(ratio = 1)
    
    return(p)
  }
  
  # 9. 创建对比ROC曲线函数
  plot_combined_roc <- function(train_data, test_data) {
    if (is.null(train_data) || is.null(test_data)) return(NULL)
    
    # 准备数据
    train_curve <- train_data$curve_data
    train_auc_label <- paste0("Train (AUC = ", round(train_data$auc_value, 3), ")")
    train_curve$Dataset <- train_auc_label
    
    test_curve <- test_data$curve_data
    test_auc_label <- paste0("Test (AUC = ", round(test_data$auc_value, 3), ")")
    test_curve$Dataset <- test_auc_label
    
    plot_data <- rbind(train_curve, test_curve)
    
    # 创建颜色映射
    color_mapping <- c(
      "#1f77b4",  # 蓝色
      "#ff7f0e"   # 橙色
    )
    names(color_mapping) <- c(train_auc_label, test_auc_label)
    
    p <- ggplot(plot_data, aes(x = FPR, y = TPR, color = Dataset)) +
      geom_line(size = 1.2) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", 
                  color = "gray50", size = 0.8) +
      
      scale_color_manual(name = "Dataset", values = color_mapping) +
      
      scale_x_continuous(
        name = "1 - Specificity",
        limits = c(0, 1),
        breaks = seq(0, 1, 0.2),
        expand = expansion(mult = c(0.02, 0.02))
      ) +
      scale_y_continuous(
        name = "Sensitivity",
        limits = c(0, 1),
        breaks = seq(0, 1, 0.2),
        expand = expansion(mult = c(0.02, 0.02))
      ) +
      
      labs(
        title = "Training vs Test ROC Curves Comparison",
        subtitle = paste0("glmnet (alpha=", alpha_value, ", lambda=", lambda_value, ")")
      ) +
      
      theme_minimal(base_size = 12) +
      theme(
        legend.position = c(0.7, 0.25),
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 10),
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey30"),
        plot.margin = margin(15, 15, 15, 15),
        aspect.ratio = 1
      ) +
      coord_fixed(ratio = 1)
    
    return(p)
  }
  
  # 10. 创建性能对比图
  plot_performance_comparison <- function(performance_table, alpha_value, lambda_value) {
    # 创建AUC对比图
    auc_plot <- ggplot(performance_table, aes(x = Dataset, y = AUC, fill = Dataset)) +
      geom_bar(stat = "identity", width = 0.6) +
      geom_text(aes(label = sprintf("%.3f", AUC)), vjust = -0.5, size = 4) +
      scale_fill_manual(values = c("#1f77b4", "#ff7f0e")) +
      labs(title = paste0("AUC Comparison (alpha=", alpha_value, ", lambda=", lambda_value, ")"), 
           x = "", y = "AUC") +
      theme_minimal() +
      theme(legend.position = "none")
    
    # 创建准确率对比图
    acc_plot <- ggplot(performance_table, aes(x = Dataset, y = Accuracy, fill = Dataset)) +
      geom_bar(stat = "identity", width = 0.6) +
      geom_text(aes(label = sprintf("%.3f", Accuracy)), vjust = -0.5, size = 4) +
      scale_fill_manual(values = c("#1f77b4", "#ff7f0e")) +
      labs(title = "Accuracy Comparison", x = "", y = "Accuracy") +
      theme_minimal() +
      theme(legend.position = "none")
    
    return(list(auc_plot = auc_plot, accuracy_plot = acc_plot))
  }
  
  # 11. 生成图形
  cat("5. 生成图形...\n")
  
  # 单个ROC曲线
  train_roc_plot <- plot_single_roc(train_roc_data, "Training Set ROC Curve", "#1f77b4")
  test_roc_plot <- plot_single_roc(test_roc_data, "Test Set ROC Curve", "#ff7f0e")
  
  # 对比ROC曲线
  combined_roc_plot <- plot_combined_roc(train_roc_data, test_roc_data)
  
  # 性能对比图
  performance_plots <- plot_performance_comparison(performance_table, alpha_value, lambda_value)
  
  # 12. 保存数据
  cat("6. 保存结果...\n")
  
  # 保存性能表格
  param_str <- paste0("alpha", round(alpha_value, 4), "_lambda", round(lambda_value, 6))
  param_str <- gsub("\\.", "_", param_str)  # 替换点号为下划线
  
  performance_filename <- paste0("glmnet_performance_table_", param_str, ".csv")
  write.csv(performance_table, file.path(DATA_DIR, performance_filename), 
            row.names = FALSE, fileEncoding = "UTF-8")
  
  # 保存ROC数据（如果有）
  if (!is.null(train_roc_data)) {
    train_roc_filename <- paste0("glmnet_train_roc_curve_points_", param_str, ".csv")
    write.csv(train_roc_data$curve_data, 
              file.path(DATA_DIR, train_roc_filename), 
              row.names = FALSE, fileEncoding = "UTF-8")
  }
  
  if (!is.null(test_roc_data)) {
    test_roc_filename <- paste0("glmnet_test_roc_curve_points_", param_str, ".csv")
    write.csv(test_roc_data$curve_data, 
              file.path(DATA_DIR, test_roc_filename), 
              row.names = FALSE, fileEncoding = "UTF-8")
  }
  
  # 保存合并的ROC数据（如果都有）
  # 跳过合并的ROC数据保存（因为点数不同）
  cat("注意：训练集和测试集ROC曲线点数不同，跳过合并数据保存\n")
  cat("训练集点数:", if(!is.null(train_roc_data)) nrow(train_roc_data$curve_data) else "NULL", "\n")
  cat("测试集点数:", if(!is.null(test_roc_data)) nrow(test_roc_data$curve_data) else "NULL", "\n")
  
  # 改为只保存比较汇总
  if (!is.null(train_roc_data) && !is.null(test_roc_data)) {
    comparison_summary <- data.frame(
      Dataset = c("train", "test"),
      AUC = c(train_roc_data$auc_value, test_roc_data$auc_value),
      Points = c(nrow(train_roc_data$curve_data), nrow(test_roc_data$curve_data)),
      stringsAsFactors = FALSE
    )
    
    summary_filename <- paste0("glmnet_roc_comparison_summary_", param_str, ".csv")
    write.csv(comparison_summary, 
              file.path(DATA_DIR, summary_filename), 
              row.names = FALSE, fileEncoding = "UTF-8")
  }
  # 13. 保存图形
  # 保存性能对比图
  if (length(performance_plots) > 0) {
    combined_performance_plot <- grid.arrange(
      performance_plots$auc_plot,
      performance_plots$accuracy_plot,
      ncol = 2,
      top = paste0("glmnet模型性能分析 (alpha=", alpha_value, ", lambda=", lambda_value, ")")
    )
    
    performance_plot_filename <- paste0("glmnet_performance_", param_str, ".png")
    ggsave(file.path(FIG_DIR, performance_plot_filename),
           combined_performance_plot, 
           width = 10, 
           height = 5, 
           dpi = 300)
  }
  
  # 保存测试集ROC曲线图
  if (!is.null(test_roc_plot)) {
    test_roc_filename <- paste0("glmnet_test_roc_curve_", param_str, ".png")
    ggsave(file.path(FIG_DIR, test_roc_filename),
           test_roc_plot, 
           width = 6, 
           height = 6, 
           dpi = 300)
  }
  
  # 保存训练集ROC曲线图
  if (!is.null(train_roc_plot)) {
    train_roc_filename <- paste0("glmnet_train_roc_curve_", param_str, ".png")
    ggsave(file.path(FIG_DIR, train_roc_filename),
           train_roc_plot, 
           width = 6, 
           height = 6, 
           dpi = 300)
  }
  
  # 保存对比图
  if (!is.null(combined_roc_plot)) {
    combined_roc_filename <- paste0("glmnet_train_test_roc_comparison_", param_str, ".png")
    ggsave(file.path(FIG_DIR, combined_roc_filename),
           combined_roc_plot, 
           width = 7, 
           height = 7, 
           dpi = 300)
  }
  
  # 14. 提取重要特征
  cat("7. 提取重要特征...\n")
  
  if (!is.null(glmnet_model)) {
    # 提取系数
    coef_matrix <- coef(glmnet_model, s = lambda_value)
    coef_df <- as.data.frame(as.matrix(coef_matrix))
    colnames(coef_df) <- "Coefficient"
    coef_df <- coef_df[abs(coef_df$Coefficient) > 0, , drop = FALSE]
    
    if (nrow(coef_df) > 0) {
      coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), , drop = FALSE]
      
      cat("重要特征系数 (非零系数):\n")
      print(coef_df)
      
      features_filename <- paste0("glmnet_important_features_", param_str, ".csv")
      write.csv(coef_df, file.path(DATA_DIR, features_filename), 
                row.names = TRUE, fileEncoding = "UTF-8")
      
      # 打印系数摘要
      cat("\n=== 系数摘要 ===\n")
      cat("总特征数:", ncol(x_train), "\n")
      cat("非零系数特征数:", nrow(coef_df), "\n")
      cat("最大系数:", max(abs(coef_df$Coefficient)), "\n")
      cat("最小非零系数:", min(abs(coef_df$Coefficient)), "\n")
      
      # 保存模型参数
      model_params <- data.frame(
        Parameter = c("alpha", "lambda", "n_features", "n_nonzero_coefs", "intercept"),
        Value = c(alpha_value, lambda_value, ncol(x_train), nrow(coef_df), intercept),
        stringsAsFactors = FALSE
      )
      
      params_filename <- paste0("glmnet_model_parameters_", param_str, ".csv")
      write.csv(model_params, file.path(DATA_DIR, params_filename), 
                row.names = FALSE, fileEncoding = "UTF-8")
    }
  }
  
  # 15. 保存完整的R数据
  results <- list(
    model = glmnet_model,
    model_parameters = list(
      alpha = alpha_value,
      lambda = lambda_value,
      standardize = standardize,
      intercept = intercept,
      thresh = thresh,
      maxit = maxit,
      family = family
    ),
    train_predictions = list(
      probabilities = train_pred_prob,
      classes = train_pred_class
    ),
    test_predictions = list(
      probabilities = test_pred_prob,
      classes = test_pred_class
    ),
    train_performance = train_metrics_result,
    test_performance = test_metrics_result,
    performance_table = performance_table,
    feature_importance = if(exists("coef_df") && !is.null(coef_df)) coef_df else NULL
  )
  
  results_filename <- paste0("glmnet_model_results_", param_str, ".rdata")
  save(results, file = file.path(DATA_DIR, results_filename))
  
  # 16. 显示总结信息
  cat("\n=== glmnet模型性能总结 ===\n")
  print(performance_table)
  
  cat("\n=== 模型参数总结 ===\n")
  cat("alpha:", alpha_value, "\n")
  cat("lambda:", lambda_value, "\n")
  cat("训练集样本数:", nrow(x_train), "\n")
  cat("测试集样本数:", nrow(x_test), "\n")
  cat("特征数:", ncol(x_train), "\n")
  
  if (!is.null(train_roc_data)) {
    cat("训练集AUC:", round(train_roc_data$auc_value, 4), "\n")
  }
  if (!is.null(test_roc_data)) {
    cat("测试集AUC:", round(test_roc_data$auc_value, 4), "\n")
  }
  
  cat("\n=== 文件保存情况 ===\n")
  cat("1. ", performance_filename, "- 性能指标表格\n")
  
  if (exists("train_roc_filename") && file.exists(file.path(DATA_DIR, train_roc_filename))) {
    cat("2. ", train_roc_filename, "- 训练集ROC曲线点数据\n")
  }
  
  if (exists("test_roc_filename") && file.exists(file.path(DATA_DIR, test_roc_filename))) {
    cat("3. ", test_roc_filename, "- 测试集ROC曲线点数据\n")
  }
  
  if (exists("combined_filename") && file.exists(file.path(DATA_DIR, combined_filename))) {
    cat("4. ", combined_filename, "- 合并的ROC曲线数据\n")
  }
  
  if (exists("features_filename") && file.exists(file.path(DATA_DIR, features_filename))) {
    cat("5. ", features_filename, "- 重要特征系数\n")
  }
  
  if (exists("params_filename") && file.exists(file.path(DATA_DIR, params_filename))) {
    cat("6. ", params_filename, "- 模型参数\n")
  }
  
  cat("7. ", performance_plot_filename, "- 性能对比图\n")
  cat("8. glmnet_train_roc_curve_", param_str, ".png - 训练集ROC曲线图\n", sep = "")
  cat("9. glmnet_test_roc_curve_", param_str, ".png - 测试集ROC曲线图\n", sep = "")
  cat("10. glmnet_train_test_roc_comparison_", param_str, ".png - 训练测试ROC对比图\n", sep = "")
  cat("11. ", results_filename, "- 完整的R数据文件\n")
  
  cat("\n=== 分析完成 ===\n")
  cat("结果保存在:", DATA_DIR, "和", FIG_DIR, "目录\n")
  
  return(results)
}

# ======================================================
# 主执行流程
# ======================================================

cat("=== 执行主流程 ===\n")

# 定义glmnet参数（你可以修改这些值）
alpha_value <- 0.5281  # 平衡L1和L2正则化 (0=ridge, 1=lasso)
lambda_value <- 0.004597  # 正则化强度
standardize <- FALSE  # 是否标准化特征
intercept <- TRUE  # 是否包含截距项
thresh <- 1e-7  # 收敛阈值
maxit <- 1000  # 最大迭代次数
family <- "binomial"  # 分布类型

cat("glmnet参数配置:\n")
cat("alpha =", alpha_value, "(L1/L2混合参数)\n")
cat("lambda =", lambda_value, "(正则化强度)\n")
cat("standardize =", standardize, "(是否标准化特征)\n")
cat("intercept =", intercept, "(是否包含截距项)\n")
cat("thresh =", thresh, "(收敛阈值)\n")
cat("maxit =", maxit, "(最大迭代次数)\n")
cat("family =", family, "(分布类型)\n\n")

# 运行glmnet模型分析
results <- train_and_evaluate_glmnet_custom(
  train_data = train_data,
  test_data = test_data,
  alpha_value = alpha_value,
  lambda_value = lambda_value,
  standardize = standardize,
  intercept = intercept,
  thresh = thresh,
  maxit = maxit,
  family = family
)

cat("\n=== 所有分析完成 ===\n")

