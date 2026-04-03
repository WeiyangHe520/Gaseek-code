###############################################
## 临床预测模型开发教学代码 - 多模型输出版   ##
## 修复版本 - 包含单特征模型               ##
###############################################

rm(list = ls())
set.seed(3456)
FIG_DIR <- "figures/"
DATA_DIR <- "data/"
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
  
  return(result_df)
}

# 修复的模型训练和评估函数
evaluate_all_models <- function(train_data, test_data, method_list = c("glmnet")) {
  
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

# 修复的创建综合性能表格函数
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
  
  # 修复：使用基础R方法重新排列列顺序
  if (nrow(performance_table) > 0) {
    # 获取所有列名
    all_cols <- colnames(performance_table)
    # 移除Model和Dataset
    other_cols <- setdiff(all_cols, c("Model", "Dataset"))
    # 重新排列列顺序
    performance_table <- performance_table[, c("Model", "Dataset", other_cols)]
  }
  
  return(performance_table)
}

# 可视化所有模型性能
plot_all_models_performance <- function(performance_table) {
  plots <- list()
  
  if (nrow(performance_table) == 0) {
    cat("没有可用的性能数据用于绘图\n")
    return(plots)
  }
  
  # AUC比较图
  auc_plot <- performance_table %>%
    ggplot(aes(x = Model, y = AUC, fill = Dataset)) +
    geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
    scale_fill_manual(values = c("train" = "#1f77b4", "test" = "#ff7f0e")) +
    labs(title = "所有模型的AUC性能比较",
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

# ============================================
# 新增功能：单特征模型训练和ROC曲线绘制
# ============================================

# 训练单特征GLM模型
train_single_feature_models <- function(train_data, test_data, features = NULL) {
  if (is.null(features)) {
    features <- colnames(train_data)[-ncol(train_data)]
  }
  
  cat("=== 开始训练单特征GLM模型 ===\n")
  cat("特征数量:", length(features), "\n")
  
  single_feature_results <- list()
  roc_curves <- list()
  
  # 设置训练控制参数
  fit_control <- trainControl(
    method = "cv",
    number = 3,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    verboseIter = FALSE
  )
  
  for (i in 1:length(features)) {
    feature <- features[i]
    cat("训练单特征模型 (", i, "/", length(features), "):", feature, "\n")
    
    tryCatch({
      # 创建单特征数据集
      train_subset <- train_data[, c(feature, "class")]
      
      # 训练GLM模型
      model <- train(
        class ~ .,
        data = train_subset,
        method = "glm",
        family = "binomial",
        trControl = fit_control,
        metric = "ROC"
      )
      
      # 在测试集上预测
      test_pred <- predict(model, test_data, type = "prob")[, "cancer"]
      
      # 计算ROC
      roc_res <- roc(test_data$class, test_pred)
      
      # 存储结果
      single_feature_results[[paste0(feature, "_single")]] <- list(
        model = model,
        roc = roc_res,
        auc = auc(roc_res),
        predictions = test_pred
      )
      
      roc_curves[[paste0(feature, "_single")]] <- roc_res
      
      cat("✅", feature, "单特征模型完成 (AUC =", round(auc(roc_res), 3), ")\n")
      
    }, error = function(e) {
      cat("❌ 训练单特征模型", feature, "时出错:", e$message, "\n")
    })
  }
  
  return(list(
    results = single_feature_results,
    roc_curves = roc_curves
  ))
}

# 修复的绘制所有ROC曲线函数（每6条曲线一个图）- 统一坐标轴版本，带自动图例换行
plot_all_roc_curves_grouped <- function(multi_model_results, single_feature_results, curves_per_plot = 6, legend_width = 30) {
  
  # 收集所有ROC曲线
  all_roc_curves <- list()
  
  # 添加多特征模型的ROC曲线
  for (model_name in names(multi_model_results)) {
    model_result <- multi_model_results[[model_name]]
    if (!is.null(model_result$test_predictions)) {
      roc_obj <- roc(test_data$class, model_result$test_predictions)
      all_roc_curves[[model_name]] <- roc_obj
    }
  }
  
  # 添加单特征模型的ROC曲线
  if (!is.null(single_feature_results$roc_curves)) {
    all_roc_curves <- c(all_roc_curves, single_feature_results$roc_curves)
  }
  
  if (length(all_roc_curves) == 0) {
    cat("没有可用的ROC曲线数据\n")
    return(NULL)
  }
  
  # 计算AUC值并排序
  auc_values <- sapply(all_roc_curves, function(x) as.numeric(x$auc))
  sorted_models <- names(sort(auc_values, decreasing = TRUE))
  
  cat("总ROC曲线数量:", length(sorted_models), "\n")
  cat("每组的曲线数量:", curves_per_plot, "\n")
  
  # 将模型分组 - 修复分组逻辑
  model_groups <- list()
  total_models <- length(sorted_models)
  
  for (i in 1:ceiling(total_models / curves_per_plot)) {
    start_idx <- (i - 1) * curves_per_plot + 1
    end_idx <- min(i * curves_per_plot, total_models)
    model_groups[[i]] <- sorted_models[start_idx:end_idx]
    
    cat("第", i, "组包含", length(model_groups[[i]]), "个模型:\n")
    cat(paste(model_groups[[i]], collapse = ", "), "\n")
  }
  
  # 创建颜色方案 - 增加到9种颜色
  colors <- viridis::viridis(curves_per_plot)
  
  plot_list <- list()
  
  for (i in 1:length(model_groups)) {
    model_group <- model_groups[[i]]
    group_size <- length(model_group)
    
    cat("正在绘制第", i, "组，包含", group_size, "个模型\n")
    
    # 准备当前分组的ROC数据
    roc_data <- data.frame()
    for (j in 1:group_size) {
      model_name <- model_group[j]
      roc_obj <- all_roc_curves[[model_name]]
      
      temp_data <- data.frame(
        FPR = 1 - roc_obj$specificities,
        TPR = roc_obj$sensitivities,
        Model = model_name,
        AUC = round(roc_obj$auc, 3),
        stringsAsFactors = FALSE
      )
      roc_data <- rbind(roc_data, temp_data)
    }
    
    # 为模型创建显示标签（包含AUC值）
    roc_data$Model_Label <- paste0(roc_data$Model, " (AUC = ", roc_data$AUC, ")")
    
    # 按AUC值排序因子水平
    group_auc <- auc_values[model_group]
    model_levels <- paste0(model_group, " (AUC = ", round(group_auc, 3), ")")
    
    # 创建自动换行的标签
    wrapped_labels <- sapply(model_levels, function(label) {
      if (nchar(label) > legend_width) {
        # 插入换行符
        words <- strsplit(label, " ")[[1]]
        current_line <- ""
        result <- ""
        
        for (word in words) {
          if (nchar(current_line) + nchar(word) + 1 <= legend_width) {
            if (current_line == "") {
              current_line <- word
            } else {
              current_line <- paste(current_line, word)
            }
          } else {
            if (result == "") {
              result <- current_line
            } else {
              result <- paste0(result, "\n", current_line)
            }
            current_line <- word
          }
        }
        
        if (result == "") {
          result <- current_line
        } else {
          result <- paste0(result, "\n", current_line)
        }
        
        return(result)
      } else {
        return(label)
      }
    })
    
    roc_data$Model_Label <- factor(roc_data$Model_Label, levels = model_levels)
    
    # 使用当前组的颜色
    current_colors <- colors[1:group_size]
    
    # 创建ROC图 - 修改这里确保坐标轴完全一致，并添加图例换行
    p <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Model_Label)) +
      geom_line(size = 1.0) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", 
                  color = "gray50", size = 0.6) +
      
      # 使用换行后的标签
      scale_color_manual(values = current_colors, 
                         name = "Models",
                         labels = wrapped_labels) +
      
      # 固定坐标轴范围和刻度，确保所有图一致
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
      
      theme_minimal(base_size = 12) +
      theme(
        legend.position = "right",
        legend.text = element_text(size = 9),  # 减小图例文字大小以容纳更多内容
        legend.title = element_text(size = 10, face = "bold"),
        legend.key.height = unit(0.8, "lines"),  # 减小图例项高度
        legend.spacing.y = unit(0.3, "lines"),   # 减小图例项间距
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 10),
        plot.margin = margin(15, 15, 15, 15)
      ) +
      
      # 使用coord_equal确保1:1比例，但固定范围
      coord_equal(ratio = 1)  # 使用coord_equal而不是coord_fixed，更灵活
    
    plot_list[[i]] <- p
  }
  
  return(list(
    plots = plot_list,
    all_roc_curves = all_roc_curves,
    auc_values = auc_values,
    model_groups = model_groups  # 添加这个以保存分组信息
  ))
}
# 辅助函数：智能文本换行
wrap_text <- function(text, width = 30) {
  if (nchar(text) <= width) {
    return(text)
  }
  
  # 按空格分割单词
  words <- strsplit(text, " ")[[1]]
  lines <- character(0)
  current_line <- ""
  
  for (word in words) {
    # 如果添加这个词会超过宽度，则开始新的一行
    if (nchar(current_line) + nchar(word) + 1 > width) {
      if (nchar(current_line) > 0) {
        lines <- c(lines, current_line)
      }
      current_line <- word
    } else {
      # 否则添加到当前行
      if (nchar(current_line) == 0) {
        current_line <- word
      } else {
        current_line <- paste(current_line, word)
      }
    }
  }
  
  # 添加最后一行
  if (nchar(current_line) > 0) {
    lines <- c(lines, current_line)
  }
  
  # 用换行符连接所有行
  return(paste(lines, collapse = "\n"))
}

# 主执行流程
cat("=== 开始多模型比较分析 ===\n")

# 使用更可靠的模型列表
reliable_methods <- c("glmnet")

# 1. 评估所有多特征模型
all_model_results <- evaluate_all_models(
  train_data = train_data,
  test_data = test_data,
  method_list = reliable_methods
)

# 2. 训练单特征GLM模型
single_feature_results <- train_single_feature_models(
  train_data = train_data,
  test_data = test_data
)

# 3. 创建综合性能表格
performance_table <- create_comprehensive_performance_table(all_model_results)

# 4. 绘制所有ROC曲线 - 现在每6个一组，使用图例自动换行
roc_results <- plot_all_roc_curves_grouped(all_model_results, single_feature_results, 
                                           curves_per_plot = 6, legend_width = 30)

# 调试信息
cat("=== 调试信息 ===\n")
cat("多特征模型数量:", length(all_model_results), "\n")
cat("成功多特征模型:", sum(sapply(all_model_results, function(x) !is.null(x$test_predictions))), "\n")
cat("单特征模型数量:", length(single_feature_results$roc_curves), "\n")

# 检查总模型数量
total_curves <- sum(sapply(all_model_results, function(x) !is.null(x$test_predictions))) + 
  length(single_feature_results$roc_curves)
cat("总ROC曲线数量:", total_curves, "\n")

cat("=== 所有模型性能总结 ===\n")

# 保存ROC曲线图
if (!is.null(roc_results)) {
  # 使用固定尺寸，确保所有图的坐标轴长度一致
  fixed_width <- 6.7  # 固定宽度
  fixed_height <- 3.8  # 固定高度
  
  # 保存为单独的PNG文件
  for (i in 1:length(roc_results$plots)) {
    cat(sprintf("保存第 %d 组ROC曲线: %d 个模型\n", 
                i, length(roc_results$model_groups[[i]])))
    
    ggsave(file.path(FIG_DIR, paste0("roc_curves_group_", i, ".png")),
           roc_results$plots[[i]], 
           width = fixed_width, 
           height = fixed_height, 
           dpi = 300)
  }
  
  # 保存为一个PDF文件，所有页面尺寸一致
  pdf(file.path(FIG_DIR, "all_roc_curves_grouped.pdf"), 
      width = fixed_width, 
      height = fixed_height,
      onefile = TRUE)
  
  for (i in 1:length(roc_results$plots)) {
    print(roc_results$plots[[i]])
  }
  
  dev.off()
  
  cat("ROC曲线图已保存到", FIG_DIR, "目录\n")
  cat("总共生成了", length(roc_results$plots), "组ROC曲线图\n")
  
  # 显示AUC排名
  cat("\n=== 所有模型AUC排名 ===\n")
  auc_ranking <- data.frame(
    Model = names(roc_results$auc_values),
    AUC = round(roc_results$auc_values, 4)
  ) %>% arrange(desc(AUC))
  
  print(auc_ranking)
  write.csv(auc_ranking, file.path(DATA_DIR, "all_models_auc_ranking.csv"), 
            row.names = FALSE, fileEncoding = "UTF-8")
}



# 显示成功的模型
successful_multi_models <- names(all_model_results)[sapply(all_model_results, function(x) !is.null(x$model))]
successful_single_models <- names(single_feature_results$results)

cat("\n=== 训练完成总结 ===\n")
cat("成功训练的多特征模型:", length(successful_multi_models), "\n")
cat("成功训练的单特征模型:", length(successful_single_models), "\n")
cat("总模型数量:", length(successful_multi_models) + length(successful_single_models), "\n")

# 保存所有结果
save(all_model_results, single_feature_results, roc_results,
     file = file.path(DATA_DIR, "complete_model_results.rdata"))

cat("\n=== 分析完成 ===\n")
cat("结果保存在:", DATA_DIR, "和", FIG_DIR, "目录\n")

