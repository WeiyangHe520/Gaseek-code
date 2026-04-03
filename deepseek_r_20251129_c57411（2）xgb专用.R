###############################################
## XGBoost模型专项分析程序（修复版）        ##
## 包含Brier评分，使用write.csv保存结果   ##
###############################################

rm(list = ls())
set.seed(278)

# 创建目录
FIG_DIR <- "figures_glmnet_fixed/xgboost/"
DATA_DIR <- "data_glmnet_fixed/xgboost/"
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)

# 加载必要的包
library(caret)
library(tidyverse)
library(xgboost)
library(pROC)
library(ggplot2)
library(Matrix)
library(viridis)

# 检查数据
if (!exists("train_data") || !exists("test_data")) {
  if (file.exists(".left_data.rdata")) {
    load(".left_data.rdata")
    cat("已加载数据\n")
  } else {
    stop("未找到数据文件")
  }
}

# 数据预处理函数
prepare_xgboost_data <- function(data) {
  # 确保分组变量名称一致且为因子
  colnames(data)[ncol(data)] <- "class"
  
  # 转换类别为0/1（cancer=1, control=0）
  y <- ifelse(data$class == "cancer", 1, 0)
  
  # 移除类别列，保留特征
  features <- data[, -ncol(data)]
  
  # 转换特征为数值矩阵
  x <- as.matrix(features)
  
  # 检查并处理缺失值
  if (any(is.na(x))) {
    cat("发现缺失值，使用中位数填充\n")
    for (i in 1:ncol(x)) {
      x[is.na(x[, i]), i] <- median(x[, i], na.rm = TRUE)
    }
  }
  
  # 标准化特征（XGBoost对标准化敏感）
  x <- scale(x)
  
  return(list(x = x, y = y))
}

# 性能评估函数（包含Brier评分）
evaluate_xgboost_model <- function(predictions, actual, model_name = "", dataset = "") {
  # 确保输入正确
  if (length(predictions) == 0 || length(actual) == 0) {
    cat("错误：预测值或实际值为空 -", model_name, "\n")
    return(NULL)
  }
  
  # 检查预测值范围
  cat("预测概率范围:", range(predictions), "\n")
  
  # 预测类别
  pred_class <- ifelse(predictions > 0.5, 1, 0)
  
  # 混淆矩阵
  conf_matrix <- table(Predicted = pred_class, Actual = actual)
  cat(model_name, dataset, "混淆矩阵:\n")
  print(conf_matrix)
  
  # 提取TP, TN, FP, FN
  TP <- ifelse("1" %in% rownames(conf_matrix) && "1" %in% colnames(conf_matrix), 
               conf_matrix["1", "1"], 0)
  TN <- ifelse("0" %in% rownames(conf_matrix) && "0" %in% colnames(conf_matrix), 
               conf_matrix["0", "0"], 0)
  FP <- ifelse("1" %in% rownames(conf_matrix) && "0" %in% colnames(conf_matrix), 
               conf_matrix["1", "0"], 0)
  FN <- ifelse("0" %in% rownames(conf_matrix) && "1" %in% colnames(conf_matrix), 
               conf_matrix["0", "1"], 0)
  
  # 计算基本指标
  accuracy <- (TP + TN) / sum(conf_matrix)
  sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
  ppv <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  npv <- ifelse((TN + FN) > 0, TN / (TN + FN), 0)
  
  # AUC计算
  auc_value <- 0
  auc_ci <- "0 (0-0)"
  auc_ci_lower <- 0
  auc_ci_upper <- 0
  
  roc_obj <- tryCatch({
    roc(actual, predictions)
  }, error = function(e) {
    cat("计算AUC时出错 -", model_name, ":", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(roc_obj)) {
    auc_value <- auc(roc_obj)
    
    # 计算AUC的95%置信区间
    ci <- tryCatch({
      ci.auc(roc_obj)
    }, error = function(e) {
      cat("计算AUC CI时出错:", e$message, "\n")
      return(c(NA, auc_value, NA))
    })
    
    if (length(ci) == 3 && !any(is.na(ci))) {
      auc_ci_lower <- ci[1]
      auc_ci_upper <- ci[3]
      auc_ci <- paste0(round(auc_value, 3), " (", 
                       round(ci[1], 3), "-", 
                       round(ci[3], 3), ")")
    } else {
      auc_ci <- paste0(round(auc_value, 3), " (NA)")
    }
  }
  
  # Brier score计算
  brier_score <- mean((predictions - actual)^2)
  
  # Youden指数
  youden_index <- sensitivity + specificity - 1
  
  # F1分数
  f1_score <- ifelse((sensitivity + ppv) > 0, 
                     2 * (sensitivity * ppv) / (sensitivity + ppv), 0)
  
  # 创建结果数据框
  result_df <- data.frame(
    Model = model_name,
    Dataset = dataset,
    AUC = round(auc_value, 4),
    AUC_CI = auc_ci,
    AUC_CI_Lower = round(auc_ci_lower, 4),
    AUC_CI_Upper = round(auc_ci_upper, 4),
    ACC = round(accuracy, 4),
    SENS = round(sensitivity, 4),
    SPEC = round(specificity, 4),
    PPV = round(ppv, 4),
    NPV = round(npv, 4),
    YDI = round(youden_index, 4),
    F1 = round(f1_score, 4),
    BRIER = round(brier_score, 4),
    TP = TP,
    TN = TN,
    FP = FP,
    FN = FN,
    Total = sum(conf_matrix),
    stringsAsFactors = FALSE
  )
  
  cat("\n", model_name, dataset, "性能指标:\n")
  print(result_df[, 1:14])  # 打印前14列，不包括混淆矩阵数据
  cat("AUC 95% CI:", auc_ci, "\n")
  cat("Brier评分:", round(brier_score, 4), "\n\n")
  
  return(result_df)
}

# XGBoost专用训练函数
train_xgboost_model <- function(train_x, train_y, test_x, test_y, model_type = "xgbTree", model_name = "") {
  
  cat("\n=== 训练", model_name, "模型 ===\n")
  
  # 转换数据为DMatrix格式（XGBoost推荐）
  dtrain <- xgb.DMatrix(data = train_x, label = train_y)
  dtest <- xgb.DMatrix(data = test_x, label = test_y)
  
  # 设置参数
  if (model_type == "xgbTree") {
    params <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = 6,
      eta = 0.3,
      gamma = 0,
      colsample_bytree = 0.8,
      min_child_weight = 1,
      subsample = 0.8,
      nthread = 2
    )
    
    nrounds <- 100
    
  } else if (model_type == "xgbLinear") {
    params <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      booster = "gblinear",
      lambda = 1,
      alpha = 0,
      nthread = 2
    )
    
    nrounds <- 50
  }
  
  # 训练模型
  model <- tryCatch({
    xgb.train(
      params = params,
      data = dtrain,
      nrounds = nrounds,
      verbose = 0
    )
  }, error = function(e) {
    cat("XGBoost训练出错:", e$message, "\n")
    
    # 尝试更简单的配置
    cat("尝试更简单的配置...\n")
    params$eta <- 0.1
    params$nrounds <- 30
    
    xgb.train(
      params = params,
      data = dtrain,
      nrounds = params$nrounds,
      verbose = 0
    )
  })
  
  if (is.null(model)) {
    cat("模型训练失败\n")
    return(NULL)
  }
  
  # 预测
  train_pred <- predict(model, dtrain)
  test_pred <- predict(model, dtest)
  
  # 评估
  train_metrics <- evaluate_xgboost_model(train_pred, train_y, 
                                          model_name, "train")
  test_metrics <- evaluate_xgboost_model(test_pred, test_y, 
                                         model_name, "test")
  
  return(list(
    model = model,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    train_predictions = train_pred,
    test_predictions = test_pred
  ))
}

# 保存结果到CSV文件
save_results_to_csv <- function(results_list, file_path) {
  # 提取所有性能指标
  all_metrics <- data.frame()
  
  for (model_name in names(results_list)) {
    result <- results_list[[model_name]]
    
    if (!is.null(result$train_metrics)) {
      all_metrics <- rbind(all_metrics, result$train_metrics)
    }
    
    if (!is.null(result$test_metrics)) {
      all_metrics <- rbind(all_metrics, result$test_metrics)
    }
  }
  
  if (nrow(all_metrics) > 0) {
    # 重新排列列顺序
    all_metrics <- all_metrics %>%
      select(Model, Dataset, AUC, AUC_CI, AUC_CI_Lower, AUC_CI_Upper,
             ACC, SENS, SPEC, PPV, NPV, YDI, F1, BRIER,
             TP, TN, FP, FN, Total)
    
    # 保存到CSV
    write.csv(all_metrics, file_path, row.names = FALSE, fileEncoding = "UTF-8")
    cat("性能指标已保存到:", file_path, "\n")
    
    # 同时保存简化版本（只包含主要指标）
    simple_file_path <- gsub("\\.csv$", "_simple.csv", file_path)
    simple_metrics <- all_metrics %>%
      select(Model, Dataset, AUC, AUC_CI, ACC, SENS, SPEC, PPV, NPV, F1, BRIER)
    write.csv(simple_metrics, simple_file_path, row.names = FALSE, fileEncoding = "UTF-8")
    cat("简化版性能指标已保存到:", simple_file_path, "\n")
    
    return(all_metrics)
  } else {
    cat("没有性能指标数据可保存\n")
    return(NULL)
  }
}

# 生成详细报告
generate_detailed_report <- function(performance_df) {
  if (is.null(performance_df) || nrow(performance_df) == 0) {
    return(NULL)
  }
  
  separator_line <- paste(rep("=", 60), collapse = "")
  dash_line <- paste(rep("-", 40), collapse = "")
  
  cat("\n", separator_line, "\n", sep="")
  cat("模型性能详细报告\n")
  cat(separator_line, "\n\n", sep="")
  
  # 按模型和数据集展示
  for (model in unique(performance_df$Model)) {
    cat("模型:", model, "\n")
    cat(dash_line, "\n", sep="")
    
    model_data <- performance_df %>% filter(Model == model)
    
    for (dataset in c("train", "test")) {
      dataset_data <- model_data %>% filter(Dataset == dataset)
      
      if (nrow(dataset_data) > 0) {
        cat(dataset, "数据集性能:\n")
        cat("  AUC: ", sprintf("%.4f", dataset_data$AUC), "\n", sep="")
        cat("  AUC 95% CI: ", dataset_data$AUC_CI, "\n", sep="")
        cat("  准确率 (ACC): ", sprintf("%.4f", dataset_data$ACC), "\n", sep="")
        cat("  敏感度 (SENS): ", sprintf("%.4f", dataset_data$SENS), "\n", sep="")
        cat("  特异度 (SPEC): ", sprintf("%.4f", dataset_data$SPEC), "\n", sep="")
        cat("  阳性预测值 (PPV): ", sprintf("%.4f", dataset_data$PPV), "\n", sep="")
        cat("  阴性预测值 (NPV): ", sprintf("%.4f", dataset_data$NPV), "\n", sep="")
        cat("  Youden指数 (YDI): ", sprintf("%.4f", dataset_data$YDI), "\n", sep="")
        cat("  F1分数: ", sprintf("%.4f", dataset_data$F1), "\n", sep="")
        cat("  Brier评分: ", sprintf("%.4f", dataset_data$BRIER), "\n", sep="")
        cat("  混淆矩阵: TP=", dataset_data$TP, ", TN=", dataset_data$TN, 
            ", FP=", dataset_data$FP, ", FN=", dataset_data$FN, "\n", sep="")
        cat("\n")
      }
    }
  }
  
  # 找出最佳模型（基于测试集AUC）
  test_results <- performance_df %>% 
    filter(Dataset == "test") %>%
    arrange(desc(AUC))
  
  if (nrow(test_results) > 0) {
    cat(separator_line, "\n", sep="")
    cat("最佳模型评选（基于测试集AUC）\n")
    cat(separator_line, "\n", sep="")
    
    for (i in 1:min(3, nrow(test_results))) {
      cat("\n第", i, "名: ", test_results$Model[i], "\n", sep="")
      cat("  AUC: ", sprintf("%.4f", test_results$AUC[i]), "\n", sep="")
      cat("  AUC 95% CI: ", test_results$AUC_CI[i], "\n", sep="")
      cat("  准确率: ", sprintf("%.4f", test_results$ACC[i]), "\n", sep="")
      cat("  Brier评分: ", sprintf("%.4f", test_results$BRIER[i]), "\n", sep="")
    }
  }
}

# 主执行函数
run_xgboost_analysis <- function() {
  separator_line <- paste(rep("=", 60), collapse = "")
  
  cat(separator_line, "\n", sep="")
  cat("XGBoost模型专项分析开始\n")
  cat(separator_line, "\n\n", sep="")
  
  # 准备数据
  train_prep <- prepare_xgboost_data(train_data)
  test_prep <- prepare_xgboost_data(test_data)
  
  cat("数据信息:\n")
  cat("  训练集: ", nrow(train_prep$x), "个样本, ", ncol(train_prep$x), "个特征\n", sep="")
  cat("  测试集: ", nrow(test_prep$x), "个样本, ", ncol(test_prep$x), "个特征\n", sep="")
  cat("  训练集类别分布: cancer = ", sum(train_prep$y), ", control = ", sum(train_prep$y == 0), "\n", sep="")
  cat("  测试集类别分布: cancer = ", sum(test_prep$y), ", control = ", sum(test_prep$y == 0), "\n\n", sep="")
  
  results <- list()
  
  # 训练xgbTree
  cat("开始训练xgbTree模型...\n")
  xgbTree_result <- train_xgboost_model(
    train_prep$x, train_prep$y,
    test_prep$x, test_prep$y,
    model_type = "xgbTree",
    model_name = "xgbTree"
  )
  results[["xgbTree"]] <- xgbTree_result
  
  # 训练xgbLinear
  cat("\n开始训练xgbLinear模型...\n")
  xgbLinear_result <- train_xgboost_model(
    train_prep$x, train_prep$y,
    test_prep$x, test_prep$y,
    model_type = "xgbLinear",
    model_name = "xgbLinear"
  )
  results[["xgbLinear"]] <- xgbLinear_result
  
  # 保存结果到CSV
  cat("\n", separator_line, "\n", sep="")
  cat("保存结果文件\n")
  cat(separator_line, "\n", sep="")
  
  performance_table <- save_results_to_csv(
    results, 
    file.path(DATA_DIR, "xgboost_performance_metrics.csv")
  )
  
  # 生成详细报告
  if (!is.null(performance_table)) {
    generate_detailed_report(performance_table)
    
    # 打印完整的性能表格
    cat("\n", separator_line, "\n", sep="")
    cat("完整性能指标表格\n")
    cat(separator_line, "\n", sep="")
    print(performance_table)
  }
  
  # 检查成功的模型
  successful_models <- names(results)[!sapply(results, is.null)]
  cat("\n训练成功的模型: ", paste(successful_models, collapse = ", "), "\n", sep="")
  
  cat("\n", separator_line, "\n", sep="")
  cat("XGBoost分析完成\n")
  cat(separator_line, "\n\n", sep="")
  
  return(list(
    results = results,
    performance_table = performance_table
  ))
}

# 运行分析
analysis_results <- run_xgboost_analysis()

# 保存工作空间
save.image(file.path(DATA_DIR, "xgboost_analysis_complete.RData"))
cat("工作空间已保存到: ", file.path(DATA_DIR, "xgboost_analysis_complete.RData"), "\n\n", sep="")

# 提供文件路径信息
cat("输出文件信息:\n")
cat("1. 完整性能指标文件: ", file.path(DATA_DIR, "xgboost_performance_metrics.csv"), "\n", sep="")
cat("2. 简化性能指标文件: ", file.path(DATA_DIR, "xgboost_performance_metrics_simple.csv"), "\n", sep="")
cat("3. R工作空间文件: ", file.path(DATA_DIR, "xgboost_analysis_complete.RData"), "\n", sep="")

# 显示前几行数据示例
if (!is.null(analysis_results$performance_table)) {
  cat("\n性能指标示例（前3行）:\n")
  print(head(analysis_results$performance_table, 3))
}