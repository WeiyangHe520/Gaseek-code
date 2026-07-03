###############################################
## XGBoost模型专项分析（含类别权重 + 5折交叉验证）##
## 功能：训练xgbTree和xgbLinear，支持加权   ##
## 输出：性能指标CSV、ROC曲线、特征重要性等  ##
###############################################

rm(list = ls())
set.seed(278)

# 创建目录
FIG_DIR <- "figures_XGBoost"
DATA_DIR <- "results_XGBoost"
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)

# 加载必要的包
library(caret)
library(tidyverse)
library(xgboost)
library(pROC)
library(ggplot2)
library(Matrix)

# 检查数据
if (!exists("train_data") || !exists("test_data")) {
  if (file.exists(".left_data.rdata")) {
    load(".left_data.rdata")
    cat("已加载数据\n")
  } else {
    stop("未找到数据文件 .left_data.rdata")
  }
}

# ================== 数据预处理函数 ==================
prepare_xgboost_data <- function(data) {
  colnames(data)[ncol(data)] <- "class"
  y <- ifelse(data$class == "cancer", 1, 0)
  features <- data[, -ncol(data)]
  x <- as.matrix(features)
  if (any(is.na(x))) {
    cat("发现缺失值，使用中位数填充\n")
    for (i in 1:ncol(x)) {
      x[is.na(x[, i]), i] <- median(x[, i], na.rm = TRUE)
    }
  }
  # 数据已经过Yeo-Johnson变换和标准化，不需要再次标准化
  # x <- scale(x)  # 移除重复标准化
  return(list(x = x, y = y))
}

# ================== 性能评估函数（含Brier和AUC_95CI） ==================
evaluate_xgboost_model <- function(predictions, actual, model_name = "", dataset = "", 
                                   weights = NULL, optimize_threshold = TRUE) {
  if (length(predictions) == 0 || length(actual) == 0) {
    cat("错误：预测值或实际值为空 -", model_name, "\n")
    return(NULL)
  }
  
  # 如果未提供权重，使用等权重
  if (is.null(weights)) {
    weights <- rep(1, length(actual))
  }
  
  # 阈值优化（基于加权Youden指数）
  if (optimize_threshold) {
    thresholds <- sort(unique(predictions))
    best_metric <- -Inf
    best_thresh <- 0.5
    for (th in thresholds) {
      pred_class <- ifelse(predictions > th, 1, 0)
      tp <- sum(weights[pred_class == 1 & actual == 1])
      fn <- sum(weights[pred_class == 0 & actual == 1])
      fp <- sum(weights[pred_class == 1 & actual == 0])
      tn <- sum(weights[pred_class == 0 & actual == 0])
      sens <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
      spec <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
      youden <- sens + spec - 1
      if (youden > best_metric) {
        best_metric <- youden
        best_thresh <- th
      }
    }
  } else {
    best_thresh <- 0.5
  }
  
  # 使用最佳阈值进行分类
  pred_class <- ifelse(predictions > best_thresh, 1, 0)
  
  # 计算混淆矩阵
  TP <- sum(pred_class == 1 & actual == 1)
  TN <- sum(pred_class == 0 & actual == 0)
  FP <- sum(pred_class == 1 & actual == 0)
  FN <- sum(pred_class == 0 & actual == 1)
  
  accuracy    <- (TP + TN) / (TP + TN + FP + FN)
  sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
  ppv         <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  npv         <- ifelse((TN + FN) > 0, TN / (TN + FN), 0)
  youden      <- sensitivity + specificity - 1
  f1          <- ifelse((sensitivity + ppv) > 0, 2 * (sensitivity * ppv) / (sensitivity + ppv), 0)
  
  # 加权Brier评分
  brier <- sum(weights * (predictions - actual)^2) / sum(weights)
  
  roc_obj <- tryCatch(roc(actual, predictions, quiet = TRUE), error = function(e) NULL)
  auc_val <- if (!is.null(roc_obj)) auc(roc_obj) else 0
  auc_ci_lower <- auc_ci_upper <- NA
  if (!is.null(roc_obj)) {
    ci <- tryCatch(ci.auc(roc_obj, conf.level = 0.95), error = function(e) c(NA, NA, NA))
    if (length(ci) == 3) {
      auc_ci_lower <- ci[1]
      auc_ci_upper <- ci[3]
    }
  }
  auc_ci_str <- if (!is.na(auc_ci_lower)) 
    paste0(round(auc_val, 3), " (", round(auc_ci_lower, 3), "-", round(auc_ci_upper, 3), ")") 
  else paste0(round(auc_val, 3), " (NA)")
  
  result_df <- data.frame(
    Model = model_name,
    Dataset = dataset,
    AUC = round(auc_val, 4),
    AUC_95CI = auc_ci_str,
    AUC_CI_Lower = round(auc_ci_lower, 4),
    AUC_CI_Upper = round(auc_ci_upper, 4),
    ACC = round(accuracy, 4),
    SENS = round(sensitivity, 4),
    SPEC = round(specificity, 4),
    PPV = round(ppv, 4),
    NPV = round(npv, 4),
    YDI = round(youden, 4),
    F1 = round(f1, 4),
    BRIER = round(brier, 4),
    Best_Threshold = round(best_thresh, 4),
    TP = TP, TN = TN, FP = FP, FN = FN, Total = TP + TN + FP + FN,
    stringsAsFactors = FALSE
  )
  
  return(result_df)
}

# ================== 5折交叉验证函数（XGBoost版本） ==================
perform_cv_xgboost <- function(X, y, weights, model_type = "xgbTree", nfolds = 5) {
  set.seed(278)
  
  # 创建分层折叠
  folds <- createFolds(y, k = nfolds, list = TRUE, returnTrain = FALSE)
  
  # 存储每折的结果
  cv_results <- list()
  cv_predictions <- list()
  cv_models <- list()
  
  for (i in 1:nfolds) {
    cat(sprintf("\n--- 交叉验证第 %d/%d 折 ---\n", i, nfolds))
    
    # 划分训练和验证集
    val_indices <- folds[[i]]
    train_indices <- setdiff(1:nrow(X), val_indices)
    
    X_train_fold <- X[train_indices, ]
    y_train_fold <- y[train_indices]
    weights_train_fold <- weights[train_indices]
    
    X_val_fold <- X[val_indices, ]
    y_val_fold <- y[val_indices]
    weights_val_fold <- weights[val_indices]
    
    # 准备XGBoost数据
    dtrain <- xgb.DMatrix(data = X_train_fold, label = y_train_fold, weight = weights_train_fold)
    dval <- xgb.DMatrix(data = X_val_fold, label = y_val_fold)
    
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
    } else {
      stop("未知的 model_type")
    }
    
    # 训练模型
    model_fold <- tryCatch({
      xgb.train(params = params, data = dtrain, nrounds = nrounds, verbose = 0)
    }, error = function(e) {
      cat("XGBoost训练出错:", e$message, "\n尝试简化配置...\n")
      params$eta <- 0.1
      nrounds <- 30
      xgb.train(params = params, data = dtrain, nrounds = nrounds, verbose = 0)
    })
    
    if (is.null(model_fold)) {
      cat("模型训练失败，跳过此折\n")
      next
    }
    
    # 预测验证集
    pred_prob <- predict(model_fold, dval)
    
    # 保存结果
    cv_predictions[[i]] <- pred_prob
    cv_models[[i]] <- model_fold
    
    # 评估验证集（使用加权指标）
    val_metrics <- evaluate_xgboost_model(
      predictions = pred_prob,
      actual = y_val_fold,
      model_name = model_type,
      dataset = paste0("CV_Fold_", i),
      weights = weights_val_fold,
      optimize_threshold = TRUE
    )
    
    if (!is.null(val_metrics)) {
      cv_results[[i]] <- val_metrics
      cat(sprintf("  验证集样本数: %d\n", length(val_indices)))
      cat(sprintf("  验证集癌症/对照: %d/%d\n", 
                  sum(y_val_fold == 1), sum(y_val_fold == 0)))
      cat(sprintf("  AUC = %.3f\n", val_metrics$AUC))
      cat(sprintf("  ACC = %.3f\n", val_metrics$ACC))
      cat(sprintf("  SENS = %.3f\n", val_metrics$SENS))
      cat(sprintf("  SPEC = %.3f\n", val_metrics$SPEC))
    }
  }
  
  # 合并所有折的结果
  if (length(cv_results) > 0) {
    cv_results_df <- do.call(rbind, cv_results)
    
    # 计算平均值和标准差
    cv_summary <- data.frame(
      Metric = c("AUC", "ACC", "SENS", "SPEC", "PPV", "NPV", "YDI", "F1", "BRIER"),
      Mean = apply(cv_results_df[, c("AUC", "ACC", "SENS", "SPEC", "PPV", "NPV", "YDI", "F1", "BRIER")], 2, mean, na.rm = TRUE),
      SD = apply(cv_results_df[, c("AUC", "ACC", "SENS", "SPEC", "PPV", "NPV", "YDI", "F1", "BRIER")], 2, sd, na.rm = TRUE)
    )
  } else {
    cv_results_df <- NULL
    cv_summary <- NULL
  }
  
  return(list(
    results = cv_results_df,
    summary = cv_summary,
    predictions = cv_predictions,
    models = cv_models,
    folds = folds
  ))
}

# ================== XGBoost 训练函数（支持样本权重） ==================
train_xgboost_model <- function(train_x, train_y, test_x, test_y, 
                                model_type = "xgbTree", model_name = "",
                                sample_weights = NULL, optimize_threshold = TRUE) {
  
  cat("\n=== 训练", model_name, "模型 ===\n")
  
  dtrain <- xgb.DMatrix(data = train_x, label = train_y, weight = sample_weights)
  dtest  <- xgb.DMatrix(data = test_x, label = test_y)
  
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
  } else {
    stop("未知的 model_type")
  }
  
  model <- tryCatch({
    xgb.train(params = params, data = dtrain, nrounds = nrounds, verbose = 0)
  }, error = function(e) {
    cat("XGBoost训练出错:", e$message, "\n尝试简化配置...\n")
    params$eta <- 0.1
    nrounds <- 30
    xgb.train(params = params, data = dtrain, nrounds = nrounds, verbose = 0)
  })
  
  if (is.null(model)) {
    cat("模型训练失败\n")
    return(NULL)
  }
  
  train_pred <- predict(model, dtrain)
  test_pred  <- predict(model, dtest)
  
  # 训练集评估（使用训练集权重）
  train_metrics <- evaluate_xgboost_model(
    predictions = train_pred,
    actual = train_y,
    model_name = model_name,
    dataset = "Training",
    weights = sample_weights,
    optimize_threshold = optimize_threshold
  )
  
  # 测试集评估（使用等权重）
  test_metrics <- evaluate_xgboost_model(
    predictions = test_pred,
    actual = test_y,
    model_name = model_name,
    dataset = "Testing",
    weights = rep(1, length(test_y)),
    optimize_threshold = optimize_threshold
  )
  
  return(list(
    model = model,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    train_predictions = train_pred,
    test_predictions = test_pred
  ))
}

# ================== 保存结果到 CSV ==================
save_results_to_csv <- function(results_list, cv_output_list, file_path) {
  all_metrics <- data.frame()
  
  # 添加最终模型结果
  for (model_name in names(results_list)) {
    result <- results_list[[model_name]]
    if (!is.null(result$train_metrics)) {
      all_metrics <- rbind(all_metrics, result$train_metrics)
    }
    if (!is.null(result$test_metrics)) {
      all_metrics <- rbind(all_metrics, result$test_metrics)
    }
  }
  
  # 添加交叉验证结果
  for (model_name in names(cv_output_list)) {
    cv_result <- cv_output_list[[model_name]]
    if (!is.null(cv_result$results)) {
      cv_result$results$Dataset <- paste0("CV_", cv_result$results$Dataset)
      all_metrics <- rbind(all_metrics, cv_result$results)
    }
  }
  
  if (nrow(all_metrics) > 0) {
    all_metrics <- all_metrics %>%
      select(Model, Dataset, AUC, AUC_95CI, AUC_CI_Lower, AUC_CI_Upper,
             ACC, SENS, SPEC, PPV, NPV, YDI, F1, BRIER, Best_Threshold,
             TP, TN, FP, FN, Total)
    write.csv(all_metrics, file_path, row.names = FALSE, fileEncoding = "UTF-8")
    cat("性能指标已保存到:", file_path, "\n")
    
    simple_path <- gsub("\\.csv$", "_simple.csv", file_path)
    simple_metrics <- all_metrics %>%
      select(Model, Dataset, AUC, AUC_95CI, ACC, SENS, SPEC, PPV, NPV, F1, BRIER)
    write.csv(simple_metrics, simple_path, row.names = FALSE, fileEncoding = "UTF-8")
    cat("简化版已保存到:", simple_path, "\n")
    return(all_metrics)
  } else {
    cat("没有性能指标可保存\n")
    return(NULL)
  }
}

# ================== 保存交叉验证汇总 ==================
save_cv_summary <- function(cv_output_list, file_path) {
  all_cv_summary <- data.frame()
  
  for (model_name in names(cv_output_list)) {
    cv_summary <- cv_output_list[[model_name]]$summary
    if (!is.null(cv_summary)) {
      cv_summary$Model <- model_name
      all_cv_summary <- rbind(all_cv_summary, cv_summary)
    }
  }
  
  if (nrow(all_cv_summary) > 0) {
    all_cv_summary <- all_cv_summary %>%
      select(Model, Metric, Mean, SD)
    write.csv(all_cv_summary, file_path, row.names = FALSE, fileEncoding = "UTF-8")
    cat("交叉验证汇总已保存到:", file_path, "\n")
    return(all_cv_summary)
  } else {
    cat("没有交叉验证汇总可保存\n")
    return(NULL)
  }
}

# ================== 生成详细报告 ==================
generate_detailed_report <- function(performance_df, cv_summary_df) {
  if (is.null(performance_df) || nrow(performance_df) == 0) return(NULL)
  
  separator <- paste(rep("=", 60), collapse = "")
  cat("\n", separator, "\n", sep="")
  cat("模型性能详细报告\n")
  cat(separator, "\n\n", sep="")
  
  # 最终模型性能
  cat("=== 最终模型性能 ===\n\n")
  for (model in unique(performance_df$Model[!grepl("CV_", performance_df$Dataset)])) {
    cat("模型:", model, "\n")
    cat(paste(rep("-", 40), collapse = ""), "\n", sep="")
    for (dataset in c("Training", "Testing")) {
      d <- performance_df %>% filter(Model == model, Dataset == dataset)
      if (nrow(d) > 0) {
        cat(dataset, ":\n")
        cat("  AUC: ", sprintf("%.4f", d$AUC), " (95% CI: ", d$AUC_95CI, ")\n", sep="")
        cat("  准确率: ", sprintf("%.4f", d$ACC), "\n", sep="")
        cat("  敏感度: ", sprintf("%.4f", d$SENS), "\n", sep="")
        cat("  特异度: ", sprintf("%.4f", d$SPEC), "\n", sep="")
        cat("  F1分数: ", sprintf("%.4f", d$F1), "\n", sep="")
        cat("  Brier评分: ", sprintf("%.4f", d$BRIER), "\n")
        cat("  最佳阈值: ", sprintf("%.4f", d$Best_Threshold), "\n\n", sep="")
      }
    }
  }
  
  # 交叉验证结果
  if (!is.null(cv_summary_df) && nrow(cv_summary_df) > 0) {
    cat("\n=== 5折交叉验证平均值 ===\n\n")
    for (model in unique(cv_summary_df$Model)) {
      cat("模型:", model, "\n")
      cv_data <- cv_summary_df %>% filter(Model == model)
      for (i in 1:nrow(cv_data)) {
        cat(sprintf("  %s = %.3f (SD: %.3f)\n", 
                    cv_data$Metric[i], cv_data$Mean[i], cv_data$SD[i]))
      }
      cat("\n")
    }
  }
  
  # 最佳模型
  best <- performance_df %>% 
    filter(Dataset == "Testing") %>% 
    arrange(desc(AUC))
  if (nrow(best) > 0) {
    cat(separator, "\n", sep="")
    cat("最佳模型（测试集AUC）:\n")
    cat(separator, "\n", sep="")
    for (i in 1:min(3, nrow(best))) {
      cat(sprintf("%d. %s  AUC=%.4f  F1=%.4f  Brier=%.4f\n", 
                  i, best$Model[i], best$AUC[i], best$F1[i], best$BRIER[i]))
    }
  }
}

# ================== 绘制 ROC 曲线 ==================
plot_roc_curves <- function(results, test_labels, file_path) {
  png(file_path, width = 8, height = 6, units = "in", res = 300)
  colors <- c("xgbTree" = "blue", "xgbLinear" = "red")
  first <- TRUE
  for (model_name in c("xgbTree", "xgbLinear")) {
    if (!is.null(results[[model_name]]$test_predictions)) {
      roc_obj <- roc(test_labels, results[[model_name]]$test_predictions, quiet = TRUE)
      if (first) {
        plot(roc_obj, col = colors[model_name], lwd = 2, 
             main = "XGBoost 测试集 ROC 曲线")
        first <- FALSE
      } else {
        plot(roc_obj, col = colors[model_name], lwd = 2, add = TRUE)
      }
    }
  }
  legend("bottomright", legend = c("xgbTree", "xgbLinear"), col = colors, lwd = 2)
  dev.off()
  cat("ROC曲线已保存至:", file_path, "\n")
}

# ================== 绘制交叉验证性能图 ==================
plot_cv_performance <- function(cv_output_list, file_path) {
  png(file_path, width = 12, height = 8, units = "in", res = 300)
  
  metrics_names <- c("AUC", "ACC", "SENS", "SPEC", "PPV", "NPV", "F1")
  par(mfrow = c(2, 3), mar = c(4, 4, 3, 2))
  
  for (model_name in names(cv_output_list)) {
    cv_results <- cv_output_list[[model_name]]$results
    if (!is.null(cv_results) && nrow(cv_results) > 0) {
      for (metric in metrics_names) {
        if (metric %in% colnames(cv_results)) {
          values <- cv_results[, metric]
          barplot(values, 
                  names.arg = paste0("Fold", 1:length(values)),
                  main = paste(model_name, metric, "(CV)"),
                  ylab = metric,
                  col = "steelblue",
                  ylim = range(0, max(values, na.rm = TRUE) * 1.1))
          abline(h = mean(values, na.rm = TRUE), col = "red", lwd = 2, lty = 2)
          text(x = length(values)/2 + 0.5, y = mean(values, na.rm = TRUE) + 0.02, 
               labels = sprintf("Mean=%.3f", mean(values, na.rm = TRUE)), 
               col = "red", cex = 0.8)
        }
      }
    }
  }
  
  dev.off()
  cat("交叉验证性能图已保存至:", file_path, "\n")
}

# ================== 绘制特征重要性（仅树模型） ==================
plot_feature_importance <- function(xgb_model, feature_names, top_n = 20, file_path) {
  if (is.null(xgb_model)) return()
  importance_matrix <- xgb.importance(model = xgb_model, feature_names = feature_names)
  if (is.null(importance_matrix) || nrow(importance_matrix) == 0) {
    cat("无法获取特征重要性\n")
    return()
  }
  importance_top <- head(importance_matrix, top_n)
  p <- ggplot(importance_top, aes(x = reorder(Feature, Gain), y = Gain)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = "XGBoost 特征重要性 (Gain)",
         x = "特征", y = "重要性得分") +
    theme_minimal()
  ggsave(file_path, p, width = 10, height = 8)
  cat("特征重要性图已保存至:", file_path, "\n")
}

# ================== 主函数 ==================
run_xgboost_analysis <- function() {
  separator <- paste(rep("=", 60), collapse = "")
  cat(separator, "\nXGBoost模型专项分析开始（含类别权重 + 5折交叉验证）\n", separator, "\n\n")
  
  # 准备数据
  train_prep <- prepare_xgboost_data(train_data)
  test_prep  <- prepare_xgboost_data(test_data)
  
  # 计算类别权重（逆频率加权）
  cancer_count   <- sum(train_prep$y == 1)
  control_count  <- sum(train_prep$y == 0)
  weight_cancer  <- control_count / cancer_count
  weight_control <- 1
  cat(sprintf("训练集类别分布:\n  癌症样本: %d\n  对照样本: %d\n", cancer_count, control_count))
  cat(sprintf("类别权重:\n  癌症样本权重 = %.4f\n  对照样本权重 = %.4f\n", weight_cancer, weight_control))
  sample_weights <- ifelse(train_prep$y == 1, weight_cancer, weight_control)
  
  cat("\n数据信息:\n")
  cat("  训练集: ", nrow(train_prep$x), "样本, ", ncol(train_prep$x), "特征\n")
  cat("  测试集: ", nrow(test_prep$x), "样本, ", ncol(test_prep$x), "特征\n")
  cat("  测试集癌症比例: ", sum(test_prep$y) / length(test_prep$y), "\n\n")
  
  # ================== 5折交叉验证 ==================
  cat(separator, "\n开始5折交叉验证\n", separator, "\n")
  
  cv_output <- list()
  
  # xgbTree交叉验证
  cat("\n--- xgbTree 5折交叉验证 ---\n")
  cv_output[["xgbTree"]] <- perform_cv_xgboost(
    X = train_prep$x,
    y = train_prep$y,
    weights = sample_weights,
    model_type = "xgbTree",
    nfolds = 5
  )
  
  # xgbLinear交叉验证
  cat("\n--- xgbLinear 5折交叉验证 ---\n")
  cv_output[["xgbLinear"]] <- perform_cv_xgboost(
    X = train_prep$x,
    y = train_prep$y,
    weights = sample_weights,
    model_type = "xgbLinear",
    nfolds = 5
  )
  
  # 保存交叉验证结果
  cv_summary_df <- save_cv_summary(cv_output, file.path(DATA_DIR, "xgboost_cv_summary.csv"))
  
  # 打印交叉验证汇总
  cat("\n", separator, "\n", sep="")
  cat("交叉验证结果汇总 (5折平均)\n")
  cat(separator, "\n\n", sep="")
  if (!is.null(cv_summary_df)) {
    print(cv_summary_df)
  }
  
  # ================== 训练最终模型 ==================
  cat("\n", separator, "\n", sep="")
  cat("训练最终模型\n")
  cat(separator, "\n\n", sep="")
  
  results <- list()
  
  # 训练 xgbTree
  cat("训练 xgbTree ...\n")
  results[["xgbTree"]] <- train_xgboost_model(
    train_x = train_prep$x, train_y = train_prep$y,
    test_x  = test_prep$x,  test_y  = test_prep$y,
    model_type = "xgbTree", model_name = "xgbTree",
    sample_weights = sample_weights,
    optimize_threshold = TRUE
  )
  
  # 训练 xgbLinear
  cat("\n训练 xgbLinear ...\n")
  results[["xgbLinear"]] <- train_xgboost_model(
    train_x = train_prep$x, train_y = train_prep$y,
    test_x  = test_prep$x,  test_y  = test_prep$y,
    model_type = "xgbLinear", model_name = "xgbLinear",
    sample_weights = sample_weights,
    optimize_threshold = TRUE
  )
  
  # 保存性能指标
  perf_table <- save_results_to_csv(results, cv_output, 
                                    file.path(DATA_DIR, "xgboost_performance_metrics.csv"))
  
  # 生成报告
  if (!is.null(perf_table)) {
    generate_detailed_report(perf_table, cv_summary_df)
    cat("\n完整性能表格:\n")
    print(perf_table)
  }
  
  # 绘制 ROC 曲线
  plot_roc_curves(results, test_prep$y, file.path(FIG_DIR, "xgboost_roc_curves.png"))
  
  # 绘制交叉验证性能图
  plot_cv_performance(cv_output, file.path(FIG_DIR, "xgboost_cv_performance.png"))
  
  # 绘制特征重要性（仅对 xgbTree 有效）
  if (!is.null(results[["xgbTree"]]$model)) {
    plot_feature_importance(results[["xgbTree"]]$model, 
                            feature_names = colnames(train_prep$x),
                            file_path = file.path(FIG_DIR, "xgboost_feature_importance.png"))
  }
  
  # 保存工作空间
  save.image(file.path(DATA_DIR, "xgboost_analysis_workspace.RData"))
  cat("\n分析完成！结果保存在:", DATA_DIR, "和", FIG_DIR, "\n")
  
  return(list(results = results, performance = perf_table, cv_output = cv_output))
}

# ================== 执行分析 ==================
analysis_output <- run_xgboost_analysis()