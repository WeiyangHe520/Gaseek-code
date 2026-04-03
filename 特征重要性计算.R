###############################################
## 临床预测模型开发教学代码                 ##
## 功能：多特征筛选+多模型比较+可视化       ##
## 数据要求：CSV格式，最后一列为分组变量   ##
## 作者：罗怀超                              ##
## 版本：v2.1 (2024-05-25)                  ##
# devtools::install_github('Huaichao2018/Clabomic')
# 鸣谢部分请加We gratefully acknowledge the multidisciplinary collaboration provided by the Intelligent Clinlabomics Research Elites (iCARE) consortium. 
#cited Luo H, et al. Signal Transduct Target Ther. 2022 Oct 10;7(1):348. doi: 10.1038/s41392-022-01169-7. PMID: 36210387; PMCID: PMC9548502.
#cited Wen X, et al. Clinlabomics: leveraging clinical laboratory data by data mining strategies. BMC Bioinformatics. 2022 Sep 24;23(1):387. doi: 10.1186/s12859-022-04926-1. PMID: 36153474; PMCID: PMC9509545.
#cited Kawakami E, et al. Application of artificial intelligence for preoperative diagnostic and prognostic prediction in epithelial ovarian cancer based on blood biomarkers.Clin Cancer Res 2019; 25: 3006–15.
#本代码为基础代码，方法来自STTT PMID: 36210387，数据来自临床化学PMID: 38431275。
###############################################
rm(list = ls())
set.seed(3456)  # For reproducibility
FIG_DIR <- "figures/"    # Output directory for figures
DATA_DIR <- "data/"      # Output directory for data
dir.create(FIG_DIR, showWarnings = FALSE)
dir.create(DATA_DIR, showWarnings = FALSE)
##算法
###--------------###--------------###--------------###--------------###--------------
library(caret)
library(tidyverse)
library(viridis)
library(ggprism)
library(pROC)
load(file = ".left_data（四川内部）.rdata")
##基于test_data的算法比较
###--------------###--------------###--------------###--------------###--------------
# Model training and evaluation module
# -----------------------定义函数
#Calculate classification performance metrics

four_stats <- function(data, lev = levels(data$obs), model = NULL) {
  # Calculate ROC metrics and distance from perfect model
  out <- c(twoClassSummary(data, lev = levels(data$obs), model = NULL))
  
  coords <- matrix(c(1, 1, out["Spec"], out["Sens"]), 
                   ncol = 2, byrow = TRUE)
  colnames(coords) <- c("Spec", "Sens")
  rownames(coords) <- c("Best", "Current")
  c(out, Dist = dist(coords)[1])
}

#' Fill confusion matrix with zeros for missing categories
fill_confusion_matrix <- function(res) {
  res2 <- res
  cnames <- colnames(res)
  l <- 1
  
  for (j in 1:nrow(res)) {
    if (l <= ncol(res)) {
      if (rownames(res)[j] != colnames(res)[l]) {
        res2 <- cbind(res2, rep(0, nrow(res)))
        cnames <- c(cnames, rownames(res)[j])
      } else {
        l <- l + 1
      }
    } else {
      res2 <- cbind(res2, rep(0, nrow(res)))
      cnames <- c(cnames, rownames(res)[j])
    }
  }
  
  colnames(res2) <- cnames
  res2 <- res2[order, order]
  rownames(res2) <- label
  colnames(res2) <- label
  as.table(res2)
}

#' Calculate accuracy from confusion matrix
calc_accuracy <- function(res) {
  sum(diag(res)) / sum(res)
}

#' Train and evaluate multiple machine learning models
#' 完整模型训练和评估函数 - 使用默认参数版
train_models <- function(train_data, test_data, features, 
                         method_list = c("multinom","gamSpline","lda2","glmStepAIC","mlp","svmRadial","xgbTree","rf","xgbLinear"),
                         k_fold = 10, repeats = 1) {
  
  # 设置训练控制参数 - 关闭参数调优
  fit_control <- trainControl(
    method = "repeatedcv",
    number = k_fold,
    repeats = repeats,
    savePredictions = TRUE,
    classProbs = TRUE,
    summaryFunction = four_stats,
    verboseIter = TRUE
  )
  
  # 准备结果存储
  results <- list()
  models <- list()
  roc_curves <- list()
  predictions_df <- test_data
  
  # 确保结果列名为"class"
  colnames(train_data)[ncol(train_data)] <- "class"
  colnames(test_data)[ncol(test_data)] <- "class"
  
  cat("=== Training Single-Feature Models ===\n")
  # 训练单特征模型 - 使用glm
  for (feature in features) {
    cat("Training model for feature:", feature, "\n")
    
    train_subset <- train_data[, c(feature, "class")]
    
    model <- tryCatch({
      train(
        class ~ .,
        data = train_subset,
        method = "glm",  # 使用glm用于单特征
        family = "binomial",
        trControl = fit_control,
        metric = "Dist"
      )
    }, error = function(e) {
      cat("Error with feature", feature, ":", e$message, "\n")
      return(NULL)
    })
    
    if (!is.null(model)) {
      # 在测试集上预测
      pred_prob <- predict(model, test_data, type = "prob")[, 2]
      predictions_df[[paste0(feature, "_single")]] <- pred_prob
      
      # 计算ROC
      roc_res <- roc(test_data$class, pred_prob)
      roc_curves[[paste0(feature, "_single")]] <- roc_res
      
      # 存储结果
      results[[paste0(feature, "_single")]] <- list(
        model = model,
        roc = roc_res,
        auc = auc(roc_res)
      )
    }
  }
  
  cat("\n=== Training Multi-Feature Models ===\n")
  # 训练多特征模型 - 所有模型都使用默认参数
  for (method in method_list) {
    cat("Training model:", method, "\n")
    
    tryCatch({
      # 所有模型都使用默认参数，不进行参数调优
      if (method %in% c("glm", "glmnet")) {
        # 对于广义线性模型，使用family参数
        model <- train(
          class ~ .,
          data = train_data,
          method = method,
          family = "binomial",
          trControl = fit_control,
          metric = "Dist",
          tuneGrid = data.frame(.parameter = "default")  # 使用默认参数
        )
      } else {
        # 其他模型 - 使用默认参数
        model <- train(
          class ~ .,
          data = train_data,
          method = method,
          trControl = fit_control,
          metric = "Dist",
          tuneLength = 1  # 只使用默认参数，不进行调优
        )
      }
      
      # 存储模型
      models[[method]] <- model
      
      # 测试集预测
      pred_prob <- predict(model, test_data, type = "prob")[, 2]
      pred_class <- predict(model, test_data)
      predictions_df[[paste0("all_", method)]] <- pred_prob
      
      # 性能指标
      roc_res <- roc(test_data$class, pred_prob)
      roc_curves[[paste0("all_", method)]] <- roc_res
      
      # 计算准确率
      conf_matrix <- table(Predicted = pred_class, Actual = test_data$class)
      accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      
      results[[method]] <- list(
        model = model,
        roc = roc_res,
        auc = auc(roc_res),
        accuracy = accuracy,
        confusion = conf_matrix,
        predictions = data.frame(
          actual = test_data$class,
          predicted = pred_class,
          probability = pred_prob
        )
      )
      
      cat("  -", method, "AUC:", round(auc(roc_res), 4), "Accuracy:", round(accuracy, 4), "\n")
      
    }, error = function(e) {
      cat("  - Error training", method, "model:", e$message, "\n")
      results[[method]] <- list(error = e$message)
    })
  }
  
  # 创建性能比较总结
  performance_summary <- create_performance_summary(results)
  
  return(list(
    results = results,
    test_data = predictions_df,
    roc_curves = roc_curves,
    models = models,
    performance_summary = performance_summary
  ))
}

#' 创建性能总结
create_performance_summary <- function(results) {
  summary_df <- data.frame(
    Model = character(),
    AUC = numeric(),
    Accuracy = numeric(),
    Features = character(),
    stringsAsFactors = FALSE
  )
  
  for (model_name in names(results)) {
    model_result <- results[[model_name]]
    
    if (!is.null(model_result$auc) && !is.null(model_result$accuracy)) {
      features <- ifelse(grepl("_single$", model_name), "Single", "Multi")
      
      summary_df <- rbind(summary_df, data.frame(
        Model = model_name,
        AUC = model_result$auc,
        Accuracy = model_result$accuracy,
        Features = features,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  return(summary_df)
}

#' 绘制ROC曲线比较图
plot_roc_comparison <- function(model_results, top_n = 10) {
  roc_curves <- model_results$roc_curves
  
  # 选择性能最好的模型
  auc_values <- sapply(roc_curves, function(x) auc(x))
  top_models <- names(sort(auc_values, decreasing = TRUE)[1:min(top_n, length(auc_values))])
  
  # 创建比较图
  p <- ggroc(roc_curves[top_models], size = 1) +
    theme_minimal() +
    scale_color_viridis_d() +
    geom_abline(intercept = 1, slope = 1, linetype = "dashed", alpha = 0.5) +
    labs(title = "ROC Curves Comparison",
         color = "Models") +
    theme(legend.position = "right")
  
  # 添加AUC标注
  auc_df <- data.frame(
    Model = top_models,
    AUC = round(auc_values[top_models], 3)
  )
  
  print(auc_df)
  
  return(p)
}

# Train models
str(train_data)
str(test_data)
train_data$group=factor(train_data$group,levels = c("control" ,"cancer"))
test_data$group=factor(test_data$group,levels = c("control" ,"cancer"))
model_results <- train_models(
  train_data = train_data, 
  test_data = test_data, 
  features = colnames(train_data)[-ncol(train_data)]
)
# Save model results
save(model_results, file = file.path(DATA_DIR, "model_results.rdata"))
write.csv(model_results$test_data, file.path(DATA_DIR, "test_predictions.csv"), row.names = FALSE)

# Variable importance module
# -------------------------

#' Extract and plot variable importance across models
plot_variable_importance <- function(models, 
                                     font_size = 14, 
                                     title_size = 16,
                                     legend_text_size = 12,
                                     bar_width = 0.7) {
  
  importance_list <- lapply(names(models), function(model_name) {
    model <- models[[model_name]]
    tryCatch({
      if (!is.null(model$finalModel)) {
        imp <- varImp(model, scale = TRUE)
        
        if (inherits(imp, "varImp.train")) {
          imp_df <- imp$importance
        } else {
          imp_df <- as.data.frame(imp)
        }
        
        data.frame(
          Feature = rownames(imp_df),
          Importance = imp_df$Overall,
          Model = model_name
        )
      }
    }, error = function(e) {
      message(paste("Cannot calculate importance for", model_name, "model:", e$message))
      NULL
    })
  }) %>% 
    purrr::compact() %>%
    bind_rows()
  
  if (nrow(importance_list) == 0) {
    message("No importance data available")
    return(ggplot() + geom_blank() + labs(title = "No importance data"))
  }
  
  feature_order <- importance_list %>%
    group_by(Feature) %>%
    summarise(Total = sum(Importance)) %>%
    arrange(desc(Total)) %>%
    pull(Feature)
  
  importance_list$Feature <- factor(importance_list$Feature, levels = feature_order)
  
  # 计算模型数量用于调整条宽
  model_count <- length(unique(importance_list$Model))
  dodge_width <- ifelse(model_count > 1, bar_width, 0)
  
  p <- ggplot(importance_list, aes(x = Feature, y = Importance, fill = Model)) +
    geom_bar(stat = "identity", position = position_dodge(width = dodge_width), 
             width = bar_width) +
    scale_fill_brewer(palette = "Paired") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = font_size),
      axis.text.y = element_text(size = font_size),
      axis.title.x = element_text(size = title_size, face = "bold", margin = margin(t = 10)),
      axis.title.y = element_text(size = title_size, face = "bold", margin = margin(r = 10)),
      plot.title = element_text(size = title_size + 2, face = "bold", hjust = 0.5),
      legend.title = element_text(size = title_size, face = "bold"),
      legend.text = element_text(size = legend_text_size),
      legend.position = "right",
      panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
      panel.grid.minor = element_blank(),
      plot.margin = margin(20, 20, 20, 20)  # 上右下左边距
    ) +
    labs(
      title = "Variable Importance Across Models",
      y = "Relative Importance", 
      x = "Feature"
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05)))  # 设置y轴扩展
  
  return(p)
}

# Generate and save importance plot with custom settings
importance_plot <- plot_variable_importance(
  models = model_results$models,
  font_size = 14,           # 坐标轴标签字体大小
  title_size = 15,          # 标题字体大小
  legend_text_size = 14,     # 图例字体大小
  bar_width = 0.8          # 条形宽度
)

# 打印图形
print(importance_plot)

# 保存为PNG格式 - 支持多种图片格式
# 设置图片大小和分辨率
img_width <- 18    # 宽度（英寸）
img_height <- 8    # 高度（英寸）
img_dpi <- 300     # 分辨率（每英寸点数）

# 保存为PNG格式
ggsave(
  filename = file.path(FIG_DIR, "variable_importance.png"),
  plot = importance_plot,
  width = img_width,
  height = img_height,
  dpi = img_dpi,
  bg = "white"  # 设置背景为白色
)

# 同时保存为高分辨率PDF供出版使用（可选）
ggsave(
  filename = file.path(FIG_DIR, "variable_importance_highres.pdf"),
  plot = importance_plot,
  width = img_width,
  height = img_height,
  device = "pdf"
)

# 或者保存为其他格式（根据需要取消注释）
# TIFF格式（适合出版）
# ggsave(
#   filename = file.path(FIG_DIR, "variable_importance.tiff"),
#   plot = importance_plot,
#   width = img_width,
#   height = img_height,
#   dpi = img_dpi,
#   compression = "lzw"
# )

# JPEG格式
# ggsave(
#   filename = file.path(FIG_DIR, "variable_importance.jpg"),
#   plot = importance_plot,
#   width = img_width,
#   height = img_height,
#   dpi = img_dpi,
#   quality = 95  # JPEG质量（0-100）
# )

cat(sprintf("\nVariable importance plot saved as:\n"))
cat(sprintf("• PNG: %s (%.0f x %.0f inches, %d dpi)\n", 
            file.path(FIG_DIR, "variable_importance.png"), 
            img_width, img_height, img_dpi))
cat(sprintf("• PDF: %s (high resolution version)\n", 
            file.path(FIG_DIR, "variable_importance_highres.pdf")))

