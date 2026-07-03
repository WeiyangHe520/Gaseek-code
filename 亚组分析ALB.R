rm(list = ls())
#########################################################
## glmnet模型完整解释与评估代码（类别权重版）           ##
## 功能：glmnet模型训练（类别权重）+SHAP解释+全面评估   ##
##       +亚组分析+可视化                              ##
## 数据要求：CSV格式，最后一列为二分类变量（0/1或因子）##
## 版本：v3.0 (2024-05-25)                            ##
#########################################################

# 加载必要包
library(caret)
library(glmnet)
library(pROC)
library(ggplot2)
library(dplyr)
library(patchwork)
library(shapviz)
library(fastshap)
library(corrplot)
library(rms)
library(viridis)
library(ggrepel)
library(tidyr)
library(plotROC)
set.seed(278)  # 设置随机种子确保可重复性

# 创建输出目录
FIG_DIR <- "glmnet_interpretation_figures(亚组分析)/"
DATA_DIR <- "glmnet_interpretation_data(亚组分析)/"
if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR)
if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR)

### 1. 数据加载与预处理 ###
cat("=== 1. 数据加载与预处理 ===\n")
load(file = ".left_data.rdata")
# 假设加载后存在 train_data 和 test_data，最后一列为目标变量

# 检查数据
cat("训练集样本数:", nrow(train_data), "\n")
cat("测试集样本数:", nrow(test_data), "\n")
cat("特征数量:", ncol(train_data) - 1, "\n")

# 准备glmnet需要的矩阵格式（原始值，glmnet内部会标准化）
x_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data[, ncol(train_data)]
x_test <- as.matrix(test_data[, -ncol(test_data)])
y_test <- test_data[, ncol(test_data)]

# 确保目标变量为数值型（0/1）
if (is.factor(y_train)) {
  y_train_num <- as.numeric(y_train) - 1
  y_test_num <- as.numeric(y_test) - 1
} else {
  y_train_num <- y_train
  y_test_num <- y_test
}

# 同时保留因子格式用于后续评估
y_train_factor <- factor(y_train_num, levels = c(0,1), labels = c("control","cancer"))
y_test_factor <- factor(y_test_num, levels = c(0,1), labels = c("control","cancer"))

### 2. 计算类别权重（基于训练集）###
cat("\n=== 2. 计算类别权重 ===\n")
cancer_count <- sum(y_train_num == 1)
control_count <- sum(y_train_num == 0)
weight_cancer <- control_count / cancer_count
weight_control <- 1
sample_weights <- ifelse(y_train_num == 1, weight_cancer, weight_control)

cat(sprintf("癌症样本数: %d, 对照样本数: %d\n", cancer_count, control_count))
cat(sprintf("类别权重: 癌症=%.3f, 对照=%.3f\n", weight_cancer, weight_control))

### 3. glmnet模型训练（类别权重）###
cat("\n=== 3. glmnet模型训练 ===\n")
alpha_value <- 0.5281
lambda_value <- 0.004597

final_model <- glmnet(x_train, y_train_num,
                      family = "binomial",
                      alpha = alpha_value,
                      lambda = lambda_value,
                      weights = sample_weights,
                      standardize = TRUE,   # 允许glmnet自动标准化
                      intercept = TRUE,
                      thresh = 1e-7,
                      maxit = 1000)

### 4. 模型系数分析 ###
cat("\n=== 4. 模型系数分析 ===\n")
coef_matrix <- coef(final_model)
coef_df <- data.frame(
  feature = rownames(coef_matrix),
  coefficient = as.numeric(coef_matrix),
  stringsAsFactors = FALSE
)
coef_df <- coef_df[-1, ]  # 移除截距
coef_df <- coef_df[coef_df$coefficient != 0, ]
coef_df <- coef_df[order(abs(coef_df$coefficient), decreasing = TRUE), ]
coef_df$feature <- factor(coef_df$feature, levels = rev(coef_df$feature))

write.csv(coef_df, file.path(DATA_DIR, "glmnet_coefficients.csv"), row.names = FALSE)
cat("非零系数特征数量:", nrow(coef_df), "\n")
cat("前10个最重要的特征:\n")
print(head(coef_df, 10))



### 5. 模型预测 ###
cat("\n=== 5. 模型预测 ===\n")
train_pred <- predict(final_model, newx = x_train, type = "response")[,1]
test_pred <- predict(final_model, newx = x_test, type = "response")[,1]

### 6. 全面性能评估（含阈值优化、AUC CI、Brier评分等）###
cat("\n=== 6. 模型性能评估 ===\n")

calculate_metrics_advanced <- function(true_labels, pred_probs, dataset_name,
                                       optimize_threshold = TRUE) {
  # true_labels: 数值型 0/1
  # pred_probs: 预测概率
  # 返回一行数据框
  roc_obj <- roc(true_labels, pred_probs, ci = TRUE, quiet = TRUE)
  auc_val <- auc(roc_obj)
  auc_ci <- ci.auc(roc_obj, conf.level = 0.95)
  
  # 阈值优化（Youden指数），增加容错处理
  if (optimize_threshold) {
    best_coords <- tryCatch({
      coords(roc_obj, "best", ret = "threshold", best.method = "youden")
    }, error = function(e) NULL)
    
    if (!is.null(best_coords) && length(best_coords) > 0) {
      # 处理可能的 data.frame 或命名向量
      if (is.data.frame(best_coords)) best_thresh <- best_coords[1, "threshold"]
      else if (is.numeric(best_coords)) best_thresh <- best_coords[1]
      else best_thresh <- 0.5
    } else {
      best_thresh <- 0.5
    }
    if (is.na(best_thresh)) best_thresh <- 0.5
  } else {
    best_thresh <- 0.5
  }
  
  pred_class <- ifelse(pred_probs > best_thresh, 1, 0)
  # 确保两个因子长度一致且水平相同
  pred_factor <- factor(pred_class, levels = c(0, 1), labels = c("control", "cancer"))
  true_factor <- factor(true_labels, levels = c(0, 1), labels = c("control", "cancer"))
  
  # 检查长度一致性（调试用，可注释）
  if (length(pred_factor) != length(true_factor)) {
    stop(sprintf("长度不一致: pred_factor %d, true_factor %d", 
                 length(pred_factor), length(true_factor)))
  }
  
  cm <- confusionMatrix(pred_factor, true_factor, positive = "cancer")
  
  brier_score <- mean((pred_probs - true_labels)^2)
  ydi <- cm$byClass["Sensitivity"] + cm$byClass["Specificity"] - 1
  f1 <- 2 * (cm$byClass["Pos Pred Value"] * cm$byClass["Sensitivity"]) /
    (cm$byClass["Pos Pred Value"] + cm$byClass["Sensitivity"])
  
  result <- data.frame(
    Dataset = dataset_name,
    Alpha = alpha_value,
    Lambda = lambda_value,
    NonZero_Features = nrow(coef_df),
    Model_Type = "glmnet_weighted",
    AUC = round(auc_val, 3),
    AUC_95CI = sprintf("%.3f (%.3f-%.3f)", auc_val, auc_ci[1], auc_ci[3]),
    AUC_lower = round(auc_ci[1], 3),
    AUC_upper = round(auc_ci[3], 3),
    Brier_Score = round(brier_score, 4),
    ACC = round(cm$overall["Accuracy"], 3),
    SENS = round(cm$byClass["Sensitivity"], 3),
    SPEC = round(cm$byClass["Specificity"], 3),
    PPV = round(cm$byClass["Pos Pred Value"], 3),
    NPV = round(cm$byClass["Neg Pred Value"], 3),
    YDI = round(ydi, 3),
    F1 = round(f1, 3),
    Best_Threshold = round(best_thresh, 4)
  )
  return(result)
}

train_metrics <- calculate_metrics_advanced(y_train_num, train_pred, "Training")
test_metrics <- calculate_metrics_advanced(y_test_num, test_pred, "Testing")
metrics_df <- rbind(train_metrics, test_metrics)
write.csv(metrics_df, file.path(DATA_DIR, "glmnet_performance_metrics.csv"), row.names = FALSE)

cat("\n=== 性能指标汇总 ===\n")
print(metrics_df)




### 8. 亚组分析（基于ALB，保留原函数并微调）###
perform_subgroup_analysis_fixed <- function(data, predictions, true_labels, 
                                            clinical_features = NULL,
                                            output_dir = ".",
                                            outcome_var = "group",
                                            outcome_labels = c("0" = "control", "1" = "cancer")) {
  data$prediction_prob <- predictions
  data[[outcome_var]] <- as.factor(true_labels)
  if (!is.null(outcome_labels)) {
    current_levels <- levels(data[[outcome_var]])
    new_levels <- current_levels
    for (i in seq_along(outcome_labels)) {
      old_name <- names(outcome_labels)[i]
      new_name <- outcome_labels[i]
      idx <- which(current_levels == old_name)
      if (length(idx) > 0) new_levels[idx] <- new_name
    }
    levels(data[[outcome_var]]) <- new_levels
  }
  
  results <- list()
  
  # 自动识别数值型临床特征
  if (is.null(clinical_features)) {
    numeric_vars <- names(data)[sapply(data, is.numeric)]
    numeric_vars <- setdiff(numeric_vars, c("prediction_prob"))
    if (length(numeric_vars) > 0) {
      clinical_features <- numeric_vars[1:min(3, length(numeric_vars))]
    }
  }
  
  # 创建亚组分组（按中位数）
  for (var in clinical_features) {
    if (var %in% names(data)) {
      median_val <- median(data[[var]], na.rm = TRUE)
      data[[paste0(var, "_group")]] <- factor(
        ifelse(data[[var]] > median_val, paste0(var, ">44.5g/L"), paste0(var, "≤44.5g/L")),
        levels = c(paste0(var, "≤44.5g/L"), paste0(var, ">44.5g/L"))
      )
    }
  }
  
  results$plots <- list()
  for (var in clinical_features) {
    group_var <- paste0(var, "_group")
    if (group_var %in% names(data)) {
      outcome_levels <- levels(data[[outcome_var]])
      if (length(outcome_levels) >= 2) {
        p_subgroup <- ggplot(data, aes_string(x = outcome_var, y = "prediction_prob", fill = outcome_var)) +
          geom_boxplot(outlier.shape = NA, width = 0.6, alpha = 0.7) +
          geom_jitter(aes(color = outcome_var), position = position_jitter(width = 0.2, height = 0),
                      size = 1.5, alpha = 0.6) +
          facet_wrap(as.formula(paste("~", group_var)), ncol = 2) +
          labs(y = "Prediction probability", x = "Group") +
          scale_fill_manual(values = c("cancer" = "#e41a1c", "control" = "lightgray"), name = "Group") +
          scale_color_manual(values = c("cancer" = "#e41a1c", "control" = "lightgray"), guide = "none") +
          theme_minimal(base_size = 15) +
          theme(legend.position = "none",
                strip.background = element_rect(fill = "lightgray", color = NA),
                strip.text = element_text(size = 17, face = "bold"),
                axis.text.x = element_text(size = 16, angle = 0, hjust = 0.5),
                axis.text.y = element_text(size = 16),
                axis.title = element_text(size = 15, face = "bold"))
        
        # 添加Wilcox检验p值
        facet_groups <- levels(data[[group_var]])
        for (facet_group in facet_groups) {
          facet_data <- data[data[[group_var]] == facet_group, ]
          if (nrow(facet_data) >= 10 && length(unique(facet_data[[outcome_var]])) >= 2) {
            test_result <- wilcox.test(prediction_prob ~ get(outcome_var), data = facet_data)
            p_value <- test_result$p.value
            sig_symbol <- ifelse(p_value < 0.001, "p<0.001",
                                 ifelse(p_value < 0.01, "p<0.01",
                                        ifelse(p_value < 0.05, "p<0.05", "ns")))
            y_max <- max(facet_data$prediction_prob, na.rm = TRUE)
            y_position <- y_max * 1.05
            
          }
        }
        results$plots[[paste0("subgroup_", var)]] <- p_subgroup
        ggsave(file.path(FIG_DIR, paste0("subgroup_analysis_", var, ".png")),
               p_subgroup, width = 10, height = 4, dpi = 300)
      }
    }
  }
  
  # 计算各亚组性能指标（使用固定阈值0.5，也可改为最佳阈值）
  calculate_subgroup_performance <- function(subgroup_data, subgroup_name) {
    if (nrow(subgroup_data) < 10) return(NULL)
    if (length(unique(subgroup_data[[outcome_var]])) >= 2) {
      roc_obj <- roc(subgroup_data[[outcome_var]], subgroup_data$prediction_prob, quiet = TRUE)
      auc_val <- auc(roc_obj)
      true_numeric <- ifelse(subgroup_data[[outcome_var]] == "control", 0, 1)
      pred_class <- ifelse(subgroup_data$prediction_prob > 0.5, 1, 0)
      cm <- confusionMatrix(factor(pred_class, levels = c(0,1)),
                            factor(true_numeric, levels = c(0,1)), positive = "1")
      metrics <- data.frame(
        Subgroup = subgroup_name,
        N = nrow(subgroup_data),
        AUC = round(auc_val, 3),
        Accuracy = round(cm$overall["Accuracy"], 3),
        Sensitivity = round(cm$byClass["Sensitivity"], 3),
        Specificity = round(cm$byClass["Specificity"], 3),
        PPV = round(cm$byClass["Pos Pred Value"], 3),
        NPV = round(cm$byClass["Neg Pred Value"], 3)
      )
    } else {
      metrics <- data.frame(Subgroup = subgroup_name, N = nrow(subgroup_data),
                            AUC = NA, Accuracy = NA, Sensitivity = NA,
                            Specificity = NA, PPV = NA, NPV = NA)
    }
    return(metrics)
  }
  
  all_metrics <- list()
  all_metrics[["Overall"]] <- calculate_subgroup_performance(data, "Overall")
  for (var in clinical_features) {
    group_var <- paste0(var, "_group")
    if (group_var %in% names(data)) {
      subgroups <- levels(data[[group_var]])
      for (sg in subgroups) {
        sg_data <- data[data[[group_var]] == sg, ]
        metrics <- calculate_subgroup_performance(sg_data, paste0(var, "_", sg))
        if (!is.null(metrics)) all_metrics[[paste0(var, "_", sg)]] <- metrics
      }
    }
  }
  metrics_df <- do.call(rbind, all_metrics)
  results$performance_metrics <- metrics_df
  
  # 绘制AUC比较图
  if (nrow(metrics_df) > 1) {
    auc_plot_data <- metrics_df %>% filter(!is.na(AUC) & Subgroup != "Overall") %>% arrange(AUC)
    auc_plot_data$Subgroup <- factor(auc_plot_data$Subgroup, levels = auc_plot_data$Subgroup)
    
    # 性能热图
    if (nrow(metrics_df) > 2) {
      heatmap_data <- metrics_df %>% filter(Subgroup != "Overall") %>%
        select(Subgroup, AUC, Accuracy, Sensitivity, Specificity) %>%
        pivot_longer(cols = -Subgroup, names_to = "Metric", values_to = "Value")
    }
  }
  
  write.csv(metrics_df, file.path(DATA_DIR, "subgroup_analysis_metrics.csv"), row.names = FALSE)
  return(results)
}

# 执行亚组分析（以ALB为例）
cat("\n=== 7. 亚组分析（基于ALB） ===\n")
# 准备测试集数据框（包含原始ALB）
test_data_for_subgroup <- as.data.frame(test_data)
# 如果ALB不在列中，请确保列名正确；此处假设存在ALB列
if ("ALB" %in% colnames(test_data_for_subgroup)) {
  subgroup_results <- perform_subgroup_analysis_fixed(
    data = test_data_for_subgroup,
    predictions = test_pred,
    true_labels = y_test_factor,   # 因子格式 "control"/"cancer"
    clinical_features = "ALB",
    output_dir = "./subgroup_analysis_results",
    outcome_var = "group",
    outcome_labels = c("0" = "control", "1" = "cancer")
  )
  cat("\n亚组分析性能指标:\n")
  print(subgroup_results$performance_metrics)
} else {
  cat("注意：测试集中未找到'ALB'列，跳过亚组分析。\n")
}

### 9. 保存模型与最终输出 ###
save(final_model, file = file.path(DATA_DIR, "glmnet_weighted_model.rdata"))
cat("\n模型已保存至:", file.path(DATA_DIR, "glmnet_weighted_model.rdata"))
cat("\n=== 所有分析完成 ===\n")
### 7. 亚组截断点生成（仅保存CSV，无重复函数）###
cat("\n=== 7. 亚组截断点生成 ===\n")

# 确保测试数据框存在
if (exists("test_data_for_subgroup") && "ALB" %in% colnames(test_data_for_subgroup)) {
  
  # 提取 ALB 列（可根据需要扩展为多个临床特征）
  var <- "ALB"
  data <- test_data_for_subgroup
  
  # 计算中位数及分组信息
  median_val <- median(data[[var]], na.rm = TRUE)
  min_val <- min(data[[var]], na.rm = TRUE)
  max_val <- max(data[[var]], na.rm = TRUE)
  low_n <- sum(data[[var]] <= median_val, na.rm = TRUE)
  high_n <- sum(data[[var]] > median_val, na.rm = TRUE)
  
  # 输出到控制台
  cat(sprintf("\n特征 '%s':\n", var))
  cat(sprintf("  中位数截断点: %.3f\n", median_val))
  cat(sprintf("  数据范围: [%.3f, %.3f]\n", min_val, max_val))
  cat(sprintf("  低表达组 (%s_low): ≤ %.3f (n=%d)\n", var, median_val, low_n))
  cat(sprintf("  高表达组 (%s_high): > %.3f (n=%d)\n", var, median_val, high_n))
  
  # 构建截断点信息数据框
  cutoff_info <- data.frame(
    Feature = var,
    Median = round(median_val, 3),
    Min_Value = round(min_val, 3),
    Max_Value = round(max_val, 3),
    Cutoff_Point = round(median_val, 3),
    Low_Group_Definition = sprintf("≤ %.3f", median_val),
    High_Group_Definition = sprintf("> %.3f", median_val),
    Low_Group_N = low_n,
    High_Group_N = high_n,
    stringsAsFactors = FALSE
  )
  
  # 保存 CSV 文件
  write.csv(cutoff_info, 
            file.path(DATA_DIR, "subgroup_cutoff_information.csv"), 
            row.names = FALSE)
  
  cat("\n截断点信息已保存至: ", file.path(DATA_DIR, "subgroup_cutoff_information.csv"), "\n")
  cat("========================\n")
  
  # 可选：打印结果
  print(cutoff_info)
  
} else {
  cat("注意：测试集中未找到'ALB'列，或 test_data_for_subgroup 不存在，跳过截断点生成。\n")
}
