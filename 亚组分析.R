rm(list = ls())
#########################################################
## glmnet模型完整解释与评估代码                        ##
## 功能：glmnet模型训练+SHAP解释+全面评估+可视化      ##
## 数据要求：CSV格式，最后一列为二分类变量（0/1或因子）##
## 作者：基于临床预测模型代码优化                     ##
## 版本：v2.0 (2024-05-25)                            ##
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
# 这里假设已经加载了训练集和测试集
# train_data, test_data
# 最后一列为目标变量，其他列为特征

# 检查数据
cat("训练集样本数:", nrow(train_data), "\n")
cat("测试集样本数:", nrow(test_data), "\n")
cat("特征数量:", ncol(train_data) - 1, "\n")

# 准备glmnet需要的矩阵格式
x_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data[, ncol(train_data)]
x_test <- as.matrix(test_data[, -ncol(test_data)])
y_test <- test_data[, ncol(test_data)]

# 确保目标变量为数值型（0/1）
if (is.factor(y_train)) {
  y_train <- as.numeric(y_train) - 1
  y_test <- as.numeric(y_test) - 1
}

### 2. glmnet模型训练 ###
cat("\n=== 2. glmnet模型训练 ===\n")

# 使用最优lambda训练最终模型
final_model <- glmnet(x_train, y_train,
                      family = "binomial",
                      alpha = 0.5281,
                      lambda = 0.004597)

### 3. 模型系数分析 ###
cat("\n=== 3. 模型系数分析 ===\n")

# 提取系数
coef_matrix <- coef(final_model, s = 0.004597)
coef_df <- data.frame(
  feature = rownames(coef_matrix),
  coefficient = as.numeric(coef_matrix),
  stringsAsFactors = FALSE
)

# 移除截距并排序
coef_df <- coef_df[-1, ]  # 移除截距
coef_df <- coef_df[coef_df$coefficient != 0, ]
coef_df <- coef_df[order(abs(coef_df$coefficient), decreasing = TRUE), ]
coef_df$feature <- factor(coef_df$feature, levels = rev(coef_df$feature))

# 保存系数
write.csv(coef_df, file.path(DATA_DIR, "glmnet_coefficients.csv"), row.names = FALSE)

cat("非零系数特征数量:", nrow(coef_df), "\n")
cat("前10个最重要的特征:\n")
print(head(coef_df, 10))

# 绘制系数图
p_coef <- ggplot(coef_df, aes(x = coefficient, y = feature, fill = coefficient > 0)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  scale_fill_manual(values = c("TRUE" = "#2ca25f", "FALSE" = "#e34a33"),
                    name = "positive effect") +
  geom_text(aes(label = sprintf("%.3f", coefficient),
                hjust = ifelse(coefficient > 0, -0.1, 1.1)),
            size = 4) +
  labs(
    title = "Feature coefficients of glmnet",
    x = "coefficient value",
    y = "feature"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        panel.grid.major.y = element_blank())

ggsave(file.path(FIG_DIR, "glmnet_coefficients.pdf"), p_coef, width = 10, height = 12)
### 4. 模型性能评估 ###
cat("\n=== 4. 模型性能评估 ===\n")

# 预测概率
train_pred <- predict(final_model, newx = x_train, type = "response")[, 1]
test_pred <- predict(final_model, newx = x_test, type = "response")[, 1]

# ROC曲线分析
roc_train <- roc(y_train, train_pred)
roc_test <- roc(y_test, test_pred)

# 性能指标计算
calculate_metrics <- function(true, pred, threshold = 0.5) {
  pred_class <- ifelse(pred > threshold, 1, 0)
  cm <- confusionMatrix(as.factor(pred_class), as.factor(true), positive = "1")
  
  metrics <- c(
    AUC = auc(roc(true, pred)),
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    PPV = cm$byClass["Pos Pred Value"],
    NPV = cm$byClass["Neg Pred Value"],
    F1 = 2 * (cm$byClass["Pos Pred Value"] * cm$byClass["Sensitivity"]) /
      (cm$byClass["Pos Pred Value"] + cm$byClass["Sensitivity"])
  )
  
  return(round(metrics, 3))
}

train_metrics <- calculate_metrics(y_train, train_pred)
test_metrics <- calculate_metrics(y_test, test_pred)

# 保存性能指标
metrics_df <- data.frame(
  Metric = names(train_metrics),
  Training = train_metrics,
  Test = test_metrics
)
write.csv(metrics_df, file.path(DATA_DIR, "glmnet_performance_metrics.csv"), row.names = FALSE)

# 绘制ROC曲线
p_roc <- ggplot() +
  geom_roc(aes(d = y_train, m = train_pred), 
           color = "#4daf4a", n.cuts = 0) +
  geom_roc(aes(d = y_test, m = test_pred), 
           color = "#377eb8", n.cuts = 0) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(
    title = "glmnet模型ROC曲线",
    x = "1 - 特异度",
    y = "敏感度"
  ) +
  scale_color_manual(values = c("训练集" = "#4daf4a", "测试集" = "#377eb8")) +
  theme_minimal(base_size = 14) +
  theme(legend.position = c(0.8, 0.2)) +
  annotate("text", x = 0.6, y = 0.3,
           label = paste("训练集AUC:", round(auc(roc_train), 3)),
           color = "#4daf4a", size = 5) +
  annotate("text", x = 0.6, y = 0.2,
           label = paste("测试集AUC:", round(auc(roc_test), 3)),
           color = "#377eb8", size = 5)

ggsave(file.path(FIG_DIR, "glmnet_roc_curve.pdf"), p_roc, width = 8, height = 8)
### 修正后的亚组分析函数 ###
perform_subgroup_analysis_fixed <- function(data, predictions, true_labels, 
                                            clinical_features = NULL,
                                            output_dir = ".",  # 新增参数
                                            outcome_var = "group",
                                            outcome_labels = c("0" = "control", "1" = "cancer")) {
  
  
  
  # 确保数据格式正确
  data$prediction_prob <- predictions
  data[[outcome_var]] <- as.factor(true_labels)  # 转换为因子
  
  # 应用标签重命名（如果提供了outcome_labels）
  if (!is.null(outcome_labels) && length(outcome_labels) > 0) {
    # 重命名因子水平
    current_levels <- levels(data[[outcome_var]])
    new_levels <- current_levels
    
    for (i in seq_along(outcome_labels)) {
      old_name <- names(outcome_labels)[i]
      new_name <- outcome_labels[i]
      idx <- which(current_levels == old_name)
      if (length(idx) > 0) {
        new_levels[idx] <- new_name
      }
    }
    
    levels(data[[outcome_var]]) <- new_levels
  }
  
  # 创建结果列表
  results <- list()
  
  # 1. 自动识别数值型临床特征
  if (is.null(clinical_features)) {
    numeric_vars <- names(data)[sapply(data, is.numeric)]
    numeric_vars <- setdiff(numeric_vars, c("prediction_prob"))
    
    if (length(numeric_vars) > 0) {
      clinical_features <- numeric_vars[1:min(3, length(numeric_vars))]
    }
  }
  
  # 2. 创建亚组分组
  for (var in clinical_features) {
    if (var %in% names(data)) {
      median_val <- median(data[[var]], na.rm = TRUE)
      data[[paste0(var, "_group")]] <- factor(
        ifelse(data[[var]] > median_val, 
               paste0(var, "_high"), 
               paste0(var, "_low")),
        levels = c(paste0(var, "_low"), paste0(var, "_high"))
      )
    }
  }
  
  # 3. 创建亚组可视化
  results$plots <- list()
  
  for (var in clinical_features) {
    group_var <- paste0(var, "_group")
    
    if (group_var %in% names(data)) {
      # 检查是否有足够的因子水平
      outcome_levels <- levels(data[[outcome_var]])
      
      if (length(outcome_levels) >= 2) {
        # 创建分组箱线图
        p_subgroup <- ggplot(data, aes_string(x = outcome_var, 
                                              y = "prediction_prob", 
                                              fill = outcome_var)) +
          geom_boxplot(outlier.shape = NA, width = 0.6, alpha = 0.7) +
          geom_jitter(aes(color = outcome_var), 
                      position = position_jitter(width = 0.2, height = 0),
                      size = 1.5, alpha = 0.6) +
          facet_wrap(as.formula(paste("~", group_var)), ncol = 2) +
          labs(
            y = "Prediction probability",
            x = "Group"
          ) +
          scale_fill_manual(values = c("cancer" = "#E74C3C", "control" = "#3498DB"),
                            name = "Group") +
          scale_color_manual(values = c("cancer" = "#E74C3C", "control" = "#3498DB"),
                             guide = "none") +
          theme_minimal(base_size = 15) +
          theme(
            legend.position = "NONE",
            strip.background = element_rect(fill = "lightgray", color = NA),
            strip.text = element_text(size = 17, face = "bold"),
            axis.text.x = element_text(size = 16,angle = 0, hjust = 0.5),
            axis.text.y = element_text(size = 16,angle = 0, hjust = 0.5),
            axis.title= element_text(size = 15, face = "bold")
          )
        
        # 添加显著性标记
        facet_groups <- levels(data[[group_var]])
        
        for (facet_group in facet_groups) {
          facet_data <- data[data[[group_var]] == facet_group, ]
          
          if (nrow(facet_data) >= 10 && 
              length(unique(facet_data[[outcome_var]])) >= 2) {
            
            test_result <- wilcox.test(
              prediction_prob ~ get(outcome_var), 
              data = facet_data
            )
            
            p_value <- test_result$p.value
            sig_symbol <- ifelse(p_value < 0.001, "p<0.001",
                                 ifelse(p_value < 0.01, "p<0.01",
                                        ifelse(p_value < 0.05, "p<0.05", "ns")))
            
            y_max <- max(facet_data$prediction_prob, na.rm = TRUE)
            y_position <- y_max * 1.05
            
            p_subgroup <- p_subgroup + 
              geom_text(
                data = data.frame(
                  x = 1.5,
                  y = y_position,
                  label = sig_symbol,
                  group_var = facet_group
                ),
                aes(x = x, y = y, label = label),
                inherit.aes = FALSE,
                size = 5,
                fontface = "bold",
                color = ifelse(p_value < 0.05, "white", "gray")
              )
          }
        }
        
        results$plots[[paste0("subgroup_", var)]] <- p_subgroup
        
        # 保存单个图
        ggsave(file.path(FIG_DIR, paste0("subgroup_analysis_", var, ".png")),
               p_subgroup, width = 10, height = 4, dpi = 300)
      }
    }
  }
  
  # 4. 计算亚组性能指标
  calculate_subgroup_performance <- function(subgroup_data, subgroup_name) {
    if (nrow(subgroup_data) < 10) return(NULL)
    
    if (length(unique(subgroup_data[[outcome_var]])) >= 2) {
      # 计算ROC
      roc_obj <- roc(subgroup_data[[outcome_var]], 
                     subgroup_data$prediction_prob,
                     quiet = TRUE)
      auc_val <- auc(roc_obj)
      
      # 计算混淆矩阵
      # 注意：这里需要将control/cancer转换回0/1进行计算
      true_labels_numeric <- ifelse(subgroup_data[[outcome_var]] == "control", 0, 1)
      pred_class <- ifelse(subgroup_data$prediction_prob > 0.5, 1, 0)
      
      cm <- confusionMatrix(factor(pred_class, levels = c(0, 1)),
                            factor(true_labels_numeric, levels = c(0, 1)),
                            positive = "1")
      
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
      metrics <- data.frame(
        Subgroup = subgroup_name,
        N = nrow(subgroup_data),
        AUC = NA,
        Accuracy = NA,
        Sensitivity = NA,
        Specificity = NA,
        PPV = NA,
        NPV = NA
      )
    }
    
    return(metrics)
  }
  
  # 计算各亚组性能
  all_metrics <- list()
  all_metrics[["Overall"]] <- calculate_subgroup_performance(data, "Overall")
  
  for (var in clinical_features) {
    group_var <- paste0(var, "_group")
    
    if (group_var %in% names(data)) {
      subgroups <- levels(data[[group_var]])
      
      for (sg in subgroups) {
        sg_data <- data[data[[group_var]] == sg, ]
        metrics <- calculate_subgroup_performance(sg_data, 
                                                  paste0(var, "_", sg))
        
        if (!is.null(metrics)) {
          all_metrics[[paste0(var, "_", sg)]] <- metrics
        }
      }
    }
  }
  
  # 合并结果
  metrics_df <- do.call(rbind, all_metrics)
  results$performance_metrics <- metrics_df
  
  # 5. 创建亚组性能总结图
  if (nrow(metrics_df) > 1) {
    # 5.1 AUC比较图
    auc_plot_data <- metrics_df %>%
      filter(!is.na(AUC) & Subgroup != "Overall") %>%
      arrange(AUC)
    
    auc_plot_data$Subgroup <- factor(auc_plot_data$Subgroup,
                                     levels = auc_plot_data$Subgroup)
    
    p_auc <- ggplot(auc_plot_data, aes(x = Subgroup, y = AUC)) +
      geom_bar(stat = "identity", fill = "#2ca25f", alpha = 0.7, width = 0.6) +
      geom_text(aes(label = sprintf("%.3f\n(n=%d)", AUC, N)),
                vjust = -0.3, size = 3.5) +
      geom_hline(yintercept = metrics_df$AUC[metrics_df$Subgroup == "Overall"],
                 linetype = "dashed", color = "red", size = 1) +
      labs(
        title = "Comparison of AUC across subgroups",
        subtitle = paste("Overall AUC:", 
                         round(metrics_df$AUC[metrics_df$Subgroup == "Overall"], 3)),
        x = "subgroup",
        y = "AUC"
      ) +
      ylim(0, min(1.1, max(auc_plot_data$AUC, na.rm = TRUE) * 1.15)) +
      theme_minimal(base_size = 12) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid.major.x = element_blank()
      )
    
    results$plots$auc_comparison <- p_auc
    ggsave(file.path(FIG_DIR, "subgroup_auc_comparison.pdf"),
           p_auc, width = 10, height = 6)
    
    # 5.2 性能热图
    if (nrow(metrics_df) > 2) {
      heatmap_data <- metrics_df %>%
        filter(Subgroup != "Overall") %>%
        select(Subgroup, AUC, Accuracy, Sensitivity, Specificity) %>%
        pivot_longer(cols = -Subgroup,
                     names_to = "Metric",
                     values_to = "Value")
      
      p_heatmap <- ggplot(heatmap_data, aes(x = Metric, y = Subgroup, fill = Value)) +
        geom_tile(color = "white", size = 0.5) +
        geom_text(aes(label = ifelse(is.na(Value), "NA", sprintf("%.3f", Value))),
                  color = "black", size = 3) +
        scale_fill_gradient2(low = "#e34a33",
                             mid = "white",
                             high = "#2ca25f",
                             midpoint = 0.5,
                             na.value = "gray90",
                             name = "Performance metrics") +
        labs(
          title = "Heat map of model performance for each subgroup",
          x = "Performance metrics",
          y = "Subgroup"
        ) +
        theme_minimal(base_size = 11) +
        theme(
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "right"
        )
      
      results$plots$performance_heatmap <- p_heatmap
      ggsave(file.path(FIG_DIR, "subgroup_performance_heatmap.pdf"),
             p_heatmap, width = 8, height = 6)
    }
  }
  
  # 保存结果
  write.csv(metrics_df,
            file.path(DATA_DIR, "subgroup_analysis_metrics.csv"),
            row.names = FALSE)
  
  return(results)
}

### 在模型评估后调用亚组分析 ###

# 准备数据
test_data_for_subgroup <- as.data.frame(test_data)

# 假设你的测试集标签是y_test（0/1格式）
# 创建重命名的标签
if (exists("y_test")) {
  true_labels_named <- ifelse(y_test == 0, "control", "cancer")
  true_labels_named <- factor(true_labels_named, levels = c("control", "cancer"))
  
  # 更新数据中的标签
  test_data_for_subgroup$group <- true_labels_named
  
  # 执行亚组分析
  cat("\n=== 亚组分析（HGB分组） ===\n")
  
  # 使用修正后的函数
  subgroup_results <- perform_subgroup_analysis_fixed(
    data = test_data_for_subgroup,
    predictions = test_pred,  # 你的预测概率
    true_labels = true_labels_named,  # 使用重命名后的标签
    clinical_features = "HGB",  # 只分析HGB
    output_dir = "./subgroup_analysis_results",  # 指定输出目录
    outcome_var = "group",
    outcome_labels = c("0" = "control", "1" = "cancer")  # 确保标签正确
  )
  
  # 打印结果
  if (!is.null(subgroup_results$performance_metrics)) {
    cat("\n亚组分析性能指标:\n")
    print(subgroup_results$performance_metrics)
    
    # 保存详细结果
    saveRDS(subgroup_results,
            file.path("./subgroup_analysis_results/data", 
                      "subgroup_analysis_HGB_results.rds"))
    
    # 创建综合报告
    if (length(subgroup_results$plots) > 0) {
      # 将主要图表组合
      if ("subgroup_HGB" %in% names(subgroup_results$plots)) {
        if ("auc_comparison" %in% names(subgroup_results$plots)) {
          # 使用patchwork包需要先安装
          if (!require("patchwork")) install.packages("patchwork")
          library(patchwork)
          
          p_combined <- subgroup_results$plots$subgroup_HGB / 
            subgroup_results$plots$auc_comparison +
            plot_layout(heights = c(2, 1)) +
            plot_annotation(title = "HGB亚组分析综合报告",
                            theme = theme(plot.title = element_text(size = 14, face = "bold")))
          
          ggsave(file.path("./subgroup_analysis_results/figures", 
                           "subgroup_HGB_comprehensive_report.pdf"),
                 p_combined, width = 12, height = 10)
        }
      }
    }
  }
} else {
  cat("错误: 找不到对象'y_test'。请确保测试集标签变量存在。\n")
  cat("请检查你的变量名，可能是'true_labels'或其他名称。\n")
}