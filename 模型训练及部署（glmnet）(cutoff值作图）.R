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
FIG_DIR <- "glmnet_interpretation_figures/"
DATA_DIR <- "glmnet_interpretation_data/"
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
            title = paste("group by", var, "median"),
            y = "Prediction probability",
            x = "Group"
          ) +
          scale_fill_manual(values = c("control" = "#66c2a5", "cancer" = "#fc8d62"),
                            name = "Group") +
          scale_color_manual(values = c("control" = "#66c2a5", "cancer" = "#fc8d62"),
                             guide = "none") +
          theme_minimal(base_size = 15) +
          theme(
            legend.position = "top",
            strip.background = element_rect(fill = "lightgray", color = NA),
            strip.text = element_text(size = 15, face = "bold"),
            axis.text.x = element_text(angle = 0, hjust = 0.5)
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
                color = ifelse(p_value < 0.05, "red", "gray")
              )
          }
        }
        
        results$plots[[paste0("subgroup_", var)]] <- p_subgroup
        
        # 保存单个图
        ggsave(file.path(FIG_DIR, paste0("subgroup_analysis_", var, ".pdf")),
               p_subgroup, width = 10, height = 5)
        ggsave(file.path(FIG_DIR, paste0("subgroup_analysis_", var, ".png")),
               p_subgroup, width = 10, height = 5, dpi = 300)
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
  cat("\n=== 亚组分析（ALB分组） ===\n")
  
  # 使用修正后的函数
  subgroup_results <- perform_subgroup_analysis_fixed(
    data = test_data_for_subgroup,
    predictions = test_pred,  # 你的预测概率
    true_labels = true_labels_named,  # 使用重命名后的标签
    clinical_features = "ALB",  # 只分析ALB
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
                      "subgroup_analysis_ALB_results.rds"))
    
    # 创建综合报告
    if (length(subgroup_results$plots) > 0) {
      # 将主要图表组合
      if ("subgroup_ALB" %in% names(subgroup_results$plots)) {
        if ("auc_comparison" %in% names(subgroup_results$plots)) {
          # 使用patchwork包需要先安装
          if (!require("patchwork")) install.packages("patchwork")
          library(patchwork)
          
          p_combined <- subgroup_results$plots$subgroup_ALB / 
            subgroup_results$plots$auc_comparison +
            plot_layout(heights = c(2, 1)) +
            plot_annotation(title = "ALB亚组分析综合报告",
                            theme = theme(plot.title = element_text(size = 14, face = "bold")))
          
          ggsave(file.path("./subgroup_analysis_results/figures", 
                           "subgroup_ALB_comprehensive_report.pdf"),
                 p_combined, width = 12, height = 10)
        }
      }
    }
  }
} else {
  cat("错误: 找不到对象'y_test'。请确保测试集标签变量存在。\n")
  cat("请检查你的变量名，可能是'true_labels'或其他名称。\n")
}
### 5. SHAP解释分析 ###
cat("\n=== 5. SHAP解释分析 ===\n")

# 定义glmnet预测函数
glmnet_predict <- function(model, newdata) {
  if (!is.matrix(newdata)) {
    newdata <- as.matrix(newdata)
  }
  predictions <- predict(model, newx = newdata, 
                         type = "response", s = 0.004597)
  return(as.numeric(predictions))
}

# 选择样本进行SHAP分析（为了速度，使用子集）
set.seed(278)
n_shap_samples <- min(100, nrow(x_train))
shap_indices <- sample(nrow(x_train), n_shap_samples)
shap_x <- x_train[shap_indices, , drop = FALSE]
shap_y <- y_train[shap_indices]

# 计算SHAP值
cat("计算SHAP值中...\n")
shap_values <- fastshap::explain(
  final_model,
  X = shap_x,
  pred_wrapper = glmnet_predict,
  nsim = 50,  # 可增加以获得更精确的估计
  adjust = TRUE
)

# 创建shapviz对象
sv <- shapviz(shap_values, X = shap_x)

# 5.1 SHAP重要性图
p_shap_importance <- sv_importance(sv, kind = "bee", max_display = 20,
                                   fill = "#2ca25f", alpha = 0.7) +
  labs(title = "SHAP feature importance of glmnet",
       x = "SHAP value",
       y = "feature") +
  theme_minimal(base_size = 14)

ggsave(file.path(FIG_DIR, "glmnet_shap_importance.pdf"), 
       p_shap_importance, width = 10, height = 8)

# 5.2 创建自定义瀑布图函数
create_glmnet_waterfall <- function(sv_obj, sample_idx, 
                                    actual_label, pred_prob,
                                    n_features = 6) {
  
  # 获取SHAP值
  shap_vals <- as.numeric(sv_obj$S[sample_idx, ])
  feature_vals <- as.numeric(sv_obj$X[sample_idx, ])
  feature_names <- colnames(sv_obj$X)
  
  # 选择最重要的特征
  imp_idx <- order(abs(shap_vals), decreasing = TRUE)[1:n_features]
  
  # 创建数据框
  df <- data.frame(
    feature = feature_names[imp_idx],
    shap_value = shap_vals[imp_idx],
    feature_value = feature_vals[imp_idx]
  )
  
  # 计算累计值
  base_value <- 0  # 对于逻辑回归，基线通常为0（logit尺度）
  df <- df[order(df$shap_value, decreasing = TRUE), ]
  df$cumulative <- base_value + cumsum(df$shap_value)
  df$start <- c(base_value, head(df$cumulative, -1))
  df$end <- df$cumulative
  
  # 添加行索引
  df$y_pos <- 1:nrow(df)
  
  # 创建标签
  df$feature_label <- sprintf("%s\n= %.2f", df$feature, df$feature_value)
  df$shap_label <- sprintf("%+.3f", df$shap_value)
  
  # 绘图
  p <- ggplot(df, aes(y = y_pos)) +
    geom_segment(aes(x = start, xend = end, y = y_pos, yend = y_pos,
                     color = ifelse(shap_value > 0, "#2ca25f", "#e34a33")),
                 size = 8, alpha = 0.7) +
    geom_point(aes(x = end, y = y_pos,
                   color = ifelse(shap_value > 0, "#2ca25f", "#e34a33")),
               size = 4) +
    geom_text(aes(x = ifelse(shap_value > 0, start - 0.1, end + 0.1),
                  y = y_pos,
                  label = feature_label,
                  hjust = ifelse(shap_value > 0, 1, 0)),
              size = 3.5, fontface = "bold") +
    geom_text(aes(x = (start + end)/2, y = y_pos,
                  label = shap_label),
              color = "white", size = 3, fontface = "bold") +
    geom_vline(xintercept = base_value, linetype = "dashed", 
               color = "gray40", size = 0.8) +
    scale_color_identity() +
    labs(
      title = sprintf("SHAP瀑布图 - 样本%d", sample_idx),
      subtitle = sprintf("真实标签: %d | 预测概率: %.3f", 
                         actual_label, pred_prob),
      x = "对log-odds的贡献",
      y = ""
    ) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid.major.y = element_blank(),
      legend.position = "none"
    )
  
  return(p)
}

# 5.3 为代表性样本创建瀑布图
cat("创建代表性样本的SHAP瀑布图...\n")

# 选择代表性样本
find_representative_samples <- function(pred_probs, actual_labels, n_samples = 3) {
  # 高概率癌症样本
  cancer_idx <- which(actual_labels == 1)
  high_risk_idx <- cancer_idx[which.max(pred_probs[cancer_idx])]
  
  # 低概率癌症样本
  low_risk_idx <- cancer_idx[which.min(pred_probs[cancer_idx])]
  
  # 典型对照样本
  control_idx <- which(actual_labels == 0)
  typical_control_idx <- control_idx[which.min(abs(pred_probs[control_idx] - 0.5))]
  
  return(c(high_risk_idx, low_risk_idx, typical_control_idx))
}

# 获取代表性样本
pred_probs <- glmnet_predict(final_model, shap_x)
rep_indices <- find_representative_samples(pred_probs, shap_y)

# 创建瀑布图
waterfall_plots <- list()
for (i in 1:3) {
  idx <- which(shap_indices == rep_indices[i])
  if (length(idx) > 0) {
    waterfall_plots[[i]] <- create_glmnet_waterfall(
      sv, idx, shap_y[idx], pred_probs[idx]
    )
  }
}

# 保存瀑布图
if (length(waterfall_plots) > 0) {
  p_combined <- wrap_plots(waterfall_plots, ncol = 1) +
    plot_annotation(title = "glmnet模型SHAP瀑布图分析")
  ggsave(file.path(FIG_DIR, "glmnet_shap_waterfalls.pdf"), 
         p_combined, width = 10, height = 15)
}
### 新增代码：阈值分析与决策区域可视化 ###

# 5.1 阈值性能曲线（类似Fig.6F）
cat("\n=== 5.1 生成阈值性能曲线（类似Fig.6F） ===\n")

# 完全重写的阈值性能计算函数 - 修正版
# 完全重写的、健壮的阈值分析函数
calculate_threshold_metrics_robust <- function(y_true, y_pred) {
  # 确保输入格式正确
  cat("\n=== 输入数据检查 ===\n")
  
  # 转换y_true为数值向量（0/1）
  if (is.factor(y_true)) {
    cat("y_true是因子，转换为数值...\n")
    y_true_numeric <- as.integer(as.character(y_true))
  } else {
    y_true_numeric <- as.integer(y_true)
  }
  
  # 确保是0/1编码
  unique_vals <- unique(y_true_numeric)
  cat("y_true唯一值:", paste(sort(unique_vals), collapse=", "), "\n")
  
  # 如果是其他编码（如1/2），转换为0/1
  if (!all(c(0, 1) %in% unique_vals)) {
    cat("重新编码为0/1...\n")
    if (min(unique_vals) == 1 && max(unique_vals) == 2) {
      # 如果是1/2编码，转换为0/1
      y_true_numeric <- y_true_numeric - 1
    } else if (min(unique_vals) == 0 && max(unique_vals) == 1) {
      # 已经是0/1，无需处理
    } else {
      # 其他情况，取第一个唯一值作为0
      y_true_numeric <- ifelse(y_true_numeric == min(unique_vals), 0, 1)
    }
  }
  
  # 检查test_pred
  cat("\ntest_pred类型:", class(test_pred)[1], "\n")
  cat("test_pred范围:", range(test_pred, na.rm=TRUE), "\n")
  
  # 如果是概率，确保在[0,1]范围内
  if (max(test_pred) > 1 || min(test_pred) < 0) {
    cat("警告：预测值不在[0,1]范围内，可能不是概率值\n")
  }
  
  thresholds <- seq(0, 1, by = 0.01)
  metrics <- data.frame(
    threshold = thresholds,
    NPV = NA,
    PPV = NA,
    accuracy = NA,
    sensitivity = NA,
    specificity = NA
  )
  
  # 打印基本统计
  cat("\n=== 数据基本统计 ===\n")
  cat(sprintf("总样本数: %d\n", length(y_true_numeric)))
  cat(sprintf("阳性数(1): %d (%.1f%%)\n", 
              sum(y_true_numeric == 1), 
              100 * mean(y_true_numeric == 1)))
  cat(sprintf("阴性数(0): %d (%.1f%%)\n", 
              sum(y_true_numeric == 0), 
              100 * mean(y_true_numeric == 0)))
  
  # 先做一个简单的阈值验证
  cat("\n=== 简单阈值验证 ===\n")
  
  # 阈值0.0时
  pred_class_0 <- ifelse(test_pred > 0, 1, 0)  # 所有预测为阳性
  cat("阈值0.0:\n")
  cat(sprintf("  预测阳性数: %d\n", sum(pred_class_0 == 1)))
  cat(sprintf("  预测阴性数: %d\n", sum(pred_class_0 == 0)))
  cat(sprintf("  实际阳性数: %d\n", sum(y_true_numeric == 1)))
  cat(sprintf("  实际阴性数: %d\n", sum(y_true_numeric == 0)))
  
  # 手动计算
  TP_simple <- sum(pred_class_0 == 1 & y_true_numeric == 1)
  FP_simple <- sum(pred_class_0 == 1 & y_true_numeric == 0)
  TN_simple <- sum(pred_class_0 == 0 & y_true_numeric == 0)
  FN_simple <- sum(pred_class_0 == 0 & y_true_numeric == 1)
  
  cat(sprintf("  TP=%d, FP=%d, TN=%d, FN=%d\n", 
              TP_simple, FP_simple, TN_simple, FN_simple))
  
  # 主循环
  for (i in seq_along(thresholds)) {
    thresh <- thresholds[i]
    pred_class <- ifelse(test_pred > thresh, 1, 0)
    
    # 直接计算，避免table()的潜在问题
    TP <- sum(pred_class == 1 & y_true_numeric == 1)
    FP <- sum(pred_class == 1 & y_true_numeric == 0)
    TN <- sum(pred_class == 0 & y_true_numeric == 0)
    FN <- sum(pred_class == 0 & y_true_numeric == 1)
    
    # 验证逻辑
    if (i == 1) {  # 阈值0.00
      cat("\n=== 详细验证 - 阈值0.00 ===\n")
      cat(sprintf("预测阳性数: %d (应等于总样本数%d)\n", 
                  sum(pred_class == 1), length(y_true_numeric)))
      cat(sprintf("预测阴性数: %d (应为0)\n", sum(pred_class == 0)))
      cat(sprintf("TP=%d (应等于实际阳性数%d)\n", TP, sum(y_true_numeric == 1)))
      cat(sprintf("FP=%d (应等于实际阴性数%d)\n", FP, sum(y_true_numeric == 0)))
      cat(sprintf("FN=%d (应为0)\n", FN))
      cat(sprintf("TN=%d (应为0)\n", TN))
    }
    
    if (i == length(thresholds)) {  # 阈值1.00
      cat("\n=== 详细验证 - 阈值1.00 ===\n")
      cat(sprintf("预测阳性数: %d (应为0)\n", sum(pred_class == 1)))
      cat(sprintf("预测阴性数: %d (应等于总样本数%d)\n", 
                  sum(pred_class == 0), length(y_true_numeric)))
      cat(sprintf("TP=%d (应为0)\n", TP))
      cat(sprintf("FP=%d (应为0)\n", FP))
      cat(sprintf("FN=%d (应等于实际阳性数%d)\n", FN, sum(y_true_numeric == 1)))
      cat(sprintf("TN=%d (应等于实际阴性数%d)\n", TN, sum(y_true_numeric == 0)))
    }
    
    # 计算指标
    metrics$PPV[i] <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)
    metrics$NPV[i] <- ifelse((TN + FN) > 0, TN / (TN + FN), NA)
    metrics$accuracy[i] <- (TP + TN) / length(y_true_numeric)
    metrics$sensitivity[i] <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
    metrics$specificity[i] <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
  }
  
  # 理论验证
  cat("\n=== 理论值验证 ===\n")
  prevalence <- mean(y_true_numeric == 1)
  cat(sprintf("患病率 (阳性比例): %.3f\n", prevalence))
  cat(sprintf("阈值0.00 PPV理论值: %.3f (等于患病率)\n", prevalence))
  cat(sprintf("阈值0.00 PPV实际值: %.3f\n", metrics$PPV[1]))
  cat(sprintf("阈值1.00 NPV理论值: %.3f (等于1-患病率)\n", 1 - prevalence))
  cat(sprintf("阈值1.00 NPV实际值: %.3f\n", tail(metrics$NPV, 1)))
  
  # 正确的趋势
  cat("\n=== 期望的趋势 ===\n")
  cat("1. PPV: 从患病率(阈值0.00)逐渐增加到接近1(阈值1.00)\n")
  cat("2. NPV: 从接近0(低阈值)上升到峰值，然后下降到实际阴性比例(阈值1.00) - 先升后降\n")
  cat("3. 准确率: 呈钟形曲线，在最优阈值处最高\n")
  
  return(metrics)
}

# 运行这个版本
cat("\n")
cat(rep("=", 60), collapse = "")
cat("\n运行健壮版本的阈值分析\n")
cat(rep("=", 60), collapse = "")
cat("\n\n")

test_metrics_df_robust <- calculate_threshold_metrics_robust(y_test, test_pred)


# 现在test_metrics_df_robust是正确的
test_metrics_df <- test_metrics_df_robust

# 5.1 寻找关键阈值
cat("\n=== 5.1 寻找关键阈值 ===\n")

find_optimal_threshold <- function(metrics_df, target_ppv = 0.9, target_npv = 0.9) {
  results <- list()
  
  # 1. 最大准确率阈值
  max_acc_idx <- which.max(metrics_df$accuracy)
  results$max_acc <- list(
    threshold = metrics_df$threshold[max_acc_idx],
    value = metrics_df$accuracy[max_acc_idx],
    sensitivity = metrics_df$sensitivity[max_acc_idx],
    specificity = metrics_df$specificity[max_acc_idx]
  )
  
  # 2. 高PPV阈值（>= target_ppv）
  ppv_above_target <- metrics_df$PPV >= target_ppv
  if (any(ppv_above_target, na.rm = TRUE)) {
    high_ppv_idx <- min(which(ppv_above_target))
    results$high_ppv <- list(
      threshold = metrics_df$threshold[high_ppv_idx],
      value = metrics_df$PPV[high_ppv_idx],
      sensitivity = metrics_df$sensitivity[high_ppv_idx],
      specificity = metrics_df$specificity[high_ppv_idx]
    )
  } else {
    # 如果没有达到target_ppv，取最大值
    high_ppv_idx <- which.max(metrics_df$PPV)
    results$high_ppv <- list(
      threshold = metrics_df$threshold[high_ppv_idx],
      value = metrics_df$PPV[high_ppv_idx],
      sensitivity = metrics_df$sensitivity[high_ppv_idx],
      specificity = metrics_df$specificity[high_ppv_idx]
    )
  }
  
  # 3. 高NPV阈值（>= target_npv）
  npv_above_target <- metrics_df$NPV >= target_npv
  if (any(npv_above_target, na.rm = TRUE)) {
    high_npv_idx <- max(which(npv_above_target))
    results$high_npv <- list(
      threshold = metrics_df$threshold[high_npv_idx],
      value = metrics_df$NPV[high_npv_idx],
      sensitivity = metrics_df$sensitivity[high_npv_idx],
      specificity = metrics_df$specificity[high_npv_idx]
    )
  } else {
    # 如果没有达到target_npv，取最大值
    high_npv_idx <- which.max(metrics_df$NPV)
    results$high_npv <- list(
      threshold = metrics_df$threshold[high_npv_idx],
      value = metrics_df$NPV[high_npv_idx],
      sensitivity = metrics_df$sensitivity[high_npv_idx],
      specificity = metrics_df$specificity[high_npv_idx]
    )
  }
  
  return(results)
}

# 计算关键阈值
threshold_results <- find_optimal_threshold(test_metrics_df)

cat("\n关键阈值:\n")
cat(sprintf("  最大准确率阈值: %.3f\n", threshold_results$max_acc$threshold))
cat(sprintf("    准确率: %.3f, 敏感度: %.3f, 特异度: %.3f\n", 
            threshold_results$max_acc$value,
            threshold_results$max_acc$sensitivity,
            threshold_results$max_acc$specificity))

cat(sprintf("\n  高PPV阈值 (>=95%%): %.3f\n", threshold_results$high_ppv$threshold))
cat(sprintf("    PPV: %.3f, 敏感度: %.3f, 特异度: %.3f\n", 
            threshold_results$high_ppv$value,
            threshold_results$high_ppv$sensitivity,
            threshold_results$high_ppv$specificity))

cat(sprintf("\n  高NPV阈值 (>=95%%): %.3f\n", threshold_results$high_npv$threshold))
cat(sprintf("    NPV: %.3f, 敏感度: %.3f, 特异度: %.3f\n", 
            threshold_results$high_npv$value,
            threshold_results$high_npv$sensitivity,
            threshold_results$high_npv$specificity))

# 5.2 绘制阈值性能曲线
cat("\n=== 5.2 绘制阈值性能曲线 ===\n")

library(tidyr)
library(ggplot2)
library(ggrepel)

# 准备数据
threshold_data_long <- test_metrics_df %>%
  select(threshold, NPV, PPV, accuracy, sensitivity, specificity) %>%
  pivot_longer(cols = -threshold, names_to = "metric", values_to = "value") %>%
  mutate(
    metric = factor(metric,
                    levels = c("accuracy", "sensitivity", "specificity", "PPV", "NPV"),
                    labels = c("accuracy", "sensitivity", "specificity", "PPV", "NPV"))
  )

# 创建标注数据
annotation_data <- data.frame(
  metric = factor(c("accuracy", "PPV", "NPV"), 
                  levels = c("accuracy", "PPV", "NPV")),
  x = c(threshold_results$max_acc$threshold,
        threshold_results$high_ppv$threshold,
        threshold_results$high_npv$threshold),
  y = c(threshold_results$max_acc$value,
        threshold_results$high_ppv$value,
        threshold_results$high_npv$value),
  label = c(
    sprintf("accuracy=%.3f\nCutoff=%.3f", 
            threshold_results$max_acc$value, 
            threshold_results$max_acc$threshold),
    sprintf("PPV=%.3f\nCutoff=%.3f", 
            threshold_results$high_ppv$value, 
            threshold_results$high_ppv$threshold),
    sprintf("NPV=%.3f\nCutoff=%.3f", 
            threshold_results$high_npv$value, 
            threshold_results$high_npv$threshold)
  )
)

# 定义颜色
metric_colors <- c(
  "Accuracy" = "#4daf4a",
  "Sensitivity" = "#984ea3", 
  "Specificity" = "#ff7f00",
  "PPV" = "#e41a1c",
  "NPV" = "#377eb8"
)

# 绘制阈值性能曲线
p_threshold <- ggplot(threshold_data_long %>% 
                        filter(metric %in% c("accuracy", "PPV", "NPV")), 
                      aes(x = threshold, y = value, color = metric)) +
  geom_line(size = 1.2, alpha = 0.8) +
  
  # 添加阈值线
  geom_vline(xintercept = threshold_results$max_acc$threshold, 
             linetype = "dashed", color = "black", size = 1.2, alpha = 0.8) +
  geom_vline(xintercept = threshold_results$high_ppv$threshold, 
             linetype = "dashed", color = "#e41a1c", size = 1.2, alpha = 0.8) +
  geom_vline(xintercept = threshold_results$high_npv$threshold, 
             linetype = "dashed", color = "#377eb8", size = 1.2, alpha = 0.8) +
  
  # 添加标注点
  geom_point(data = annotation_data, aes(x = x, y = y), size = 3) +
  geom_text_repel(
    data = annotation_data,
    aes(x = x, y = y, label = label),
    box.padding = 0.5,
    point.padding = 0.3,
    size = 6,
    segment.color = "gray50",
    max.overlaps = 20,
    nudge_x = c(0, 0, 0),      # accuracy不动，PPV右移，NPV左移
    nudge_y = c(-0.28, -0.4, -0.4)    # accuracy上移，PPV上移，NPV下移
  ) +
  
  scale_color_manual(values = metric_colors) +
  
  labs(
    title = "Cutoff selection in the discovery set",
    x = "Cutoff",
    y = "Predictive score",
    color = ""
  ) +
  
  theme_minimal(base_size = 18) +
  theme(
    legend.position = "none",
    axis.text = element_text(size = 18),      # 坐标轴刻度标签
    axis.title = element_text(size = 18),     # 坐标轴标题
    axis.text.x = element_text(size = 17),    # X轴刻度（单独设置）
    axis.text.y = element_text(size = 17),    # Y轴刻度（单独设置）
    axis.title.x = element_text(size = 17, face = "bold"),   # X轴标题
    axis.title.y = element_text(size = 17, face = "bold"),   # Y轴标题
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 19,hjust = 0.5, face = "bold"),
  ) +
  
  scale_y_continuous(
    limits = c(0, 1), 
    breaks = seq(0, 1, 0.2),
    labels = scales::percent_format(accuracy = 1)
  ) +
  
  scale_x_continuous(
    limits = c(0, 1), 
    breaks = seq(0, 1, 0.1)
  ) +
  
  # 添加网格线强调
  annotate("segment", x = 0, xend = 1, y = 0.5, yend = 0.5, 
           color = "gray80", linetype = "dotted") +
  annotate("segment", x = 0.5, xend = 0.5, y = 0, yend = 1, 
           color = "gray80", linetype = "dotted")

print(p_threshold)

# 保存图片
if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR, recursive = TRUE)
ggsave(file.path(FIG_DIR, "threshold_performance_curve.tiff"), 
       p_threshold, 
       width = 7, 
       height = 4.9,
       dpi = 300,          # 设置分辨率
       device = "tiff",    # 指定设备为TIFF
       compression = "lzw") # 压缩选项（可选）
cat(sprintf("\n阈值性能曲线已保存至: %s\n", file.path(FIG_DIR, "threshold_performance_curve.Tiff")))

# 5.3 预测值分布与决策区域
cat("\n=== 5.3 预测值分布与决策区域 ===\n")

# 定义阈值
confident_npv_thresh <- threshold_results$high_npv$threshold
confident_ppv_thresh <- threshold_results$high_ppv$threshold
overall_thresh <- threshold_results$max_acc$threshold

# 创建数据框
pred_dist_df <- data.frame(
  prediction = test_pred,
  true_label = factor(y_test, levels = c(0, 1), labels = c("benign", "malignant")),
  sample_id = 1:length(test_pred)
)

# 分配决策区域
pred_dist_df <- pred_dist_df %>%
  mutate(
    region = case_when(
      prediction < confident_npv_thresh ~ "Confidence benign region",
      prediction > confident_ppv_thresh ~ "Confidence malignant region",
      TRUE ~ "uncertain region"
    ),
    region = factor(region, levels = c("Confidence benign region", "uncertain region", "Confidence malignant region"))
  )

# 计算区域统计
region_stats <- pred_dist_df %>%
  group_by(region) %>%
  summarise(
    count = n(),
    proportion = n() / nrow(pred_dist_df) * 100,
    avg_pred = mean(prediction),
    .groups = "drop"
  )

# 计算各区域的性能指标
performance_by_region <- pred_dist_df %>%
  group_by(region) %>%
  summarise(
    total = n(),
    true_benign = sum(true_label == "benign"),
    true_malignant = sum(true_label == "malignant"),
    benign_rate = mean(true_label == "benign"),
    malignant_rate = mean(true_label == "malignant"),
    .groups = "drop"
  ) %>%
  mutate(
    # 在Confidence benign region，预测为benign的样本中实际为benign的比例
    NPV = ifelse(region == "Confidence benign region", benign_rate, NA),
    # 在Confidence malignant region，预测为malignant的样本中实际为malignant的比例
    PPV = ifelse(region == "Confidence malignant region", malignant_rate, NA)
  )

cat("\n决策区域统计:\n")
print(region_stats)
cat("\n区域性能:\n")
print(performance_by_region)

# 可视化预测值分布
dens <- density(pred_dist_df$prediction)
max_density <- max(dens$y)

# 创建颜色映射
region_colors <- c("Confidence benign region" = "#377eb8", 
                   "uncertain region" = "gray70", 
                   "Confidence malignant region" = "#e41a1c")

label_colors <- c("benign" = "#377eb8", "malignant" = "#e41a1c")

p_distribution <- ggplot(pred_dist_df, aes(x = prediction, fill = true_label)) +
  # 密度图
  geom_density(alpha = 0.5, adjust = 1.2) +
  
  # 决策区域背景
  annotate("rect", 
           xmin = -Inf, xmax = confident_npv_thresh,
           ymin = 0, ymax = max_density * 1.1,
           alpha = 0.15, fill = region_colors["Confidence benign region"]) +
  
  annotate("rect", 
           xmin = confident_ppv_thresh, xmax = Inf,
           ymin = 0, ymax = max_density * 1.1,
           alpha = 0.15, fill = region_colors["Confidence malignant region"]) +
  
  annotate("rect", 
           xmin = confident_npv_thresh, xmax = confident_ppv_thresh,
           ymin = 0, ymax = max_density * 1.1,
           alpha = 0.1, fill = region_colors["uncertain region"]) +
  
  # 阈值线
  geom_vline(xintercept = confident_npv_thresh, 
             linetype = "dashed", color = region_colors["Confidence benign region"], 
             size = 1.2, alpha = 0.8) +
  
  geom_vline(xintercept = confident_ppv_thresh, 
             linetype = "dashed", color = region_colors["Confidence malignant region"], 
             size = 1.2, alpha = 0.8) +
  # 区域标签（带统计信息）
  annotate("text", 
           x = confident_npv_thresh / 2, 
           y = max_density * 1.3,
           label = sprintf("Ratio\n%.1f%%", 
                           region_stats$proportion[1],
                           performance_by_region$NPV[1] * 100),
           color = region_colors["Confidence benign region"], 
           size = 6, fontface = "bold", hjust = 0.5) +
  
  annotate("text", 
           x = (confident_npv_thresh + confident_ppv_thresh) / 2, 
           y = max_density * 1.3,
           label = sprintf("Ratio\n%.1f%%", 
                           region_stats$proportion[2]),
           color = "gray30", 
           size = 6, fontface = "bold", hjust = 0.5) +
  
  annotate("text", 
           x = (confident_ppv_thresh + 1) / 2, 
           y = max_density * 1.3,
           label = sprintf("Ratio\n%.1f%%", 
                           region_stats$proportion[3],
                           performance_by_region$PPV[3] * 100),
           color = region_colors["Confidence malignant region"], 
           size = 6, fontface = "bold", hjust = 0.5) +
  
  # 阈值标签
  annotate("text", 
           x = confident_npv_thresh, 
           y = max_density * 0.8,
           label = sprintf("NPV Cutoff\n=%.3f", confident_npv_thresh),
           color = region_colors["Confidence benign region"], 
           size = 6, hjust = -0.1, angle = 0, fontface = "bold") +
  
  annotate("text", 
           x = confident_ppv_thresh, 
           y = max_density * 0.8,
           label = sprintf("PPV Cutoff\n=%.3f", confident_ppv_thresh),
           color = region_colors["Confidence malignant region"], 
           size = 6, hjust = 1.1, angle = 0, fontface = "bold") +
  
  
  # 美学设置
  scale_fill_manual(
    name = "",
    values = label_colors,
    labels = c("control", "cancer")
  ) +
  
  labs(
    title = "Internal validation set with confident cutoff",
    x = "Prediction value",
    y = "density"
  ) +
  
  theme_minimal(base_size = 18) +
  theme(
    legend.position = c(0.8, 0.9),
    axis.text = element_text(size = 18),      # 坐标轴刻度标签
    axis.title = element_text(size = 18),     # 坐标轴标题
    axis.text.x = element_text(size = 18),    # X轴刻度（单独设置）
    axis.text.y = element_text(size = 18),    # Y轴刻度（单独设置）
    axis.title.x = element_text(size = 20,face = "bold"),   # X轴标题
    axis.title.y = element_text(size = 20,face = "bold"), 
    legend.text = element_text(size = 18),  # 图例文本字体大小
    legend.title = element_text(size = 18), # Y轴标题
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 22, hjust = 0.5, face = "bold"),
  ) +
  
  scale_x_continuous(
    limits = c(0, 1), 
    breaks = seq(0, 1, 0.2),
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  
  scale_y_continuous(
    limits = c(0, max_density * 2),
    expand = expansion(mult = c(0, 0.05))
  )

print(p_distribution)

# 保存图片
ggsave(file.path(FIG_DIR, "prediction_distribution_regions.tiff"), 
       p_distribution, width = 8, height = 5.3, dpi = 300)
cat(sprintf("预测值分布图已保存至: %s\n", file.path(FIG_DIR, "prediction_distribution_regions.pdf")))
# 5.5 绘制Overall cutoff下的预测值分布图
cat("\n=== 5.5 Overall cutoff下的预测值分布图 ===\n")

# 使用overall cutoff
overall_thresh <- 0.330  # 你计算得到的overall cutoff

# 创建数据框
pred_dist_overall_df <- data.frame(
  prediction = test_pred,
  true_label = factor(y_test, levels = c(0, 1), labels = c("benign", "malignant")),
  sample_id = 1:length(test_pred)
)

# 分配决策区域（基于overall cutoff）
pred_dist_overall_df <- pred_dist_overall_df %>%
  mutate(
    region = case_when(
      prediction < overall_thresh ~ "Negative area",
      prediction >= overall_thresh ~ "Positive area",
    ),
    region = factor(region, levels = c("Negative area", "Positive area"))
  )

# 计算区域统计
region_stats_overall <- pred_dist_overall_df %>%
  group_by(region) %>%
  summarise(
    count = n(),
    proportion = n() / nrow(pred_dist_overall_df) * 100,
    avg_pred = mean(prediction),
    .groups = "drop"
  )

# 计算各区域的性能指标
performance_by_region_overall <- pred_dist_overall_df %>%
  group_by(region) %>%
  summarise(
    total = n(),
    true_benign = sum(true_label == "benign"),
    true_malignant = sum(true_label == "malignant"),
    benign_rate = mean(true_label == "benign"),
    malignant_rate = mean(true_label == "malignant"),
    .groups = "drop"
  ) %>%
  mutate(
    NPV = ifelse(region == "Negative area", benign_rate, NA),
    PPV = ifelse(region == "Positive area", malignant_rate, NA)
  )

cat("\nOverall cutoff决策区域统计:\n")
print(region_stats_overall)
cat("\n区域性能:\n")
print(performance_by_region_overall)

# 计算整体准确率
overall_predictions <- ifelse(test_pred >= overall_thresh, 1, 0)
overall_accuracy <- mean(overall_predictions == y_test)

# 可视化预测值分布
dens_overall <- density(pred_dist_overall_df$prediction)
max_density_overall <- max(dens_overall$y)

# 创建颜色映射
region_colors_overall <- c(
  "Negative area" = "#377eb8",  # 蓝色
  "Positive area" = "#e41a1c"  # 红色
)

label_colors <- c("benign" = "#4daf4a", "malignant" = "#e41a1c")

# 绘制Overall cutoff分布图
p_distribution_overall <- ggplot(pred_dist_overall_df, aes(x = prediction, fill = true_label)) +
  # 密度图
  geom_density(alpha = 0.5, adjust = 1.2, color = NA) +
  
  # 决策区域背景
  annotate("rect", 
           xmin = -Inf, xmax = overall_thresh,
           ymin = 0, ymax = max_density_overall * 1.1,
           alpha = 0.15, fill = region_colors_overall["Negative area"]) +
  
  annotate("rect", 
           xmin = overall_thresh, xmax = Inf,
           ymin = 0, ymax = max_density_overall * 1.1,
           alpha = 0.15, fill = region_colors_overall["Positive area"]) +
  
  # Overall cutoff线
  geom_vline(xintercept = overall_thresh, 
             linetype = "dashed", color = "black", 
             size = 1.2, alpha = 0.8) +
  
  # 区域标签（带统计信息）
  annotate("text", 
           x = overall_thresh / 2, 
           y = max_density_overall * 0.95,
           label = sprintf("Ratio: %.1f%%\nNPV: %.1f%%", 
                           region_stats_overall$proportion[1],
                           performance_by_region_overall$NPV[1] * 100),
           color = region_colors_overall["Negative area"], 
           size = 5.5, fontface = "bold", hjust = 0.5) +
  
  annotate("text", 
           x = (overall_thresh + 1) / 2, 
           y = max_density_overall * 0.95,
           label = sprintf("Ratio: %.1f%%\nPPV: %.1f%%", 
                           region_stats_overall$proportion[2],
                           performance_by_region_overall$PPV[2] * 100),
           color = region_colors_overall["Positive area"], 
           size = 5.5, fontface = "bold", hjust = 0.5) +
  
  # Overall cutoff标签
  annotate("text", 
           x = overall_thresh, 
           y = max_density_overall * 0.75,
           label = sprintf("Overall cutoff\n= %.3f", overall_thresh),
           color = "black", 
           size = 5, hjust = -0.1, angle = 0, fontface = "bold") +
  
  # 美学设置
  scale_fill_manual(
    name = "",
    values = label_colors,
    labels = c("control", "cancer")
  ) +
  
  labs(
    title = "Internal validation set (early-stage) with overall cutoff",
    x = "Prediction value",
    y = "density"
  ) +
  
  theme_minimal(base_size = 18) +
  theme(
    legend.position = c(0.85, 0.85),
    axis.text = element_text(size = 18),
    axis.title = element_text(size = 18),
    axis.text.x = element_text(size = 18),
    axis.text.y = element_text(size = 18),
    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 18),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
    legend.title = element_text(size = 18, face = "bold")
  ) +
  
  scale_x_continuous(
    limits = c(0, 1), 
    breaks = seq(0, 1, 0.2),
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  
  scale_y_continuous(
    limits = c(0, max_density_overall * 2),
    expand = expansion(mult = c(0, 0.05))
  )

print(p_distribution_overall)

# 保存图片
ggsave(file.path(FIG_DIR, "prediction_distribution_overall_cutoff.tiff"), 
       p_distribution_overall, width = 10.5, height = 7.03,, dpi = 300)
cat(sprintf("Overall cutoff预测值分布图已保存至: %s\n", 
            file.path(FIG_DIR, "prediction_distribution_overall_cutoff.pdf")))

# 输出总结报告
cat("\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Overall Cutoff分析总结\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat(sprintf("\n样本统计:\n"))
cat(sprintf("   总样本数: %d\n", length(y_test)))
cat(sprintf("   恶性样本: %d (%.1f%%)\n", 
            sum(y_test == 1), 100 * mean(y_test == 1)))
cat(sprintf("   良性样本: %d (%.1f%%)\n", 
            sum(y_test == 0), 100 * mean(y_test == 0)))

cat(sprintf("\nOverall cutoff: %.3f\n", overall_thresh))
cat(sprintf("整体准确率: %.1f%%\n", overall_accuracy * 100))

cat(sprintf("\n决策区域分析:\n"))
cat(sprintf("   • Negative area (预测值 < %.3f):\n", overall_thresh))
cat(sprintf("     比例: %.1f%%\n", region_stats_overall$proportion[1]))
cat(sprintf("     NPV: %.1f%%\n", performance_by_region_overall$NPV[1] * 100))

cat(sprintf("\n   • Positive area (预测值 ≥ %.3f):\n", overall_thresh))
cat(sprintf("     比例: %.1f%%\n", region_stats_overall$proportion[2]))
cat(sprintf("     PPV: %.1f%%\n", performance_by_region_overall$PPV[2] * 100))

cat(paste(rep("=", 60), collapse = ""), "\n")
# 5.4 输出总结报告
cat("\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("阈值分析总结报告\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat(sprintf("\n1. 总体统计:\n"))
cat(sprintf("   总样本数: %d\n", length(y_test)))
cat(sprintf("   阳性样本(malignant): %d (%.1f%%)\n", 
            sum(y_test == 1), 100 * mean(y_test == 1)))
cat(sprintf("   阴性样本(benign): %d (%.1f%%)\n", 
            sum(y_test == 0), 100 * mean(y_test == 0)))

cat(sprintf("\n2. 关键阈值:\n"))
cat(sprintf("   • 最大准确率阈值: %.3f (准确率: %.1f%%)\n", 
            threshold_results$max_acc$threshold, 
            threshold_results$max_acc$value * 100))
cat(sprintf("   • 高PPV阈值 (>=95%%): %.3f (PPV: %.1f%%)\n", 
            threshold_results$high_ppv$threshold, 
            threshold_results$high_ppv$value * 100))
cat(sprintf("   • 高NPV阈值 (>=95%%): %.3f (NPV: %.1f%%)\n", 
            threshold_results$high_npv$threshold, 
            threshold_results$high_npv$value * 100))

cat(sprintf("\n3. 决策区域分析:\n"))
for (i in 1:nrow(region_stats)) {
  region_name <- as.character(region_stats$region[i])
  cat(sprintf("   • %s:\n", region_name))
  cat(sprintf("     样本比例: %.1f%% (%d个样本)\n", 
              region_stats$proportion[i], region_stats$count[i]))
  cat(sprintf("     平均预测概率: %.3f\n", region_stats$avg_pred[i]))
  
  if (region_name == "Confidence benign region") {
    cat(sprintf("     NPV: %.1f%%\n", performance_by_region$NPV[i] * 100))
  } else if (region_name == "Confidence malignant region") {
    cat(sprintf("     PPV: %.1f%%\n", performance_by_region$PPV[i] * 100))
  }
}

cat(sprintf("\n4. 临床建议:\n"))
cat(sprintf("   • Confidence benign region (预测概率 < %.3f): 可考虑保守治疗或观察\n", confident_npv_thresh))
cat(sprintf("   • uncertain region (%.3f < 预测概率 < %.3f): 建议进一步检查\n", 
            confident_npv_thresh, confident_ppv_thresh))
cat(sprintf("   • Confidence malignant region (预测概率 > %.3f): 建议积极干预或治疗\n", confident_ppv_thresh))

cat(paste(rep("=", 60), collapse = ""), "\n")