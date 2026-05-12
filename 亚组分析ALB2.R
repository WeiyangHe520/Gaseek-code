### 修正后的亚组分析函数（包含截断点查看和保存）###
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
  
  # 2. 查看并保存截断点（中位数）- 修改后的代码块
  cat("\n=== 亚组分组截断点信息 ===\n")
  
  # 创建数据框保存截断点信息
  cutoff_info <- data.frame(
    Feature = character(),
    Median = numeric(),
    Low_Group_Range = character(),
    High_Group_Range = character(),
    Low_Group_N = integer(),
    High_Group_N = integer(),
    stringsAsFactors = FALSE
  )
  
  for (var in clinical_features) {
    if (var %in% names(data)) {
      median_val <- median(data[[var]], na.rm = TRUE)
      min_val <- min(data[[var]], na.rm = TRUE)
      max_val <- max(data[[var]], na.rm = TRUE)
      
      # 输出到控制台
      cat(sprintf("\n特征 '%s':\n", var))
      cat(sprintf("  中位数截断点: %.3f\n", median_val))
      cat(sprintf("  数据范围: [%.3f, %.3f]\n", min_val, max_val))
      cat(sprintf("  低表达组 (%s_low): ≤ %.3f (n=%d)\n", 
                  var, median_val, sum(data[[var]] <= median_val, na.rm = TRUE)))
      cat(sprintf("  高表达组 (%s_high): > %.3f (n=%d)\n", 
                  var, median_val, sum(data[[var]] > median_val, na.rm = TRUE)))
      
      # 保存到数据框
      cutoff_info <- rbind(cutoff_info, data.frame(
        Feature = var,
        Median = round(median_val, 3),
        Min_Value = round(min_val, 3),
        Max_Value = round(max_val, 3),
        Cutoff_Point = round(median_val, 3),
        Low_Group_Definition = sprintf("≤ %.3f", median_val),
        High_Group_Definition = sprintf("> %.3f", median_val),
        Low_Group_N = sum(data[[var]] <= median_val, na.rm = TRUE),
        High_Group_N = sum(data[[var]] > median_val, na.rm = TRUE),
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # 保存截断点信息到CSV文件
  write.csv(cutoff_info, 
            file.path(DATA_DIR, "subgroup_cutoff_information.csv"), 
            row.names = FALSE)
  
  cat("\n截断点信息已保存至: ", file.path(DATA_DIR, "subgroup_cutoff_information.csv"), "\n")
  cat("========================\n")
  
  # 3. 创建亚组分组
  for (var in clinical_features) {
    if (var %in% names(data)) {
      median_val <- median(data[[var]], na.rm = TRUE)
      data[[paste0(var, "_group")]] <- factor(
        ifelse(data[[var]] > median_val, 
               paste0(var, ">54"), 
               paste0(var, "<54")),
        levels = c(paste0(var, "<54"), paste0(var, ">54"))
      )
    }
  }
  
  # 4. 创建亚组可视化（保持原样）
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
          scale_fill_manual(values = c("cancer" = "#e41a1c", "control" = "lightgray"),
                            name = "Group") +
          scale_color_manual(values = c("cancer" = "#e41a1c", "control" = "lightgray"),
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
                color = ifelse(p_value < 0.05, "black", "gray")
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
  
  # 5. 计算亚组性能指标
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
  
  # 将截断点信息也添加到结果中
  results$cutoff_information <- cutoff_info
  
  # 6. 创建亚组性能总结图
  if (nrow(metrics_df) > 1) {
    # 6.1 AUC比较图
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
    
    # 6.2 性能热图
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
  
  # 打印截断点信息表格
  cat("\n\n=== 截断点信息汇总 ===\n")
  print(cutoff_info)
  
  return(results)
}
# 运行亚组分析（需要替换为您的实际数据）
subgroup_results <- perform_subgroup_analysis_fixed(
  data = test_data_for_subgroup,  # 您的测试数据框
  predictions = test_pred,         # 模型的预测概率
  true_labels = true_labels_named, # 真实标签（0/1或因子）
  clinical_features = "ALB",       # 要分析的临床特征（可以是向量，如 c("ALB", "BMI", "gender")）
  output_dir = "./subgroup_analysis_results",  # 输出目录
  outcome_var = "group",           # 结果变量名
  outcome_labels = c("0" = "control", "1" = "cancer")  # 标签映射
)

# 查看截断点信息
print(subgroup_results$cutoff_information)

# 或者保存为CSV
write.csv(subgroup_results$cutoff_information, 
          "cutoff_values.csv", row.names = FALSE)