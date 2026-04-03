# 简洁学术风格的ROC曲线比较 - 3队列版本
library(ggplot2)
library(dplyr)
library(purrr)

# 定义队列文件路径和标签（3个队列）
cohort_files <- list(
  "Training cohort" = "glmnet_roc_curve_points（training set）.csv",
  "Internal validation cohort" = "glmnet_roc_curve_points（SCH）.csv",
  "External validation cohort" = "glmnet_roc_curve_points（外部验证）.csv"
)

# 数据读取函数
read_cohort_data <- function(file_path, cohort_name) {
  if (!file.exists(file_path)) {
    warning(paste("文件不存在:", file_path))
    return(NULL)
  }
  
  df <- read.csv(file_path)
  cat(sprintf("\n读取文件: %s\n", cohort_name))
  cat("原始列名:", paste(colnames(df), collapse = ", "), "\n")
  
  # 标准化列名
  colnames(df) <- tolower(colnames(df))
  
  # 识别FPR和TPR列
  fpr_col <- NULL
  tpr_col <- NULL
  
  # 可能的FPR列名
  fpr_candidates <- c("fpr", "1.specificity", "specificity", "false.positive.rate", "1-specificity")
  tpr_candidates <- c("tpr", "sensitivity", "true.positive.rate")
  
  for (candidate in fpr_candidates) {
    if (candidate %in% colnames(df)) {
      fpr_col <- candidate
      break
    }
  }
  
  for (candidate in tpr_candidates) {
    if (candidate %in% colnames(df)) {
      tpr_col <- candidate
      break
    }
  }
  
  if (is.null(fpr_col) || is.null(tpr_col)) {
    warning(paste("无法识别FPR/TPR列 in", cohort_name))
    return(NULL)
  }
  
  # 提取数据
  fpr_values <- as.numeric(df[[fpr_col]])
  tpr_values <- as.numeric(df[[tpr_col]])
  
  # 如果FPR列是Specificity，转换为1-Specificity
  if (fpr_col == "specificity") {
    fpr_values <- 1 - fpr_values
  }
  
  result <- data.frame(
    FPR = fpr_values,
    TPR = tpr_values,
    Cohort = cohort_name
  )
  
  # 移除缺失值
  result <- result[!is.na(result$FPR) & !is.na(result$TPR), ]
  
  cat(sprintf("成功读取 %d 个数据点\n", nrow(result)))
  
  return(result)
}

# 批量读取数据
cat("\n=== 开始读取数据 ===\n")
cohort_data_list <- map2(cohort_files, names(cohort_files), read_cohort_data)

# 合并所有有效数据
combined_data <- bind_rows(compact(cohort_data_list))

# 检查数据
if (is.null(combined_data) || nrow(combined_data) == 0) {
  stop("错误: 未能加载任何数据。请检查文件格式。")
}

cat(sprintf("\n总数据点: %d\n", nrow(combined_data)))
cat("\n各队列数据点数量:\n")
print(table(combined_data$Cohort))

# AUC计算函数
calculate_auc <- function(data) {
  if (nrow(data) == 0) return(NA)
  
  # 按FPR排序
  data <- data[order(data$FPR), ]
  
  # 确保包含(0,0)和(1,1)
  x <- c(0, data$FPR, 1)
  y <- c(0, data$TPR, 1)
  
  # 梯形法计算AUC
  auc_value <- sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
  
  return(round(auc_value, 3))
}

# 计算AUC
cat("\n=== 计算AUC值 ===\n")
auc_values <- combined_data %>%
  group_by(Cohort) %>%
  summarise(AUC = calculate_auc(cur_data()))

print(auc_values)

# 定义顺序
custom_order <- c(
  "Training cohort",
  "Internal validation cohort", 
  "External validation cohort"
)

# 为每个队列创建带AUC的标签
combined_data <- combined_data %>%
  left_join(auc_values, by = "Cohort") %>%
  mutate(
    Cohort_With_AUC = paste0(Cohort, " (AUC = ", AUC, ")")
  )

# 确保顺序
combined_data$Cohort_With_AUC <- factor(
  combined_data$Cohort_With_AUC,
  levels = paste0(custom_order, " (AUC = ", 
                  auc_values$AUC[match(custom_order, auc_values$Cohort)], ")")
)

# 颜色方案 - 简单直接的映射
color_scheme <- c(
  "Training cohort (AUC = 0.xxx)" = "#E41A1C",      # 红色
  "Internal validation cohort (AUC = 0.xxx)" = "#377EB8",  # 蓝色
  "External validation cohort (AUC = 0.xxx)" = "#4DAF4A"   # 绿色
)

# 更新颜色方案中的AUC值
for (cohort in custom_order) {
  auc_val <- auc_values$AUC[auc_values$Cohort == cohort]
  old_name <- paste0(cohort, " (AUC = 0.xxx)")
  new_name <- paste0(cohort, " (AUC = ", auc_val, ")")
  
  if (old_name %in% names(color_scheme)) {
    color_scheme[new_name] <- color_scheme[old_name]
    color_scheme <- color_scheme[names(color_scheme) != old_name]
  }
}

# 创建ROC图 - 使用正确的美学映射
roc_plot <- ggplot(combined_data, aes(x = FPR, y = TPR, 
                                      color = Cohort_With_AUC, 
                                      group = Cohort_With_AUC)) +
  # 对角线参考线
  geom_abline(
    slope = 1, intercept = 0, 
    linetype = "dashed", color = "grey70", linewidth = 0.5
  ) +
  # ROC曲线
  geom_line(linewidth = 1) +
  # 颜色设置 - 使用实际存在的标签
  scale_color_manual(
    name = NULL,
    values = color_scheme,
    breaks = levels(combined_data$Cohort_With_AUC),
    labels = levels(combined_data$Cohort_With_AUC)
  ) +
  # 坐标轴设置
  scale_x_continuous(
    name = "1 - Specificity",
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = c(0, 0.02)
  ) +
  scale_y_continuous(
    name = "Sensitivity",
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = c(0, 0.02)
  ) +
  # 标题
  labs( ) +
  # 主题
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(
      hjust = 0.5, 
      face = "bold",
      size = 14,
      margin = margin(b = 10)
    ),
    axis.title.x = element_text(size = 13, face = "bold"),
    axis.title.y = element_text(size = 13, face = "bold"),
    axis.text = element_text(color = "black", size = 14),
    axis.line = element_line(color = "black", linewidth = 0.5),
    axis.ticks = element_line(color = "black", linewidth = 0.5),
    legend.position = c(0.57, 0.12),
    legend.title = element_blank(),
    legend.text = element_text(size = 10.5),
    legend.background = element_rect(
      fill = "white", 
      linewidth = 0.3
    ),
    legend.key.height = unit(0.8, "lines"),
    legend.key.width = unit(1.5, "lines"),
    legend.margin = margin(1, 1, 1, 1),
    aspect.ratio = 1,
    plot.margin = margin(15, 15, 15, 15)
  ) +
  coord_fixed(ratio = 1)

# 显示图形
print(roc_plot)

# 保存图形
if (!dir.exists("output")) dir.create("output")

ggsave(
  file.path("output", "ROC_3_Cohorts.png"),
  plot = roc_plot,
  width = 4,
  height = 4,
  dpi = 300,
  bg = "white"
)


cat("\n图形已保存到 output/ 目录\n")

# 生成汇总报告
cat("\n=== AUC汇总表 ===\n")
print(auc_values)

# 计算95%特异性下的敏感性
calculate_sensitivity_at_95 <- function(data) {
  if (nrow(data) == 0) return(NA)
  
  data_sorted <- data[order(data$FPR), ]
  
  # 找到FPR最接近0.05的点（对应95%特异性）
  target_fpr <- 0.05
  idx <- which.min(abs(data_sorted$FPR - target_fpr))
  
  if (length(idx) > 0) {
    return(round(data_sorted$TPR[idx], 3))
  }
  return(NA)
}

cat("\n=== 95%特异性下的敏感性 ===\n")
for (cohort in custom_order) {
  cohort_data <- combined_data[combined_data$Cohort == cohort, ]
  sensitivity <- calculate_sensitivity_at_95(cohort_data)
  auc_val <- auc_values$AUC[auc_values$Cohort == cohort]
  
  cat(sprintf("%s: Sensitivity = %.3f, AUC = %.3f\n", 
              cohort, sensitivity, auc_val))
}

cat("\n分析完成！\n")