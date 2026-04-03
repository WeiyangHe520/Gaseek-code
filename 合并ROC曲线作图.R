# 简洁学术风格的ROC曲线比较 - 5队列版本
library(ggplot2)
library(dplyr)
library(purrr)

# 定义队列文件路径和标签
cohort_files <- list(
  "Training cohort" = "glmnet_roc_curve_points（training set）.csv",
  "SCH internal validation cohort" = "glmnet_roc_curve_points（SCH）.csv",
  "ZHWHU external validation cohort" = "glmnet_roc_curve_points（ZHWHU）.csv",
  "ZCH external validation cohort" = "glmnet_roc_curve_points（ZCH）.csv",
  "CITWH external validation cohort" = "glmnet_roc_curve_points（CITWH）.csv"
)

# 读取所有数据并添加队列标识
read_cohort_data <- function(file_path, cohort_name) {
  if (file.exists(file_path)) {
    df <- read.csv(file_path)
    df$Cohort <- cohort_name
    return(df)
  } else {
    warning(paste("File not found:", file_path))
    return(NULL)
  }
}

# 批量读取数据
cohort_data_list <- map2(cohort_files, names(cohort_files), read_cohort_data)

# 合并所有有效数据
combined_data <- bind_rows(compact(cohort_data_list))

# 检查数据读取情况
cat("=== 数据读取情况 ===\n")
for (i in seq_along(cohort_files)) {
  cohort_name <- names(cohort_files)[i]
  file_path <- cohort_files[[i]]
  if (file.exists(file_path)) {
    n_points <- nrow(cohort_data_list[[i]])
    cat(sprintf("%-35s: %d points loaded\n", cohort_name, n_points))
  } else {
    cat(sprintf("%-35s: FILE NOT FOUND\n", cohort_name))
  }
}
cat("\n")

# 改进的AUC计算函数
calculate_auc <- function(data) {
  if (is.null(data) || nrow(data) == 0) return(NA)
  
  # 按FPR排序并确保唯一性
  data <- data %>%
    arrange(FPR) %>%
    distinct(FPR, .keep_all = TRUE)
  
  # 确保包含起点和终点
  if (nrow(data) == 0) return(NA)
  
  if (data$FPR[1] > 0) {
    data <- bind_rows(
      data.frame(FPR = 0, TPR = 0, Cohort = data$Cohort[1]),
      data
    )
  }
  
  if (tail(data$FPR, 1) < 1) {
    data <- bind_rows(
      data,
      data.frame(FPR = 1, TPR = 1, Cohort = data$Cohort[1])
    )
  }
  
  # 梯形法计算AUC
  x <- data$FPR
  y <- data$TPR
  auc_value <- sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
  
  return(round(auc_value, 3))
}

# 计算各队列AUC
auc_values <- combined_data %>%
  group_by(Cohort) %>%
  group_modify(~ data.frame(AUC = calculate_auc(.))) %>%
  ungroup()

# 显示AUC值
cat("=== 各队列AUC值 ===\n")
print(auc_values)
cat("\n")

# 定义您想要的图例顺序
custom_order <- c(
  "Training cohort",
  "SCH internal validation cohort", 
  "CITWH external validation cohort",
  "ZHWHU external validation cohort",
  "ZCH external validation cohort"
)

# 在数据中添加AUC标签
combined_data <- combined_data %>%
  left_join(auc_values, by = "Cohort") %>%
  mutate(
    Cohort_Label = paste0(Cohort, " (AUC = ", AUC, ")")
  )

# 将Cohort和Cohort_Label设为因子，保持自定义顺序
# 首先按custom_order排序auc_values
auc_values_ordered <- auc_values %>%
  mutate(Cohort = factor(Cohort, levels = custom_order)) %>%
  arrange(Cohort)

# 创建标签水平
label_levels <- paste0(custom_order, " (AUC = ", 
                       auc_values_ordered$AUC[match(custom_order, auc_values_ordered$Cohort)], 
                       ")")

# 设置因子水平
combined_data$Cohort <- factor(combined_data$Cohort, levels = custom_order)
combined_data$Cohort_Label <- factor(combined_data$Cohort_Label, levels = label_levels)

# 为5个队列定义颜色方案
color_scheme <- c(
  "Training cohort" = "#E41A1C",               # 红色
  "SCH internal validation cohort" = "#377EB8", # 蓝色
  "ZHWHU external validation cohort" = "#4DAF4A", # 绿色
  "ZCH external validation cohort" = "#984EA3",  # 紫色
  "CITWH external validation cohort" = "#FF7F00" # 橙色
)

# 创建ROC图
roc_plot <- ggplot(combined_data, aes(x = FPR, y = TPR, color = Cohort_Label, group = Cohort)) +
  # 对角线参考线
  geom_abline(
    slope = 1, intercept = 0, 
    linetype = "dashed", color = "grey70", linewidth = 0.5
  ) +
  # ROC曲线
  geom_line(linewidth = 1) +
  # 颜色设置
  scale_color_manual(
    name = NULL,
    values = setNames(
      color_scheme[levels(combined_data$Cohort)],
      label_levels
    ),
    breaks = label_levels
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
  # 标题和主题
  labs() +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(
      hjust = 0.5, 
      face = "bold",
      size = 14,
      margin = margin(b = 5)
    ),
    plot.subtitle = element_text(
      hjust = 0.5,
      color = "grey40",
      size = 10,
      margin = margin(b = 15)
    ),
    axis.title.x = element_text(size = 14, face = "bold"),  # x轴标题大小
    axis.title.y = element_text(size = 14, face = "bold"),  # y轴标题大小
    axis.text = element_text(color = "black",size = 14),
    axis.line = element_line(color = "black", linewidth = 0.5),
    axis.ticks = element_line(color = "black", linewidth = 0.5),
    legend.position = c(0.75, 0.15),
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    legend.background = element_rect(
      fill = "white", 
      color = "black", 
      linewidth = 0.3
    ),
    legend.key.height = unit(0.8, "lines"),
    legend.key.width = unit(1.2, "lines"),
    legend.margin = margin(5, 8, 5, 8),
    aspect.ratio = 1,
    plot.margin = margin(15, 15, 15, 15)
  ) +
  coord_fixed(ratio = 1)

# 显示图形
print(roc_plot)

# 保存图形
save_plot <- function(plot, filename, width = 7, height = 5) {
  # 保存PDF（矢量图）
  ggsave(
    paste0(filename, ".pdf"),
    plot = plot,
    width = width,
    height = height,
    dpi = 600,
    device = "pdf"
  )
  
  # 保存PNG（位图）
  ggsave(
    paste0(filename, ".png"),
    plot = plot,
    width = width,
    height = height,
    dpi = 300,
    bg = "white"
  )
  
  cat("Saved:", filename, ".pdf and .png\n")
}

# 保存图形
save_plot(roc_plot, "ROC_5_Cohorts")

# 创建AUC汇总表格
auc_summary_table <- auc_values %>%
  mutate(
    Cohort = factor(Cohort, levels = custom_order)
  ) %>%
  arrange(Cohort) %>%
  mutate(
    Rank = row_number(),
    Cohort_Type = case_when(
      grepl("Training", Cohort, ignore.case = TRUE) ~ "Training",
      grepl("internal", Cohort, ignore.case = TRUE) ~ "Internal Validation",
      grepl("external", Cohort, ignore.case = TRUE) ~ "External Validation",
      TRUE ~ "Other"
    )
  ) %>%
  select(Rank, Cohort, Cohort_Type, AUC)

# 打印详细汇总信息
cat("\n=== AUC 汇总表（按自定义顺序排列） ===\n")
print(as.data.frame(auc_summary_table), row.names = FALSE)

cat("\n=== 队列类型统计 ===\n")
type_summary <- auc_summary_table %>%
  group_by(Cohort_Type) %>%
  summarise(
    Count = n(),
    Mean_AUC = mean(AUC, na.rm = TRUE),
    SD_AUC = sd(AUC, na.rm = TRUE),
    Min_AUC = min(AUC, na.rm = TRUE),
    Max_AUC = max(AUC, na.rm = TRUE),
    .groups = "drop"
  )

print(as.data.frame(type_summary), row.names = FALSE)

# 数据点统计
points_summary <- combined_data %>%
  group_by(Cohort) %>%
  summarise(
    Points = n(),
    Min_FPR = round(min(FPR, na.rm = TRUE), 3),
    Max_FPR = round(max(FPR, na.rm = TRUE), 3),
    Min_TPR = round(min(TPR, na.rm = TRUE), 3),
    Max_TPR = round(max(TPR, na.rm = TRUE), 3),
    .groups = "drop"
  ) %>%
  left_join(auc_values, by = "Cohort") %>%
  arrange(Cohort)

cat("\n=== 数据点详细信息 ===\n")
print(as.data.frame(points_summary), row.names = FALSE)

# 显示图例顺序
cat("\n=== 图例显示顺序 ===\n")
for (i in 1:length(label_levels)) {
  cat(sprintf("%d. %s\n", i, label_levels[i]))
}

# 生成报告
cat("\n=== 分析报告 ===\n")
cat(sprintf("分析完成时间: %s\n", Sys.time()))
cat(sprintf("总队列数: %d\n", length(unique(combined_data$Cohort))))
cat(sprintf("总数据点: %d\n", nrow(combined_data)))
cat(sprintf("平均AUC: %.3f\n", mean(auc_values$AUC, na.rm = TRUE)))
cat(sprintf("AUC范围: %.3f - %.3f\n", min(auc_values$AUC, na.rm = TRUE), max(auc_values$AUC, na.rm = TRUE)))
cat("\n图形文件已保存为: ROC_5_Cohorts.pdf 和 ROC_5_Cohorts.png\n")