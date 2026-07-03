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
#第一节课
###--------------###--------------###--------------###--------------###--------------
# Global configuration
# -------------------
set.seed(278)  # For reproducibility
FIG_DIR <- "figures（T检验对比图）/"    # Output directory for figures
DATA_DIR <- "data（T检验对比图）/"      # Output directory for data
dir.create(FIG_DIR, showWarnings = FALSE)
dir.create(DATA_DIR, showWarnings = FALSE)

##数据探索
###--------------###--------------###--------------###--------------###--------------
library(DataExplorer)
# Load data
cancer_data <- read.csv("./四川省肿瘤医院合并数据（2）.csv")
#基础探索
str(cancer_data)
dim(cancer_data)
#直接看
View(cancer_data)
#data explore包
introduce(cancer_data)
plot_intro(cancer_data)
#离群值
plot_missing(cancer_data)
#分类或者文本变量
plot_bar(cancer_data)
#连续变量
plot_histogram(cancer_data)
plot_qq(cancer_data)
log_qq_data <- update_columns(cancer_data, 1:3, function(x) log(x + 1))
plot_qq(log_qq_data[, 1:3], sampled_rows = 1000L)

##多变量
#相关
plot_correlation(na.omit(cancer_data[,-ncol(cancer_data)]))
#主成分
plot_prcomp(na.omit(cancer_data[,-ncol(cancer_data)]))
#箱体图
plot_boxplot(cancer_data, by = "group")
create_report(cancer_data)

pdf("./figures/eda.pdf",10,10)
plot_intro(cancer_data)
#离群值
plot_missing(cancer_data)
plot_bar(cancer_data)
plot_histogram(cancer_data)
plot_qq(cancer_data)
log_qq_data <- update_columns(cancer_data, 1:3, function(x) log(x + 1))
plot_qq(log_qq_data[, 1:3], sampled_rows = 1000L)
plot_correlation(na.omit(cancer_data[,-ncol(cancer_data)]))
plot_prcomp(na.omit(cancer_data[,-ncol(cancer_data)]))
plot_boxplot(cancer_data, by = "group")
dev.off()

##数据准备
###--------------###--------------###--------------###--------------###--------------
# Split into training and test sets (70%/30%)
library(caret) #机器学习
library(tidyverse) #包括ggplot2,等基础R包
library(gtsummary) #表格输出
library(eoffice) #输出office文件
train_index <- createDataPartition(cancer_data$group, p = 0.8, list = FALSE)
train_data <- cancer_data[train_index, ]
test_data <- cancer_data[-train_index, ]
cancer_data_test=cancer_data
cancer_data_test$type=NA
cancer_data_test$type[train_index]='train'
cancer_data_test$type[is.na(cancer_data_test$type)]='test'

#输出
table_all <- 
  tbl_summary(
    cancer_data_test[,-ncol(cancer_data_test)],
    by = group, # split table by group
    missing = "no" # don't list missing data separately
  ) %>%
  add_n() %>% # add column with total number of non-missing observations
  add_p() %>% # test for a difference between groups
  modify_header(label = "**Variable**") %>% # update the column header
  bold_labels() 

table_list=split(cancer_data_test,f = cancer_data_test$type)
table_test <- table_list[[1]]%>% 
  tbl_summary(
    by = group, # split table by group
    missing = "no" # don't list missing data separately
  ) %>%
  add_n() %>% # add column with total number of non-missing observations
  add_p() %>% # test for a difference between groups
  modify_header(label = "**Variable**") %>% # update the column header
  bold_labels()


table_train <- table_list[[2]]%>% 
  tbl_summary(
    by = group, # split table by group
    missing = "no" # don't list missing data separately
  ) %>%
  add_n() %>% # add column with total number of non-missing observations
  add_p() %>% # test for a difference between groups
  modify_header(label = "**Variable**") %>% # update the column header
  bold_labels()


table_all %>%
  as_flex_table() %>%
  flextable::save_as_docx(path = "./data/tab1_all.docx")

table_train %>%
  as_flex_table() %>%
  flextable::save_as_docx(path = "./data/tab1_train.docx")

table_test %>%
  as_flex_table() %>%
  flextable::save_as_docx(path = "./data/tab1_test.docx")
# ==================== 修改后的柱状图生成代码 ====================

# 1. 提取有显著差异的变量（p < 0.05）
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)  # 用于组合图形

# 从gtsummary对象中提取p值信息
extract_significant_vars <- function(tbl_summary_obj, threshold = 0.05) {
  # 转换为数据框
  tbl_df <- as_tibble(tbl_summary_obj$table_body)
  
  # 提取变量名和p值
  sig_vars <- tbl_df %>%
    select(variable, p.value) %>%
    filter(!is.na(p.value)) %>%
    mutate(p.value = as.numeric(p.value)) %>%
    filter(p.value < threshold) %>%
    pull(variable)
  
  return(sig_vars)
}

# 2. 生成总体样本显著差异变量的单个柱状图
all_sig_vars <- extract_significant_vars(table_all)

if(length(all_sig_vars) > 0) {
  cat("发现", length(all_sig_vars), "个显著差异变量\n")
  
  # 创建单个变量图的函数
  # 修改create_single_var_plot函数，使用英文字符
  create_single_var_plot <- function(data, var_name, title_suffix = "Overall Sample") {
    # 计算统计量
    stats <- data %>%
      group_by(group) %>%
      summarise(
        mean = mean(.data[[var_name]], na.rm = TRUE),
        sd = sd(.data[[var_name]], na.rm = TRUE),
        n = n(),
        .groups = 'drop'
      )
    # 方法A：使用factor重新定义顺序
    stats$group <- factor(stats$group, levels = c("control", "cancer"))    
    # 获取p值
    p_value <- table_all$table_body %>%
      filter(variable == var_name) %>%
      pull(p.value) %>%
      as.numeric()
    # 格式化p值
    p_text <- ifelse(p_value < 0.0001, "p < 0.0001", 
                     sprintf("p = %.4f", p_value))
    
    # 创建图形
    p <- ggplot(stats, aes(x = group, y = mean, fill = group)) +
      # 柱状图
      geom_bar(stat = "identity", width = 0.6, alpha = 0.8) +
      # 误差棒
      geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), 
                    width = 0.2, size = 0.8, color = "black") +
      # 计算合适的nudge值
    # 均值±标准差标签
      geom_label(aes(label = sprintf("%.2f ± %.2f", mean, sd)), 
                 position = position_nudge(y = max(stats$mean + stats$sd) * 0.01),
                 size = 3.5, fontface = "bold",
                 fill = "white", alpha = 0.8,
                 label.size = 0.2, label.padding = unit(0.2, "lines")) +
      # 颜色设置
      scale_fill_manual(values = c("cancer" = "#E74C3C", "control" = "#3498DB"),
      guide = "none") +
      # 坐标轴标签
      labs(
        title = paste("Variable:", var_name),
        subtitle = p_text,
        x = "",
        y = "Mean ± Standard Deviation"
      ) +
      # 主题设置
      theme_classic(base_size = 12) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "plain", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 12, face = "italic"),
        axis.text.x = element_text(size = 11, face = "plain"),
        axis.title.y = element_text(size = 11, face = "plain"),
        plot.margin = unit(c(1, 1, 1, 1), "cm")
      ) +
      # Y轴范围
      scale_y_continuous(expand = expansion(mult = c(0, 0.01)))
    
    return(p)
  }
  
  # 为每个显著变量生成单独的PDF
  cat("为每个显著变量生成单独的柱状图...\n")
  
  for (var in all_sig_vars) {
    tryCatch({
      # 生成单个变量的图
      single_plot <- create_single_var_plot(cancer_data, var, "总体样本")
      
      # 保存为单独的文件
      filename <- paste0(FIG_DIR, "barplot_", var, ".pdf")
      ggsave(filename, single_plot, width = 6, height = 5)
      
      cat("已保存：", filename, "\n")
      
    }, error = function(e) {
      cat("生成变量", var, "的图形时出错：", e$message, "\n")
    })
  }
  
  # 同时生成组合图（所有变量在一个PDF中，每页一个变量）
  cat("生成组合PDF文件...\n")
  pdf(paste0(FIG_DIR, "all_significant_variables_single_pages.pdf"), 
      width = 6, height = 5)
  
  for (var in all_sig_vars) {
    tryCatch({
      single_plot <- create_single_var_plot(cancer_data, var, "总体样本")
      print(single_plot)
    }, error = function(e) {
      plot(0, 0, type = "n", main = paste("无法绘制变量:", var))
      text(0, 0, paste("错误:", e$message))
    })
  }
  
  dev.off()
  cat("组合PDF已保存：", paste0(FIG_DIR, "all_significant_variables_single_pages.pdf"), "\n")
  
  # 3. 生成训练集和测试集的单个变量图
  train_sig_vars <- extract_significant_vars(table_train)
  test_sig_vars <- extract_significant_vars(table_test)
  
  # 训练集单个变量图
  if(length(train_sig_vars) > 0) {
    cat("\n为训练集显著变量生成柱状图...\n")
    
    # 训练集单独PDF
    pdf(paste0(FIG_DIR, "train_significant_variables_single_pages.pdf"), 
        width = 6, height = 5)
    
    for (var in train_sig_vars) {
      tryCatch({
        single_plot <- create_single_var_plot(train_data, var, "训练集")
        print(single_plot)
      }, error = function(e) {
        plot(0, 0, type = "n", main = paste("训练集 - 无法绘制变量:", var))
      })
    }
    
    dev.off()
    cat("训练集PDF已保存\n")
  }
  
  # 测试集单个变量图
  if(length(test_sig_vars) > 0) {
    cat("\n为测试集显著变量生成柱状图...\n")
    
    # 测试集单独PDF
    pdf(paste0(FIG_DIR, "test_significant_variables_single_pages.pdf"), 
        width = 6, height = 5)
    
    for (var in test_sig_vars) {
      tryCatch({
        single_plot <- create_single_var_plot(test_data, var, "测试集")
        print(single_plot)
      }, error = function(e) {
        plot(0, 0, type = "n", main = paste("测试集 - 无法绘制变量:", var))
      })
    }
    
    dev.off()
    cat("测试集PDF已保存\n")
  }
  
  # 4. 生成汇总报告
  cat("\n=== 汇总报告 ===\n")
  cat("总体样本显著变量数：", length(all_sig_vars), "\n")
  cat("训练集显著变量数：", length(train_sig_vars), "\n")
  cat("测试集显著变量数：", length(test_sig_vars), "\n")
  
  # 5. 创建效应大小的森林图（保持不变）
  effect_sizes <- cancer_data %>%
    select(all_of(all_sig_vars), group) %>%
    pivot_longer(cols = -group, names_to = "variable", values_to = "value") %>%
    group_by(variable) %>%
    summarise(
      mean_diff = mean(value[group == levels(group)[2]], na.rm = TRUE) - 
        mean(value[group == levels(group)[1]], na.rm = TRUE),
      pooled_sd = sqrt((sd(value[group == levels(group)[1]], na.rm = TRUE)^2 + 
                          sd(value[group == levels(group)[2]], na.rm = TRUE)^2) / 2),
      cohens_d = mean_diff / pooled_sd
    ) %>%
    arrange(desc(abs(cohens_d)))
  
  # 生成森林图风格的柱状图
  effect_plot <- ggplot(effect_sizes, aes(x = reorder(variable, cohens_d), y = cohens_d)) +
    geom_bar(stat = "identity", fill = "#E74C3C", width = 0.7) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
    geom_text(aes(label = sprintf("%.2f", cohens_d)), 
              hjust = -0.2, size = 3) +
    coord_flip() +
    labs(title = "Effect size of between-group differences（Cohen's d）",
         subtitle = "A positive value indicates that the cancer group is greater than the control group, while a negative value indicates that the cancer group is less than the control group",
         x = "variable", y = "Cohen's d (effect size)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5))
  
  # 保存效应大小图
  ggsave(paste0(FIG_DIR, "effect_size_barplot.pdf"), 
         effect_plot, width = 8, height = max(6, length(all_sig_vars) * 0.3))
  
  # 6. 输出显著变量列表
  sig_vars_df <- data.frame(
    Variable = all_sig_vars,
    P_value = as.numeric(gsub("[^0-9.-]", "", 
                              table_all$table_body$p.value[match(all_sig_vars, table_all$table_body$variable)])),
    Variable_Type = sapply(all_sig_vars, function(x) {
      if(is.numeric(cancer_data[[x]])) "连续变量" else "分类变量"
    })
  )
  
  write.csv(sig_vars_df, 
            paste0(DATA_DIR, "significant_variables.csv"), 
            row.names = FALSE)
  
  cat("\n显著变量列表已保存至：", paste0(DATA_DIR, "significant_variables.csv"), "\n")
  
  # 7. 额外功能：生成变量类型特定的图形
  cat("\n根据变量类型生成特定图形...\n")
  
  # 为分类变量创建分组条形图
  categorical_vars <- sig_vars_df %>% 
    filter(Variable_Type == "分类变量") %>% 
    pull(Variable)
  
  for (var in categorical_vars) {
    if(var %in% names(cancer_data)) {
      # 计算频数
      freq_data <- cancer_data %>%
        group_by(group, !!sym(var)) %>%
        summarise(count = n(), .groups = 'drop') %>%
        group_by(group) %>%
        mutate(percentage = count / sum(count) * 100)
      
      # 创建分组条形图
      p_cat <- ggplot(freq_data, aes(x = !!sym(var), y = percentage, fill = group)) +
        geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
        geom_text(aes(label = sprintf("%.1f%%", percentage)), 
                  position = position_dodge(width = 0.8), 
                  vjust = -0.5, size = 3.5) +
        scale_fill_manual(values = c("cancer" = "#E74C3C", "control" = "#3498DB")) +
        labs(
          title = paste(var, "- Distribution of categorical variables"),
          x = var,
          y = "percentage (%)",
          fill = "group"
        ) +
        theme_minimal() +
        theme(
          plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      ggsave(paste0(FIG_DIR, "barplot_categorical_", var, ".pdf"), 
             p_cat, width = 8, height = 5)
      cat("分类变量图已保存：", var, "\n")
    }
  }
  
} else {
  cat("未发现显著差异的变量（p < 0.05）\n")
}

# 打印完成信息
cat("\n=== 图形生成完成 ===\n")
cat("1. 每个显著变量的单独PDF：", paste0(FIG_DIR, "barplot_[变量名].pdf"), "\n")
cat("2. 组合PDF（每页一个变量）：", paste0(FIG_DIR, "all_significant_variables_single_pages.pdf"), "\n")
cat("3. 训练集PDF：", paste0(FIG_DIR, "train_significant_variables_single_pages.pdf"), "\n")
cat("4. 测试集PDF：", paste0(FIG_DIR, "test_significant_variables_single_pages.pdf"), "\n")
cat("5. 效应大小图：", paste0(FIG_DIR, "effect_size_barplot.pdf"), "\n")

