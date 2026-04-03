rm(list = ls())
#################################################
## glmnet模型解释代码（固定参数版本）         ##
## 功能：模型解释+特征重要性+可视化          ##
## 参数：alpha = 0.5281, lambda = 0.004597   ##
## 数据要求：CSV格式，最后一列为分组变量     ##
## 作者：基于罗怀超代码改编                   ##
## 版本：v1.0 (2024-11-06)                    ##
#################################################

library(caret)
library(glmnet)
library(pROC)
library(ggsignif)
library(corrplot)
library(rms)
library(dplyr)
library(ggplot2)
library(stringr)
library(vip)
library(DALEX)
library(ggrepel)
library(shapviz)
library(fastshap)
library(patchwork)
library(grid)
library(gridExtra)
set.seed(278)  # 可重复性

# 输出目录
FIG_DIR <- "figures_glmnet_混淆矩阵/"    # 图片输出目录
DATA_DIR <- "data_glmnet_混淆矩阵/"      # 数据输出目录

# 创建目录
if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR, recursive = TRUE)
if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR, recursive = TRUE)

### 1. 数据加载 ###
cat("Loading data...\n")

# 加载数据（请确保数据格式正确）
load(file = ".left_data.rdata")  # 请确保这个文件存在

# 检查数据结构
cat("Data Information:\n")
cat("Training samples:", nrow(train_data), "\n")
cat("Test samples:", nrow(test_data), "\n")
cat("Cancer prevalence - Training:", mean(train_data$group == "cancer"), 
    "Test:", mean(test_data$group == "cancer"), "\n")

# 准备数据
x_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$group
x_test <- as.matrix(test_data[, -ncol(test_data)])
y_test <- test_data$group

### 2. 使用固定参数训练glmnet模型 ###
cat("\nTraining glmnet model with fixed parameters...\n")
cat("alpha = 0.5281, lambda = 0.004597\n")

# 训练模型
fixed_model <- glmnet(x_train, y_train, 
                      family = "binomial",
                      alpha = 0.5281,
                      lambda = 0.004597)

cat("Model training completed.\n")
cat("Number of features:", ncol(x_train), "\n")
### 5. 模型性能评估 ###
cat("\nEvaluating model performance...\n")

# 预测概率
train_pred <- predict(fixed_model, newx = x_train, type = "response", s = 0.004597)[, 1]
test_pred <- predict(fixed_model, newx = x_test, type = "response", s = 0.004597)[, 1]
### 10. 混淆矩阵分析 ###
cat("\nPerforming cutoff analysis...\n")

# 计算ROC曲线（如果还没计算）
if (!exists("roc_obj")) {
  roc_obj <- roc(y_train, train_pred)
}

# 使用图中显示的固定cutoff值
roc_c1 <- 0.32  # 最大准确率 cutoff
roc_c2 <- 0.12 # NPV=0.900 cutoff
roc_c3 <- 0.68  # PPV=0.901 cutoff

# 应用三个截断值进行预测
test_data$pred_prob <- test_pred
test_data$pre_value_acc <- ifelse(test_data$pred_prob > roc_c1, "cancer", "control")
test_data$pre_value_npv <- ifelse(test_data$pred_prob > roc_c2, "cancer", "control")
test_data$pre_value_ppv <- ifelse(test_data$pred_prob > roc_c3, "cancer", "control")

# 设置因子水平
test_data$Truth <- test_data$group
test_data$Truth <- factor(test_data$Truth, levels = c("control", "cancer"))
test_data$pre_value_acc <- factor(test_data$pre_value_acc, levels = c("control", "cancer"))
test_data$pre_value_npv <- factor(test_data$pre_value_npv, levels = c("control", "cancer"))
test_data$pre_value_ppv <- factor(test_data$pre_value_ppv, levels = c("control", "cancer"))

# 打印三个cutoff值
cat("\nCutoff values (from threshold_performance_curve.pdf):\n")
cat("1. Max Accuracy cutoff:", roc_c1, "\n")
cat("2. NPV=0.900 cutoff:", roc_c2, "\n")
cat("3. PPV=0.901 cutoff:", roc_c3, "\n")

# 计算性能指标函数（替代confusionMatrix）
calculate_metrics <- function(predicted, actual, positive_class = "cancer") {
  # 确保是因子且水平相同
  predicted <- factor(predicted, levels = levels(actual))
  
  # 构建混淆矩阵
  cm <- table(Predicted = predicted, Actual = actual)
  
  # 提取TP, TN, FP, FN
  if (positive_class == "cancer") {
    TP <- cm["cancer", "cancer"]
    TN <- cm["control", "control"]
    FP <- cm["cancer", "control"]
    FN <- cm["control", "cancer"]
  } else {
    TP <- cm["control", "control"]
    TN <- cm["cancer", "cancer"]
    FP <- cm["control", "cancer"]
    FN <- cm["cancer", "control"]
  }
  
  # 计算指标
  accuracy <- (TP + TN) / sum(cm)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  ppv <- TP / (TP + FP)
  npv <- TN / (TN + FN)
  
  return(list(
    table = cm,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    ppv = ppv,
    npv = npv,
    TP = TP, TN = TN, FP = FP, FN = FN
  ))
}

# 计算三个混淆矩阵
c1 <- calculate_metrics(test_data$pre_value_acc, test_data$Truth, "cancer")
c2 <- calculate_metrics(test_data$pre_value_npv, test_data$Truth, "cancer")
c3 <- calculate_metrics(test_data$pre_value_ppv, test_data$Truth, "cancer")

# 性能指标汇总
cat("\n=== Performance metrics ===\n")
cat("\n1. Max Accuracy (Threshold =", roc_c1, "):\n")
cat("   Accuracy:", round(c1$accuracy, 4), "\n")
cat("   Sensitivity:", round(c1$sensitivity, 4), "\n")
cat("   Specificity:", round(c1$specificity, 4), "\n")
cat("   PPV:", round(c1$ppv, 4), "\n")
cat("   NPV:", round(c1$npv, 4), "\n")

cat("\n2. NPV=0.900 (Threshold =", roc_c2, "):\n")
cat("   Accuracy:", round(c2$accuracy, 4), "\n")
cat("   Sensitivity:", round(c2$sensitivity, 4), "\n")
cat("   Specificity:", round(c2$specificity, 4), "\n")
cat("   PPV:", round(c2$ppv, 4), "\n")
cat("   NPV:", round(c2$npv, 4), "\n")

cat("\n3. PPV=0.901 (Threshold =", roc_c3, "):\n")
cat("   Accuracy:", round(c3$accuracy, 4), "\n")
cat("   Sensitivity:", round(c3$sensitivity, 4), "\n")
cat("   Specificity:", round(c3$specificity, 4), "\n")
cat("   PPV:", round(c3$ppv, 4), "\n")
cat("   NPV:", round(c3$npv, 4), "\n")

# 绘制混淆矩阵函数
plot_confusion_matrix <- function(cm, title, subtitle = NULL, cutoff = NULL) {
  # 转换混淆矩阵为数据框
  cm_df <- as.data.frame(cm$table)
  colnames(cm_df) <- c("Prediction", "Reference", "Count")
  
  # 确保因子水平正确
  cm_df$Prediction <- factor(cm_df$Prediction, levels = c("control", "cancer"))
  cm_df$Reference <- factor(cm_df$Reference, levels = c("control", "cancer"))
  
  # 创建标题
  if (is.null(subtitle) && !is.null(cutoff)) {
    subtitle <- paste("Cutoff =", cutoff)
  }
  
  full_title <- title
  if (!is.null(subtitle)) {
    full_title <- paste(full_title, "\n", subtitle)
  }
  
  # 添加性能指标到副标题
  metrics_text <- paste(
    "Accuracy: ", round(cm$accuracy, 3))
  
  # 绘制热图
  p <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Count)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = Count), color = "black", 
              size = 8, fontface = "bold") +
    scale_fill_gradient(low = "#edf8fb", high = "darkgreen") +
    labs(
      title = full_title,
      subtitle = metrics_text,
      x = "Actual Class",
      y = "Predicted Class",
      fill = "Count"
    ) +
    theme_minimal(base_size = 22) +
    theme(
      panel.grid = element_blank(),
      legend.position = "right",
      text = element_text(size = 22),
      axis.title = element_text(size = 22, face = "bold"),
      axis.text.x = element_text(size = 22, face = "bold"),
      axis.text.y = element_text(size = 22, face = "bold", angle = 45),
      plot.title = element_text(size = 24, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 22, hjust = 0.5, lineheight = 1.2),
      legend.title = element_text(size = 22,face = "bold")
    ) +
    scale_x_discrete(labels = c("control" = "Control", "cancer" = "Cancer")) +
    scale_y_discrete(labels = c("control" = "Control", "cancer" = "Cancer"))
  
  return(p)
}

# 生成三个混淆矩阵图
p_cm_acc <- plot_confusion_matrix(c1, 
                                  "Maximum Accuracy", 
                                  cutoff = roc_c1)
p_cm_npv <- plot_confusion_matrix(c2, 
                                  "High-confidence negative", 
                                  cutoff = roc_c2)
p_cm_ppv <- plot_confusion_matrix(c3, 
                                  "High-confidence positive", 
                                  cutoff = roc_c3)


# 保存合并的混淆矩阵
cat("\nCreating combined plot...\n")
pdf(file.path(FIG_DIR, "all_confusion_matrices.pdf"), 
    width = 20, height = 6.4)
grid.arrange(p_cm_acc, p_cm_npv, p_cm_ppv, 
             ncol = 3, 
             top = textGrob(" ", 
                            gp = gpar(fontsize = 18, fontface = "bold")))
dev.off()
cat("Saved: all_confusion_matrices.pdf\n")

# 保存性能指标到CSV文件
performance_df <- data.frame(
  Method = c("Maximum Accuracy", "NPV = 0.900", "PPV = 0.901"),
  Cutoff = c(roc_c1, roc_c2, roc_c3),
  Accuracy = c(c1$accuracy, c2$accuracy, c3$accuracy),
  Sensitivity = c(c1$sensitivity, c2$sensitivity, c3$sensitivity),
  Specificity = c(c1$specificity, c2$specificity, c3$specificity),
  PPV = c(c1$ppv, c2$ppv, c3$ppv),
  NPV = c(c1$npv, c2$npv, c3$npv),
  TP = c(c1$TP, c2$TP, c3$TP),
  TN = c(c1$TN, c2$TN, c3$TN),
  FP = c(c1$FP, c2$FP, c3$FP),
  FN = c(c1$FN, c2$FN, c3$FN)
)

write.csv(performance_df, 
          file.path(FIG_DIR, "cutoff_performance_summary.csv"), 
          row.names = FALSE)
cat("Saved: cutoff_performance_summary.csv\n")

# 打印性能总结表格
cat("\n=== Performance Summary ===\n")
print(knitr::kable(performance_df, format = "simple", digits = 4))

cat("\n" , strrep("=", 60), "\n")
cat("Analysis completed successfully!\n")
cat("All results saved to:", FIG_DIR, "\n")
cat(strrep("=", 60), "\n")
### 11. 输出个体预测概率 ###
cat("\nGenerating individual prediction probabilities...\n")

# 创建预测概率数据框
prediction_df <- data.frame(
  Sample_ID = rownames(test_data),  # 假设行名为样本ID，如果没有则使用1:nrow(test_data)
  True_Label = as.character(test_data$group),
  True_Label_Numeric = ifelse(test_data$group == "cancer", 1, 0),
  Prediction_Probability = round(test_pred, 6),  # 保留6位小数
  Prediction_MaxAcc = ifelse(test_pred > roc_c1, "cancer", "control"),
  Prediction_NPVopt = ifelse(test_pred > roc_c2, "cancer", "control"),
  Prediction_PPVopt = ifelse(test_pred > roc_c3, "cancer", "control"),
  stringsAsFactors = FALSE
)

# 添加预测正确性标识
prediction_df$Correct_MaxAcc <- prediction_df$True_Label == prediction_df$Prediction_MaxAcc
prediction_df$Correct_NPVopt <- prediction_df$True_Label == prediction_df$Prediction_NPVopt
prediction_df$Correct_PPVopt <- prediction_df$True_Label == prediction_df$Prediction_PPVopt

# 添加风险等级分类（基于预测概率）
prediction_df$Risk_Level <- cut(prediction_df$Prediction_Probability, 
                                breaks = c(-Inf, roc_c2, roc_c1, roc_c3, Inf),
                                labels = c("Very Low Risk", "Low Risk", "Medium Risk", "High Risk"),
                                right = FALSE)

# 查看数据概览
cat("\nPrediction probability summary:\n")
cat("Min:", min(prediction_df$Prediction_Probability), "\n")
cat("1st Qu:", quantile(prediction_df$Prediction_Probability, 0.25), "\n")
cat("Median:", median(prediction_df$Prediction_Probability), "\n")
cat("Mean:", mean(prediction_df$Prediction_Probability), "\n")
cat("3rd Qu:", quantile(prediction_df$Prediction_Probability, 0.75), "\n")
cat("Max:", max(prediction_df$Prediction_Probability), "\n")

# 按预测概率排序
prediction_df <- prediction_df[order(prediction_df$Prediction_Probability, decreasing = TRUE), ]

# 添加排名
prediction_df$Rank <- 1:nrow(prediction_df)

# 重新排列列的顺序
prediction_df <- prediction_df[, c("Rank", "Sample_ID", "True_Label", "True_Label_Numeric",
                                   "Prediction_Probability", "Risk_Level",
                                   "Prediction_MaxAcc", "Correct_MaxAcc",
                                   "Prediction_NPVopt", "Correct_NPVopt",
                                   "Prediction_PPVopt", "Correct_PPVopt")]

# 保存为CSV文件（Excel可以打开）
write.csv(prediction_df, 
          file.path(DATA_DIR, "test_set_predictions.csv"), 
          row.names = FALSE,
          fileEncoding = "UTF-8")
cat("Saved: test_set_predictions.csv (CSV format, can be opened by Excel)\n")

# 如果需要真正的Excel格式，添加openxlsx包支持
if (!require(openxlsx, quietly = TRUE)) {
  cat("Installing openxlsx package for Excel export...\n")
  install.packages("openxlsx")
  library(openxlsx)
} else {
  library(openxlsx)
}

# 创建Excel工作簿
wb <- createWorkbook()

# 添加第一个工作表：详细预测结果
addWorksheet(wb, "Individual Predictions")
writeData(wb, "Individual Predictions", prediction_df)

# 设置列样式 - 标题行
headerStyle <- createStyle(fontSize = 12, fontColour = "#FFFFFF", 
                           fgFill = "#4F81BD", halign = "center",
                           textDecoration = "bold")
addStyle(wb, "Individual Predictions", headerStyle, rows = 1, cols = 1:ncol(prediction_df))

# 设置数字列的格式（预测概率列）
prob_col <- which(names(prediction_df) == "Prediction_Probability")
probStyle <- createStyle(numFmt = "0.000000")
addStyle(wb, "Individual Predictions", probStyle, rows = 2:(nrow(prediction_df)+1), 
         cols = prob_col)

# 分别设置每个布尔值列的样式
bool_cols <- which(names(prediction_df) %in% c("Correct_MaxAcc", "Correct_NPVopt", "Correct_PPVopt"))
boolStyle <- createStyle(fontColour = "#008000", textDecoration = "bold")

# 对每个布尔值列单独应用样式
for (col_idx in bool_cols) {
  addStyle(wb, "Individual Predictions", boolStyle, 
           rows = 2:(nrow(prediction_df)+1), 
           cols = col_idx)
}

# 设置风险等级的样式
risk_col <- which(names(prediction_df) == "Risk_Level")
riskStyle <- createStyle(fontColour = "#000080", halign = "center")
addStyle(wb, "Individual Predictions", riskStyle, 
         rows = 2:(nrow(prediction_df)+1), 
         cols = risk_col)

# 设置条件格式：根据预测正确性设置TRUE/FALSE的显示
# 为正确预测的行添加背景色（可选）
for (i in 2:(nrow(prediction_df)+1)) {
  for (col_idx in bool_cols) {
    cell_value <- prediction_df[i-1, col_idx]
    if (isTRUE(cell_value)) {
      # 为TRUE值添加背景色
      trueStyle <- createStyle(fgFill = "#E2F0D9", fontColour = "#008000", 
                               textDecoration = "bold")
      addStyle(wb, "Individual Predictions", trueStyle, rows = i, cols = col_idx)
    } else {
      # 为FALSE值添加背景色
      falseStyle <- createStyle(fgFill = "#F2DCDB", fontColour = "#FF0000")
      addStyle(wb, "Individual Predictions", falseStyle, rows = i, cols = col_idx)
    }
  }
}

# 调整列宽
setColWidths(wb, "Individual Predictions", cols = 1:ncol(prediction_df), widths = "auto")

# 冻结首行
freezePane(wb, "Individual Predictions", firstActiveRow = 2, firstActiveCol = 1)

# 添加第二个工作表：按阈值分类的统计摘要
addWorksheet(wb, "Threshold Summary")

# 创建统计摘要
summary_stats <- data.frame(
  Threshold = c("Max Accuracy", "NPV Optimized", "PPV Optimized"),
  Cutoff_Value = c(roc_c1, roc_c2, roc_c3),
  Total_Samples = nrow(test_data),
  Cancer_Predicted = c(
    sum(prediction_df$Prediction_MaxAcc == "cancer"),
    sum(prediction_df$Prediction_NPVopt == "cancer"),
    sum(prediction_df$Prediction_PPVopt == "cancer")
  ),
  Control_Predicted = c(
    sum(prediction_df$Prediction_MaxAcc == "control"),
    sum(prediction_df$Prediction_NPVopt == "control"),
    sum(prediction_df$Prediction_PPVopt == "control")
  ),
  Correct_Predictions = c(
    sum(prediction_df$Correct_MaxAcc),
    sum(prediction_df$Correct_NPVopt),
    sum(prediction_df$Correct_PPVopt)
  ),
  Accuracy = round(c(
    sum(prediction_df$Correct_MaxAcc)/nrow(test_data),
    sum(prediction_df$Correct_NPVopt)/nrow(test_data),
    sum(prediction_df$Correct_PPVopt)/nrow(test_data)
  ), 4)
)

writeData(wb, "Threshold Summary", summary_stats)
addStyle(wb, "Threshold Summary", headerStyle, rows = 1, cols = 1:ncol(summary_stats))

# 添加第三个工作表：按真实标签的分组统计
addWorksheet(wb, "Group Statistics")

group_stats <- prediction_df %>%
  group_by(True_Label) %>%
  summarise(
    Count = n(),
    Mean_Probability = round(mean(Prediction_Probability), 4),
    SD_Probability = round(sd(Prediction_Probability), 4),
    Min_Probability = round(min(Prediction_Probability), 4),
    Max_Probability = round(max(Prediction_Probability), 4),
    Q1 = round(quantile(Prediction_Probability, 0.25), 4),
    Median = round(median(Prediction_Probability), 4),
    Q3 = round(quantile(Prediction_Probability, 0.75), 4)
  )

writeData(wb, "Group Statistics", group_stats)
addStyle(wb, "Group Statistics", headerStyle, rows = 1, cols = 1:ncol(group_stats))

# 添加第四个工作表：风险等级分布
addWorksheet(wb, "Risk Level Distribution")

risk_dist <- table(prediction_df$Risk_Level, prediction_df$True_Label)
risk_dist_df <- as.data.frame.matrix(risk_dist)
risk_dist_df$Risk_Level <- rownames(risk_dist_df)
risk_dist_df <- risk_dist_df[, c("Risk_Level", "cancer", "control")]
risk_dist_df$Total <- rowSums(risk_dist_df[,c("cancer", "control")])
risk_dist_df$Cancer_Rate <- round(risk_dist_df$cancer / risk_dist_df$Total, 4)

# 按风险等级排序
risk_order <- c("Very Low Risk", "Low Risk", "Medium Risk", "High Risk")
risk_dist_df <- risk_dist_df[match(risk_order, risk_dist_df$Risk_Level), ]

writeData(wb, "Risk Level Distribution", risk_dist_df)
addStyle(wb, "Risk Level Distribution", headerStyle, rows = 1, cols = 1:ncol(risk_dist_df))

# 添加第五个工作表：高风险个体列表
addWorksheet(wb, "High Risk Individuals")
high_risk <- prediction_df[prediction_df$Risk_Level == "High Risk", 
                           c("Rank", "Sample_ID", "True_Label", "Prediction_Probability",
                             "Prediction_MaxAcc", "Prediction_NPVopt", "Prediction_PPVopt")]
if (nrow(high_risk) > 0) {
  writeData(wb, "High Risk Individuals", high_risk)
  addStyle(wb, "High Risk Individuals", headerStyle, rows = 1, cols = 1:ncol(high_risk))
} else {
  writeData(wb, "High Risk Individuals", data.frame(Message = "No high risk individuals found"))
}

# 保存Excel文件
excel_file <- file.path(DATA_DIR, "test_set_predictions_detailed.xlsx")
saveWorkbook(wb, excel_file, overwrite = TRUE)
cat("Saved detailed Excel file:", excel_file, "\n")

# 同时保存一个简单的文本格式，便于其他程序读取
write.table(prediction_df[, c("Sample_ID", "True_Label", "Prediction_Probability")],
            file.path(DATA_DIR, "test_set_predictions_simple.txt"),
            sep = "\t", row.names = FALSE, quote = FALSE)
cat("Saved simple tab-delimited file: test_set_predictions_simple.txt\n")

cat("\nIndividual prediction probabilities have been exported successfully.\n")
cat("Files saved in:", DATA_DIR, "\n")
