rm(list = ls())
#################################################
## glmnet模型解释代码（带类别权重版本）         ##
## 功能：模型训练+评估+混淆矩阵+个体预测       ##
## 参数：alpha = 0.5281, lambda = 0.004597     ##
## 版本：v2.0 (2024-11-06) - 集成类别权重      ##
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
library(tidyr)

set.seed(3456)  # 使用权重模型的随机种子

# 输出目录
FIG_DIR <- "figures_glmnet_混淆矩阵/"    # 图片输出目录
DATA_DIR <- "data_glmnet_混淆矩阵/"      # 数据输出目录

# 创建目录
if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR, recursive = TRUE)
if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR, recursive = TRUE)

### 1. 数据加载与标准化验证 ###
cat("=== 1. 数据加载 ===\n")
load(file = ".left_data.rdata")  # 请确保这个文件存在

# 定义特征列
feature_cols <- c("gender", "age", "BMI", "Smoking_history", "drinking_history", 
                  "Family_history_of_cancer", "LYMPH_percentage", "MONO_percentage", 
                  "HGB", "MCV", "MCHC", "PLT", "DBIL", "IBIL", "ALB", "GLB", "ALT")

# 注意：假设数据已经通过 preProcess 完成了标准化
# train_pre <- preProcess(train_data, method = c("center", "scale", "YeoJohnson"))
# train_data <- predict(train_pre, train_data)
# test_pre <- preProcess(test_data, method = c("center", "scale", "YeoJohnson"))
# test_data <- predict(test_pre, test_data)

cat("\n=== 数据信息 ===\n")
cat("训练集样本数:", nrow(train_data), "\n")
cat("测试集样本数:", nrow(test_data), "\n")
cat("特征数量:", length(feature_cols), "\n")
cat("训练集癌症比例:", mean(train_data$group == "cancer"), "\n")
cat("测试集癌症比例:", mean(test_data$group == "cancer"), "\n")

# 准备矩阵和标签
x_train <- as.matrix(train_data[, feature_cols])
y_train <- ifelse(train_data$group == "cancer", 1, 0)
x_test <- as.matrix(test_data[, feature_cols])
y_test <- ifelse(test_data$group == "cancer", 1, 0)
y_train_factor <- train_data$group
y_test_factor <- test_data$group

# 验证数据已标准化
cat("\n=== 标准化验证（前3个特征）===\n")
for(i in 1:min(3, length(feature_cols))) {
  cat(sprintf("  %s: mean=%.6f, sd=%.6f\n", 
              feature_cols[i], 
              mean(x_train[, i]), 
              sd(x_train[, i])))
}

### 2. 计算类别权重 ###
cat("\n=== 2. 计算类别权重 ===\n")

cancer_count <- sum(y_train == 1)
control_count <- sum(y_train == 0)

# 逆频率权重
weight_cancer <- control_count / cancer_count
weight_control <- 1

cat(sprintf("癌症样本数: %d, 对照样本数: %d\n", cancer_count, control_count))
cat(sprintf("类别权重: 癌症=%.3f, 对照=%.3f\n", weight_cancer, weight_control))

# 创建权重向量
sample_weights <- ifelse(y_train == 1, weight_cancer, weight_control)

### 3. 训练带权重的glmnet模型 ###
cat("\n=== 3. 训练带权重的glmnet模型 ===\n")
cat("alpha = 0.5281, lambda = 0.004597\n")

# 定义参数
alpha_value <- 0.5281
lambda_value <- 0.004597

# 训练模型（standardize = FALSE 因为数据已标准化）
final_model <- glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = alpha_value,
  lambda = lambda_value,
  weights = sample_weights,
  standardize = FALSE,
  intercept = TRUE,
  thresh = 1e-7,
  maxit = 1000
)

cat("模型训练完成\n")
cat("非零系数数量（不包括截距）:", sum(coef(final_model)[-1] != 0), "\n")

### 4. 特征重要性 ###
cat("\n=== 4. 特征重要性 ===\n")

coefficients <- coef(final_model)
feature_importance <- data.frame(
  Feature = rownames(coefficients)[-1],
  Coefficient = as.vector(coefficients[-1]),
  Abs_Coefficient = abs(as.vector(coefficients[-1]))
)
feature_importance <- feature_importance[order(-feature_importance$Abs_Coefficient), ]

cat("\nTop 10 重要特征:\n")
print(head(feature_importance, 10))

# 保存特征重要性
write.csv(feature_importance, file.path(DATA_DIR, "glmnet_feature_importance.csv"), row.names = FALSE)

### 5. 模型性能评估 ###
cat("\n=== 5. 模型性能评估 ===\n")

# 预测概率
train_pred <- predict(final_model, newx = x_train, type = "response")[, 1]
test_pred <- predict(final_model, newx = x_test, type = "response")[, 1]

# 评估函数
calculate_metrics <- function(true_labels, pred_probs, dataset_name, optimize_threshold = TRUE) {
  
  true_numeric <- ifelse(true_labels == "cancer", 1, 0)
  true_factor <- factor(true_labels, levels = c("control", "cancer"))
  
  # 计算AUC和置信区间
  roc_obj <- roc(true_numeric, pred_probs, ci = TRUE)
  auc_val <- auc(roc_obj)
  auc_ci <- ci.auc(roc_obj, conf.level = 0.95)
  
  # 阈值优化（使用Youden指数）
  if (optimize_threshold) {
    coords <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"))
    best_thresh <- coords[1, "threshold"]
  } else {
    best_thresh <- 0.5
  }
  
  # 分类
  pred_class <- ifelse(pred_probs > best_thresh, "cancer", "control")
  pred_class <- factor(pred_class, levels = c("control", "cancer"))
  
  cm <- confusionMatrix(pred_class, true_factor, positive = "cancer")
  
  # Brier评分
  brier_score <- mean((pred_probs - true_numeric)^2)
  
  # 各项指标
  acc <- cm$overall["Accuracy"]
  sens <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  ppv <- cm$byClass["Pos Pred Value"]
  npv <- cm$byClass["Neg Pred Value"]
  ydi <- sens + spec - 1
  f1 <- 2 * (ppv * sens) / (ppv + sens)
  
  result <- data.frame(
    Dataset = dataset_name,
    Alpha = alpha_value,
    Lambda = lambda_value,
    NonZero_Features = sum(coef(final_model)[-1] != 0),
    Model_Type = "glmnet_weighted",
    AUC = round(auc_val, 3),
    AUC_95CI = sprintf("%.3f (%.3f-%.3f)", auc_val, auc_ci[1], auc_ci[3]),
    AUC_lower = round(auc_ci[1], 3),
    AUC_upper = round(auc_ci[3], 3),
    Brier_Score = round(brier_score, 4),
    ACC = round(acc, 3),
    SENS = round(sens, 3),
    SPEC = round(spec, 3),
    PPV = round(ppv, 3),
    NPV = round(npv, 3),
    YDI = round(ydi, 3),
    F1 = round(f1, 3),
    Best_Threshold = round(best_thresh, 4)
  )
  
  return(list(metrics = result, roc_obj = roc_obj, best_thresh = best_thresh))
}

# 计算训练集和测试集指标
train_result <- calculate_metrics(y_train_factor, train_pred, "Training")
test_result <- calculate_metrics(y_test_factor, test_pred, "Testing")

train_metrics <- train_result$metrics
test_metrics <- test_result$metrics
roc_obj <- train_result$roc_obj  # 用于后续

# 保存性能指标
metrics_df <- rbind(train_metrics, test_metrics)
write.csv(metrics_df, file.path(DATA_DIR, "glmnet_performance_metrics.csv"), row.names = FALSE)

# 打印结果
cat("\n=== 训练集性能 ===\n")
cat(sprintf("  AUC = %.3f (95%% CI: %.3f-%.3f)\n", 
            train_metrics$AUC, train_metrics$AUC_lower, train_metrics$AUC_upper))
cat(sprintf("  ACC = %.3f\n", train_metrics$ACC))
cat(sprintf("  SENS = %.3f\n", train_metrics$SENS))
cat(sprintf("  SPEC = %.3f\n", train_metrics$SPEC))
cat(sprintf("  PPV = %.3f\n", train_metrics$PPV))
cat(sprintf("  NPV = %.3f\n", train_metrics$NPV))
cat(sprintf("  F1 = %.3f\n", train_metrics$F1))
cat(sprintf("  Brier = %.4f\n", train_metrics$Brier_Score))
cat(sprintf("  最佳阈值 = %.4f\n", train_metrics$Best_Threshold))

cat("\n=== 测试集性能 ===\n")
cat(sprintf("  AUC = %.3f (95%% CI: %.3f-%.3f)\n", 
            test_metrics$AUC, test_metrics$AUC_lower, test_metrics$AUC_upper))
cat(sprintf("  ACC = %.3f\n", test_metrics$ACC))
cat(sprintf("  SENS = %.3f\n", test_metrics$SENS))
cat(sprintf("  SPEC = %.3f\n", test_metrics$SPEC))
cat(sprintf("  PPV = %.3f\n", test_metrics$PPV))
cat(sprintf("  NPV = %.3f\n", test_metrics$NPV))
cat(sprintf("  F1 = %.3f\n", test_metrics$F1))
cat(sprintf("  Brier = %.4f\n", test_metrics$Brier_Score))
cat(sprintf("  最佳阈值 = %.4f\n", test_metrics$Best_Threshold))

# 保存模型
save(final_model, file = file.path(DATA_DIR, "glmnet_weighted_model.rdata"))
cat(sprintf("模型已保存至: %s\n", file.path(DATA_DIR, "glmnet_weighted_model.rdata")))

### 6. 混淆矩阵分析（使用三个cutoff值）###
cat("\n=== 6. 混淆矩阵分析 ===\n")

# 使用图中显示的固定cutoff值（可以从阈值性能曲线图中获取）
roc_c1 <- 0.6  # 最大准确率 cutoff
roc_c2 <- 0.14  # NPV=0.900 cutoff
roc_c3 <- 0.95 # PPV=0.901 cutoff

# 应用三个截断值进行预测
test_data_clean <- test_data  # 创建副本
test_data_clean$pred_prob <- test_pred
test_data_clean$pre_value_acc <- ifelse(test_data_clean$pred_prob > roc_c1, "cancer", "control")
test_data_clean$pre_value_npv <- ifelse(test_data_clean$pred_prob > roc_c2, "cancer", "control")
test_data_clean$pre_value_ppv <- ifelse(test_data_clean$pred_prob > roc_c3, "cancer", "control")

# 设置因子水平
test_data_clean$Truth <- test_data_clean$group
test_data_clean$Truth <- factor(test_data_clean$Truth, levels = c("control", "cancer"))
test_data_clean$pre_value_acc <- factor(test_data_clean$pre_value_acc, levels = c("control", "cancer"))
test_data_clean$pre_value_npv <- factor(test_data_clean$pre_value_npv, levels = c("control", "cancer"))
test_data_clean$pre_value_ppv <- factor(test_data_clean$pre_value_ppv, levels = c("control", "cancer"))

# 打印三个cutoff值
cat("\nCutoff values:\n")
cat("1. Max Accuracy cutoff:", roc_c1, "\n")
cat("2. NPV=0.900 cutoff:", roc_c2, "\n")
cat("3. PPV=0.901 cutoff:", roc_c3, "\n")

# 计算性能指标函数
calculate_metrics_cm <- function(predicted, actual, positive_class = "cancer") {
  predicted <- factor(predicted, levels = levels(actual))
  
  cm <- table(Predicted = predicted, Actual = actual)
  
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
c1 <- calculate_metrics_cm(test_data_clean$pre_value_acc, test_data_clean$Truth, "cancer")
c2 <- calculate_metrics_cm(test_data_clean$pre_value_npv, test_data_clean$Truth, "cancer")
c3 <- calculate_metrics_cm(test_data_clean$pre_value_ppv, test_data_clean$Truth, "cancer")

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
  cm_df <- as.data.frame(cm$table)
  colnames(cm_df) <- c("Prediction", "Reference", "Count")
  
  cm_df$Prediction <- factor(cm_df$Prediction, levels = c("control", "cancer"))
  cm_df$Reference <- factor(cm_df$Reference, levels = c("control", "cancer"))
  
  if (is.null(subtitle) && !is.null(cutoff)) {
    subtitle <- paste("Cutoff =", cutoff)
  }
  
  full_title <- title
  if (!is.null(subtitle)) {
    full_title <- paste(full_title, "\n", subtitle)
  }
  
  metrics_text <- paste("Accuracy:", round(cm$accuracy, 3))
  
  p <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Count)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = Count), color = "black", size = 8, fontface = "bold") +
    scale_fill_gradient(low = "white", high = "#e41a1c") +
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
      legend.title = element_text(size = 22, face = "bold")
    ) +
    scale_x_discrete(labels = c("control" = "Control", "cancer" = "Cancer")) +
    scale_y_discrete(labels = c("control" = "Control", "cancer" = "Cancer"))
  
  return(p)
}

# 生成三个混淆矩阵图
p_cm_acc <- plot_confusion_matrix(c1, "Maximum Accuracy", cutoff = roc_c1)
p_cm_npv <- plot_confusion_matrix(c2, "High-confidence negative", cutoff = roc_c2)
p_cm_ppv <- plot_confusion_matrix(c3, "High-confidence positive", cutoff = roc_c3)

# 保存合并的混淆矩阵
cat("\nCreating combined plot...\n")
pdf(file.path(FIG_DIR, "all_confusion_matrices.pdf"), width = 20, height = 6.4)
grid.arrange(p_cm_acc, p_cm_npv, p_cm_ppv, 
             ncol = 3, 
             top = textGrob(" ", gp = gpar(fontsize = 18, fontface = "bold")))
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

write.csv(performance_df, file.path(DATA_DIR, "cutoff_performance_summary.csv"), row.names = FALSE)
cat("Saved: cutoff_performance_summary.csv\n")

### 7. 输出个体预测概率 ###
cat("\n=== 7. 输出个体预测概率 ===\n")

# 创建预测概率数据框
prediction_df <- data.frame(
  Sample_ID = rownames(test_data),
  True_Label = as.character(test_data$group),
  True_Label_Numeric = ifelse(test_data$group == "cancer", 1, 0),
  Prediction_Probability = round(test_pred, 6),
  Prediction_MaxAcc = ifelse(test_pred > roc_c1, "cancer", "control"),
  Prediction_NPVopt = ifelse(test_pred > roc_c2, "cancer", "control"),
  Prediction_PPVopt = ifelse(test_pred > roc_c3, "cancer", "control"),
  stringsAsFactors = FALSE
)

# 添加预测正确性标识
prediction_df$Correct_MaxAcc <- prediction_df$True_Label == prediction_df$Prediction_MaxAcc
prediction_df$Correct_NPVopt <- prediction_df$True_Label == prediction_df$Prediction_NPVopt
prediction_df$Correct_PPVopt <- prediction_df$True_Label == prediction_df$Prediction_PPVopt

# 添加风险等级分类
prediction_df$Risk_Level <- cut(prediction_df$Prediction_Probability, 
                                breaks = c(-Inf, roc_c2, roc_c1, roc_c3, Inf),
                                labels = c("Very Low Risk", "Low Risk", "Medium Risk", "High Risk"),
                                right = FALSE)

# 按预测概率排序
prediction_df <- prediction_df[order(prediction_df$Prediction_Probability, decreasing = TRUE), ]
prediction_df$Rank <- 1:nrow(prediction_df)

# 重新排列列的顺序
prediction_df <- prediction_df[, c("Rank", "Sample_ID", "True_Label", "True_Label_Numeric",
                                   "Prediction_Probability", "Risk_Level",
                                   "Prediction_MaxAcc", "Correct_MaxAcc",
                                   "Prediction_NPVopt", "Correct_NPVopt",
                                   "Prediction_PPVopt", "Correct_PPVopt")]

# 保存为CSV文件
write.csv(prediction_df, 
          file.path(DATA_DIR, "test_set_predictions.csv"), 
          row.names = FALSE,
          fileEncoding = "UTF-8")
cat("Saved: test_set_predictions.csv\n")

# 导出Excel文件（如果需要）
if (!require(openxlsx, quietly = TRUE)) {
  cat("Note: openxlsx package not available. Skipping Excel export.\n")
  cat("To enable Excel export, run: install.packages('openxlsx')\n")
} else {
  library(openxlsx)
  
  wb <- createWorkbook()
  addWorksheet(wb, "Individual Predictions")
  writeData(wb, "Individual Predictions", prediction_df)
  
  # 设置标题样式
  headerStyle <- createStyle(fontSize = 12, fontColour = "#FFFFFF", 
                             fgFill = "#4F81BD", halign = "center",
                             textDecoration = "bold")
  addStyle(wb, "Individual Predictions", headerStyle, rows = 1, cols = 1:ncol(prediction_df))
  
  # 设置数字列格式
  prob_col <- which(names(prediction_df) == "Prediction_Probability")
  probStyle <- createStyle(numFmt = "0.000000")
  addStyle(wb, "Individual Predictions", probStyle, rows = 2:(nrow(prediction_df)+1), cols = prob_col)
  
  setColWidths(wb, "Individual Predictions", cols = 1:ncol(prediction_df), widths = "auto")
  freezePane(wb, "Individual Predictions", firstActiveRow = 2, firstActiveCol = 1)
  
  # 添加阈值汇总工作表
  addWorksheet(wb, "Threshold Summary")
  writeData(wb, "Threshold Summary", performance_df)
  addStyle(wb, "Threshold Summary", headerStyle, rows = 1, cols = 1:ncol(performance_df))
  setColWidths(wb, "Threshold Summary", cols = 1:ncol(performance_df), widths = "auto")
  
  # 添加风险等级分布工作表
  addWorksheet(wb, "Risk Level Distribution")
  risk_dist <- table(prediction_df$Risk_Level, prediction_df$True_Label)
  risk_dist_df <- as.data.frame.matrix(risk_dist)
  risk_dist_df$Risk_Level <- rownames(risk_dist_df)
  risk_dist_df <- risk_dist_df[, c("Risk_Level", "cancer", "control")]
  risk_dist_df$Total <- rowSums(risk_dist_df[,c("cancer", "control")])
  risk_dist_df$Cancer_Rate <- round(risk_dist_df$cancer / risk_dist_df$Total, 4)
  
  risk_order <- c("Very Low Risk", "Low Risk", "Medium Risk", "High Risk")
  risk_dist_df <- risk_dist_df[match(risk_order, risk_dist_df$Risk_Level), ]
  
  writeData(wb, "Risk Level Distribution", risk_dist_df)
  addStyle(wb, "Risk Level Distribution", headerStyle, rows = 1, cols = 1:ncol(risk_dist_df))
  setColWidths(wb, "Risk Level Distribution", cols = 1:ncol(risk_dist_df), widths = "auto")
  
  excel_file <- file.path(DATA_DIR, "test_set_predictions_detailed.xlsx")
  saveWorkbook(wb, excel_file, overwrite = TRUE)
  cat("Saved detailed Excel file:", excel_file, "\n")
}

# 保存简单文本格式
write.table(prediction_df[, c("Sample_ID", "True_Label", "Prediction_Probability")],
            file.path(DATA_DIR, "test_set_predictions_simple.txt"),
            sep = "\t", row.names = FALSE, quote = FALSE)
cat("Saved simple tab-delimited file: test_set_predictions_simple.txt\n")

cat("\n", strrep("=", 60), "\n")
cat("Analysis completed successfully!\n")
cat("All results saved to:", DATA_DIR, "\n")
cat("Figures saved to:", FIG_DIR, "\n")
cat(strrep("=", 60), "\n")