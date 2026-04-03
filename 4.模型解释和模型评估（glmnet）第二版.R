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
set.seed(278)  # 可重复性

# 输出目录
FIG_DIR <- "figures_glmnet_fixed/"    # 图片输出目录
DATA_DIR <- "data_glmnet_fixed/"      # 数据输出目录

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

### 3. 特征重要性分析 ###
cat("\nPerforming feature importance analysis...\n")

# 提取系数
coef_matrix <- coef(fixed_model, s = 0.004597)
feature_importance <- data.frame(
  feature = rownames(coef_matrix)[-1],  # 移除截距项
  coefficient = as.numeric(coef_matrix[-1, 1]),
  abs_coefficient = abs(as.numeric(coef_matrix[-1, 1])),
  stringsAsFactors = FALSE
)

# 移除零系数
feature_importance <- feature_importance[feature_importance$coefficient != 0, ]
feature_importance <- feature_importance[order(-feature_importance$abs_coefficient), ]

cat("Number of non-zero coefficients:", nrow(feature_importance), "\n")
cat("\nTop 10 features by absolute coefficient:\n")
print(head(feature_importance, 10))

# 保存系数
write.csv(feature_importance, 
          file.path(DATA_DIR, "glmnet_fixed_coefficients.csv"), 
          row.names = FALSE)

# 可视化系数重要性
p_coefficient_importance <- ggplot(head(feature_importance, 6), 
                                   aes(x = reorder(feature, coefficient), y = coefficient)) +
  geom_bar(stat = "identity", aes(fill = coefficient > 0), alpha = 0.8) +
  scale_fill_manual(values = c("TRUE" = "#2ca25f", "FALSE" = "#e34a33"), 
                    name = "Positive Effect") +
  geom_text(aes(label = sprintf("%.3f", coefficient)), 
            position = position_stack(vjust = 0.5), 
            size = 6) +
  coord_flip() +
  labs(
    title = "Feature Coefficients",
    x = "Features",
    y = "Coefficient Value"
  ) +
  theme_minimal(base_size = 22) +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 22, face = "bold"),
    plot.subtitle = element_text(size = 22),
    axis.title = element_text(size = 22, face = "bold"),
    axis.text.x = element_text(size = 23),
    axis.text.y = element_text(size = 23, angle = 45, hjust = 1)
  )

print(p_coefficient_importance)

ggsave(file.path(FIG_DIR, "glmnet_fixed_coefficient_importance.tiff"), 
  p_coefficient_importance, width = 7.5, height= 6.5,
  dpi= 300,                    # 设置分辨率（300 dpi适合大多数期刊）
  device= "tiff",              # 指定TIFF格式
  compression= "lzw",          # LZW无损压缩，减小文件大小
  bg = "white")  
### 8. 模型校准 ###
cat("\nEvaluating model performance and calibration...\n")

# 预测概率
train_pred <- predict(fixed_model, newx = x_train, type = "response", s = 0.004597)[, 1]
test_pred <- predict(fixed_model, newx = x_test, type = "response", s = 0.004597)[, 1]

# 模型性能评估
cat("\n--- Model Performance ---\n")
cat("Training AUC:", auc(roc(y_train, train_pred)), "\n")
cat("Test AUC:", auc(roc(y_test, test_pred)), "\n")

# 计算最优截断值
roc_obj_train <- roc(y_train, train_pred)
cutoff_youden <- function(roc) {
  cutoff <- roc$thresholds[which.max(roc$sensitivities + roc$specificities)]
  return(round(cutoff, 4))
}
roc_c1 <- cutoff_youden(roc_obj_train)
cat("Optimal cutoff (Youden):", roc_c1, "\n")

# 应用截断值
test_data$pred_prob <- test_pred
test_data$pre_value_youden <- ifelse(test_data$pred_prob > roc_c1, "cancer", "control")
test_data$Truth <- test_data$group
test_data$pre_value_youden <- factor(test_data$pre_value_youden, levels = c("control", "cancer"))

# 计算混淆矩阵
c1 <- confusionMatrix(test_data$pre_value_youden, test_data$Truth, positive = "cancer")

cat("Test set performance at cutoff", roc_c1, ":\n")
cat("  Sensitivity:", round(c1$byClass["Sensitivity"], 4), "\n")
cat("  Specificity:", round(c1$byClass["Specificity"], 4), "\n")
cat("  Accuracy:", round(c1$overall["Accuracy"], 4), "\n")

### 9. 校正图 ###
cat("\nGenerating calibration curve...\n")

# 创建校准数据
train_cal_df <- data.frame(
  pred_prob = train_pred,
  group = y_train,
  dataset = "Training"
)

test_cal_df <- data.frame(
  pred_prob = test_pred,
  group = y_test,
  dataset = "Test"
)

cal_df <- rbind(train_cal_df, test_cal_df)

# 校正图函数
plot_calibration <- function(data, title, bin_width = 0.1) {
  # 创建校准组
  data$cal_group <- cut(data$pred_prob, 
                        breaks = seq(0, 1, bin_width),
                        include.lowest = TRUE)
  
  cal_summary <- data %>%
    group_by(cal_group) %>%
    summarise(
      mean_pred = mean(pred_prob),
      mean_actual = mean(as.numeric(group) - 1),
      n = n(),
      ci_lower = binom.test(sum(as.numeric(group) - 1), n())$conf.int[1],
      ci_upper = binom.test(sum(as.numeric(group) - 1), n())$conf.int[2]
    ) %>%
    filter(n > 0)
  
  # 计算Brier分数
  brier_score <- mean((data$pred_prob - (as.numeric(data$group) - 1))^2)
  ece_score <- mean(abs(cal_summary$mean_pred - cal_summary$mean_actual))
  
  # 创建图
  p <- ggplot(cal_summary, aes(x = mean_pred, y = mean_actual)) +
    geom_point(aes(size = n, color = n), alpha = 0.8) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                  width = 0.02, alpha = 0.6) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40", size = 1) +
    geom_smooth(method = "loess", se = TRUE, color = "#e34a33", alpha = 0.3, size = 1.2) +
    scale_color_gradient(low = "#b2e2e2", high = "#006d2c", name = "Sample size") +
    scale_size_continuous(range = c(3, 10), name = "Sample size") +
    labs(
      title = title,
      subtitle = sprintf("Brier score: %.4f | ECE: %.4f", brier_score, ece_score),
      x = "Predicted Probability",
      y = "Actual Proportion"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 16, hjust = 0.5, color = "black"),
      axis.title = element_text(size = 17, face = "bold",),
      axis.text = element_text(size = 17),
      legend.position = "right",
      panel.grid.major = element_line(color = "gray90", size = 0.2),
      panel.grid.minor = element_line(color = "gray95", size = 0.1),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    annotate("text", x = 0.8, y = 0.2, 
             label = sprintf("N = %d", nrow(data)),
             size = 6, color = "black")
  
  return(p)
}

# 创建校正图
p_cal_train <- plot_calibration(train_cal_df, 
                                "Training Set Calibration",
                                bin_width = 0.1)

p_cal_test <- plot_calibration(test_cal_df,
                               "External Validation Set Calibration",
                               bin_width = 0.1)

# 合并校正图
library(patchwork)
p_calibration_combined <- p_cal_train + p_cal_test +
  plot_annotation(
    theme = theme(
      plot.title = element_text(size = 20, face = "bold", hjust = 0.5)
    )
  )

# 保存校正图
print(p_calibration_combined)

ggsave(file.path(FIG_DIR, "glmnet_fixed_calibration.png"), 
       p_calibration_combined, width = 10, height = 5, dpi = 300)
ggsave(file.path(FIG_DIR, "glmnet_fixed_calibration.pdf"), 
       p_calibration_combined, width = 10, height = 5, dpi = 300)

### 4. 置换重要性分析 ###
cat("\nCalculating permutation importance...\n")

calculate_permutation_importance_fixed <- function(model, x_data, y_data, n_permutations = 10) {
  # 基线预测
  baseline_pred <- predict(model, newx = x_data, type = "response", s = 0.004597)[, 1]
  baseline_auc <- auc(roc(y_data, baseline_pred))
  
  importance_df <- data.frame(
    feature = colnames(x_data),
    importance = 0,
    stringsAsFactors = FALSE
  )
  
  # 计算每个特征的重要性
  for (feature in colnames(x_data)) {
    auc_drops <- numeric(n_permutations)
    
    for (i in 1:n_permutations) {
      perm_data <- x_data
      perm_data[, feature] <- sample(perm_data[, feature])
      perm_pred <- predict(model, newx = perm_data, type = "response", s = 0.004597)[, 1]
      perm_auc <- auc(roc(y_data, perm_pred))
      auc_drops[i] <- baseline_auc - perm_auc
    }
    
    importance_df$importance[importance_df$feature == feature] <- mean(auc_drops)
  }
  
  importance_df <- importance_df[order(-importance_df$importance), ]
  return(importance_df)
}

# 计算置换重要性
perm_importance <- calculate_permutation_importance_fixed(fixed_model, x_train, y_train)

# 可视化置换重要性
p_perm_importance <- ggplot(head(perm_importance, 10), 
                            aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "#2ca25f", alpha = 0.8) +
  geom_text(aes(label = sprintf("%.4f", importance)), 
            hjust = -0.1, size = 6, color = "darkgreen") +
  coord_flip() +
  labs(
    title = "glmnet Fixed Parameters - Permutation Importance",
    subtitle = "Decrease in AUC after feature permutation",
    x = "Feature",
    y = "AUC Decrease"
  ) +
  theme_minimal(base_size = 22) +
  theme(
    plot.title = element_text(size = 26, face = "bold"),
    plot.subtitle = element_text(size = 22),
    axis.title.x = element_text(size = 24),
    axis.title.y = element_text(size = 24),
    axis.text.x = element_text(size = 21),
    axis.text.y = element_text(size = 20)
  )

print(p_perm_importance)
ggsave(file.path(FIG_DIR, "glmnet_fixed_permutation_importance.pdf"), 
       p_perm_importance, width = 21, height = 15)

# 保存置换重要性结果
write.csv(perm_importance, 
          file.path(DATA_DIR, "glmnet_fixed_permutation_importance.csv"), 
          row.names = FALSE)

### 5. 模型性能评估 ###
cat("\nEvaluating model performance...\n")

# 预测概率
train_pred <- predict(fixed_model, newx = x_train, type = "response", s = 0.004597)[, 1]
test_pred <- predict(fixed_model, newx = x_test, type = "response", s = 0.004597)[, 1]


### 6. SHAP分析 ###
cat("\nPerforming SHAP analysis...\n")

# 为glmnet模型创建预测函数
glmnet_predict_fixed <- function(model, newdata) {
  if (!is.matrix(newdata)) {
    newdata <- as.matrix(newdata)
  }
  predictions <- predict(model, newx = newdata, type = "response", s = 0.004597)
  return(as.numeric(predictions))
}

# 选择代表性样本进行SHAP分析
set.seed(278)
n_explain <- min(100, nrow(x_train))
explain_indices <- sample(nrow(x_train), n_explain)
explain_data <- x_train[explain_indices, , drop = FALSE]
colnames(explain_data) <- colnames(x_train)

# 计算SHAP值
cat("Calculating SHAP values...\n")
shap_values <- fastshap::explain(
  fixed_model,
  X = explain_data,
  pred_wrapper = glmnet_predict_fixed,
  nsim = 20,
  adjust = TRUE
)

# 创建shapviz对象
sv <- shapviz(shap_values, X = explain_data)
actual_labels <- as.character(y_train[explain_indices])
pred_probs <- glmnet_predict_fixed(fixed_model, explain_data)

# 1. 蜂群图
cat("Generating SHAP beeswarm plot...\n")
tiff(file.path(FIG_DIR, "glmnet_fixed_SHAP_beeswarm.tiff"), 
     width = 7.5, 
     height = 6.5, 
     units = "in",  # 单位设为英寸
     res = 300,     # 分辨率300 dpi
     compression = "lzw")  # LZW压缩

p_beeswarm <- sv_importance(sv, kind = "bee", max_display = 6, 
                            fill = "#2ca25f", alpha = 0.7,
                            bee_width = 0.25) + 
  ggtitle("SHAP Importance") +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(size = 22, face = "bold"),
    axis.title = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 25),
    legend.title = element_text(size = 21),
    legend.text = element_text(size = 20)
  ) +
  labs(x = "SHAP value", y = "Features")

print(p_beeswarm)
dev.off()

# 2. 条形重要性图
cat("Generating SHAP bar importance plot...\n")
pdf(file.path(FIG_DIR, "glmnet_fixed_SHAP_bar_importance.pdf"), width = 21, height = 15)

p_bar <- sv_importance(sv, kind = "bar", max_display = 6, 
                       fill = "#2ca25f", color = "darkgreen") + 
  ggtitle("glmnet Fixed Parameters - Mean |SHAP| Value (Top 30 Features)") +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(size = 26, face = "bold"),
    axis.title = element_text(size = 24),
    axis.text = element_text(size = 21),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 18)
  ) +
  labs(x = "Mean |SHAP| value", y = "Feature")

print(p_bar)
dev.off()

### 7. 选择代表性样本进行详细分析 ###
cat("\nSelecting representative samples for detailed analysis...\n")

cancer_indices_all <- explain_indices[actual_labels == "cancer"]
control_indices_all <- explain_indices[actual_labels == "control"]

if (length(cancer_indices_all) >= 2 && length(control_indices_all) >= 1) {
  cancer_probs <- pred_probs[actual_labels == "cancer"]
  control_probs <- pred_probs[actual_labels == "control"]
  
  # 选择样本
  high_risk_idx <- which.max(cancer_probs)
  low_risk_idx <- which.min(cancer_probs)
  control_idx <- which.min(abs(control_probs - 0.5))
  
  high_risk_sample <- cancer_indices_all[high_risk_idx]
  low_risk_sample <- cancer_indices_all[low_risk_idx]
  typical_control <- control_indices_all[control_idx]
  
  sample_indices <- c(high_risk_sample, low_risk_sample, typical_control)
  sample_names <- c("High Risk Cancer", "Low Risk Cancer", "Typical Control")
  
  # 创建详细的SHAP分析PDF
  cat("Generating detailed SHAP analysis PDF...\n")
  pdf(file.path(FIG_DIR, "glmnet_fixed_SHAP_detailed_analysis.pdf"), width = 21, height = 20)
  
  for (i in 1:3) {
    idx <- which(explain_indices == sample_indices[i])
    actual_label <- ifelse(is.na(actual_labels[idx]), "Unknown", actual_labels[idx])
    
    # 瀑布图
    cat(sprintf("  Generating waterfall plot for %s sample...\n", sample_names[i]))
    p_waterfall <- sv_waterfall(
      sv, 
      row_id = idx, 
      max_display = 6, 
      fill_phi = ifelse(pred_probs[idx] > 0.5, "#e34a33", "#2ca25f"),
      size = 15,
      size_waterfall = 15,
      size_annotation = 12,
      show_annotation = TRUE,
      annotation_size = 10,
      digits = 3,
      format_fun = function(x) sprintf("%.3f", x)
    ) +  
      ggtitle(sprintf("Waterfall Plot - %s\nActual: %s | Predicted: %.3f", 
                      sample_names[i], actual_label, pred_probs[idx])) +
      theme_classic(base_size = 14) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 26),
        axis.title = element_text(size = 22),
        axis.text = element_text(size = 21),
        legend.title = element_text(size = 22),
        legend.text = element_text(size = 22)
      )
    
    print(p_waterfall)
    
    # 力图
    cat(sprintf("  Generating force plot for %s sample...\n", sample_names[i]))
    p_force <- sv_force(
      sv, 
      row_id = idx, 
      max_display = 6, 
      fill_positive = "#2ca25f", 
      fill_negative = "#e34a33",
      size = 0,
      size_force = 0,
      size_annotation = 0, 
      show_annotation = TRUE,
      annotation_size = 10
    ) +
      ggtitle(sprintf("Force Plot - %s\nActual: %s | Predicted: %.3f", 
                      sample_names[i], actual_label, pred_probs[idx])) +
      theme_classic(base_size = 14) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 26),
        axis.title = element_text(size = 22),
        axis.text = element_text(size = 21),
        legend.title = element_text(size = 22),
        legend.text = element_text(size = 22)
      )
    
    print(p_force)
  }
  
  dev.off()
  
  # 保存SHAP数据
  cat("Saving SHAP data...\n")
  shap_data <- list(
    shap_values = shap_values,
    explain_data = explain_data,
    actual_labels = actual_labels,
    pred_probs = pred_probs,
    feature_names = colnames(x_train),
    sample_analysis = data.frame(
      sample_id = sample_indices,
      sample_type = sample_names,
      actual_label = actual_labels[match(sample_indices, explain_indices)],
      predicted_prob = pred_probs[match(sample_indices, explain_indices)]
    )
  )
  
  saveRDS(shap_data, file.path(DATA_DIR, "glmnet_fixed_SHAP_data.rds"))
  
  # 打印样本信息
  cat("\n=== SHAP Analysis - Sample Information ===\n")
  cat(sprintf("Total samples analyzed: %d\n", n_explain))
  cat(sprintf("Cancer samples: %d, Control samples: %d\n", 
              sum(actual_labels == "cancer"), sum(actual_labels == "control")))
  
} else {
  cat("Warning: Insufficient samples for detailed SHAP analysis.\n")
}
# 4. 生成单独的依赖图PDF
cat("Generating SHAP dependence plots...\n")

# 获取最重要的特征
importance_data <- as.data.frame(sv_importance(sv, kind = "bar")$data)
if (nrow(importance_data) > 0) {
  top_features <- as.character(importance_data$feature[1:min(6, nrow(importance_data))])
  
  if (length(top_features) > 0) {
    pdf(file.path(FIG_DIR, "glmnetAIC_SHAP_dependence.pdf"), width = 10, height = 8)
    
    # 创建多个依赖图
    for (feature in top_features) {
      if (feature %in% colnames(explain_data)) {
        p_dependence <- sv_dependence(sv, v = feature, 
                                      color_var = NULL,
                                      alpha = 0.6,
                                      size = 1.5) +
          geom_smooth(method = "loess", se = FALSE, color = "#e34a33", size = 1.2) +
          ggtitle(sprintf("SHAP Dependence - %s", feature)) +
          theme_classic(base_size = 14) +
          theme(
            plot.title = element_text(hjust = 0.5, face = "bold", size = 26),      # 标题字体大小
            axis.title = element_text(size = 24),      # 坐标轴标题字体大小
            axis.text = element_text(size = 21),       # 坐标轴刻度标签字体大小
            panel.grid.major = element_line(color = "gray90", size = 0.2)
          ) +
          labs(x = feature, y = "SHAP value")
        
        print(p_dependence)
      }
    }
    
    dev.off()
  }
}
### 6. Confounding Adjustment Analysis ###
cat("Performing confounding adjustment...\n")

# Prepare test data for confounding analysis
test_data_numeric <- test_data
test_data_numeric$group <- as.numeric(test_data_numeric$group) - 1
test_data_numeric$pred_prob <- test_pred

# Select top features for adjustment
top_features_confound <- head(perm_importance$feature, 6)

if(length(top_features_confound) > 0) {
  formula_str <- paste("group ~ pred_prob +", paste(top_features_confound, collapse = " + "))
  
  model <- glm(as.formula(formula_str), data = test_data_numeric, 
               na.action = "na.exclude", family = "binomial") 
  
  tem <- summary(model)$coefficients
  tem <- as.data.frame(tem)
  tem$`-logP` <- -log10(tem$`Pr(>|z|)`)
  tem <- tem[-1, ]  # Remove intercept
  tem$id <- rownames(tem)
  tem$weight <- tem$Estimate
  
  p_confound <- ggplot(tem, aes(x = reorder(id, `-logP`), y = `-logP`, fill = weight)) +
    geom_bar(stat = "identity") +
    scale_fill_gradient2(low = "#e34a33", mid = "#f7f7f7", high = "#2ca25f",
                         midpoint = 0, name = "weight") +
    coord_flip() + 
    theme_minimal() + 
    labs(title = 'Logit regression adjustment', 
         x = 'Features', y = '-log10 P_value') +
    geom_hline(yintercept = 1.30103, linetype = "dashed", color = "red", size = 1.0) +
    geom_hline(yintercept = 1.30103, linetype = "dashed", color = "red", size = 1.0) +
    theme(
      text = element_text(size = 14),  # 全局字体大小（默认12）
      plot.title = element_text(size = 22, face = "bold"),      # 主标题大小
      plot.subtitle = element_text(size = 22),                 # 副标题大小
      axis.title = element_text(size = 20, face = "bold"),                    # 坐标轴标题大小
      axis.text.x = element_text(size = 21),  # 修改这里
      axis.text.y = element_text(size = 21, angle = 45, hjust = 1),
      legend.title = element_text(size = 20),  # 图例标题
      legend.text = element_text(size = 18), # 图例文本
    ) + 
    annotate(geom = 'text', y = 1.35, x = 1, size = 8.0, label = 'p=0.05')
  
  print(p_confound)
  ggsave(file.path(FIG_DIR, "glmnetAIC_confounding_adjustment.tiff"), 
         p_confound, width = 7.5, height = 6.5, dpi = 300)
}
### 8. 相关性分析 ###
cat("\nAnalyzing feature correlations...\n")

# 选择重要特征进行相关性分析
top_features <- head(perm_importance$feature, 6)

if(length(top_features) > 1) {
  # 创建相关性矩阵
  cor_data <- train_data[, top_features]
  cor_matrix <- cor(cor_data, method = "spearman", use = "complete.obs")
  
  pdf(file.path(FIG_DIR, "glmnet_fixed_feature_correlations.pdf"), width = 10, height = 8)
  corrplot(cor_matrix, 
           type = "lower",
           tl.col = "black",
           tl.srt = 30,
           tl.cex = 1.4,
           method = "square",
           order = "AOE",
           addCoef.col = "black",
           number.cex = 1.4,
           col = COL2("BrBG"),
           diag = FALSE,
           mar = c(0, 0, 0, 0),
           cl.cex = 1.4,                # 颜色图例字体大小（新增）  
           cl.ratio = 0.1,              # 图例宽度比例
           cl.align.text = "c"          # 图例文本对齐方式
  )
  title(main = "",
        cex.main = 1.5,
        font.main = 2,
        line = -1)
  dev.off()
}

### 8. 模型校准 ###
cat("\nEvaluating model performance and calibration...\n")

# 预测概率
train_pred <- predict(fixed_model, newx = x_train, type = "response", s = 0.004597)[, 1]
test_pred <- predict(fixed_model, newx = x_test, type = "response", s = 0.004597)[, 1]

# 模型性能评估
cat("\n--- Model Performance ---\n")
cat("Training AUC:", auc(roc(y_train, train_pred)), "\n")
cat("Test AUC:", auc(roc(y_test, test_pred)), "\n")

# 计算最优截断值
roc_obj_train <- roc(y_train, train_pred)
cutoff_youden <- function(roc) {
  cutoff <- roc$thresholds[which.max(roc$sensitivities + roc$specificities)]
  return(round(cutoff, 4))
}
roc_c1 <- cutoff_youden(roc_obj_train)
cat("Optimal cutoff (Youden):", roc_c1, "\n")

# 应用截断值
test_data$pred_prob <- test_pred
test_data$pre_value_youden <- ifelse(test_data$pred_prob > roc_c1, "cancer", "control")
test_data$Truth <- test_data$group
test_data$pre_value_youden <- factor(test_data$pre_value_youden, levels = c("control", "cancer"))

# 计算混淆矩阵
c1 <- confusionMatrix(test_data$pre_value_youden, test_data$Truth, positive = "cancer")

cat("Test set performance at cutoff", roc_c1, ":\n")
cat("  Sensitivity:", round(c1$byClass["Sensitivity"], 4), "\n")
cat("  Specificity:", round(c1$byClass["Specificity"], 4), "\n")
cat("  Accuracy:", round(c1$overall["Accuracy"], 4), "\n")

### 9. 校正图 ###
cat("\nGenerating calibration curve...\n")

# 创建校准数据
train_cal_df <- data.frame(
  pred_prob = train_pred,
  group = y_train,
  dataset = "Training"
)

test_cal_df <- data.frame(
  pred_prob = test_pred,
  group = y_test,
  dataset = "Test"
)

cal_df <- rbind(train_cal_df, test_cal_df)

# 校正图函数
plot_calibration <- function(data, title, bin_width = 0.1) {
  # 创建校准组
  data$cal_group <- cut(data$pred_prob, 
                        breaks = seq(0, 1, bin_width),
                        include.lowest = TRUE)
  
  cal_summary <- data %>%
    group_by(cal_group) %>%
    summarise(
      mean_pred = mean(pred_prob),
      mean_actual = mean(as.numeric(group) - 1),
      n = n(),
      ci_lower = binom.test(sum(as.numeric(group) - 1), n())$conf.int[1],
      ci_upper = binom.test(sum(as.numeric(group) - 1), n())$conf.int[2]
    ) %>%
    filter(n > 0)
  
  # 计算Brier分数
  brier_score <- mean((data$pred_prob - (as.numeric(data$group) - 1))^2)
  ece_score <- mean(abs(cal_summary$mean_pred - cal_summary$mean_actual))
  
  # 创建图
  p <- ggplot(cal_summary, aes(x = mean_pred, y = mean_actual)) +
    geom_point(aes(size = n, color = n), alpha = 0.8) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                  width = 0.02, alpha = 0.6) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40", size = 1) +
    geom_smooth(method = "loess", se = TRUE, color = "#e34a33", alpha = 0.3, size = 1.2) +
    scale_color_gradient(low = "#b2e2e2", high = "#006d2c", name = "Sample size") +
    scale_size_continuous(range = c(3, 10), name = "Sample size") +
    labs(
      title = title,
      subtitle = sprintf("Brier score: %.4f | ECE: %.4f", brier_score, ece_score),
      x = "Predicted Probability",
      y = "Actual Proportion"
    ) +
    theme_minimal(base_size = 20) +
    theme(
      plot.title = element_text(size = 23, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 20, hjust = 0.5),
      axis.title = element_text(size = 20,face = "bold"),
      axis.text = element_text(size = 20),
      legend.position = "right",
      legend.title = element_text(size = 18),  # 增加并加粗
      panel.grid.major = element_line(color = "gray90", size = 0.2),
      panel.grid.minor = element_line(color = "gray95", size = 0.1),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    annotate("text", x = 0.8, y = 0.2, 
             label = sprintf("N = %d", nrow(data)),
             size = 7)
  
  return(p)
}

# 创建校正图
p_cal_train <- plot_calibration(train_cal_df, 
                                "Training Set Calibration",
                                bin_width = 0.1)

p_cal_test <- plot_calibration(test_cal_df,
                               "External Validation Set Calibration",
                               bin_width = 0.1)

# 合并校正图
library(patchwork)
p_calibration_combined <- p_cal_train + p_cal_test +
  plot_annotation(
    title = "",
    theme = theme(
      plot.title = element_text(size = 19, face = "bold", hjust = 0.5)
    )
  )

# 保存校正图
print(p_calibration_combined)
ggsave(file.path(FIG_DIR, "glmnet_fixed_calibration.pdf"), 
       p_calibration_combined, width = 12, height = 6, dpi = 300)
ggsave(file.path(FIG_DIR, "glmnet_fixed_calibration.png"), 
       p_calibration_combined, width = 12, height = 6, dpi = 300)

# 单独保存每个校正图
ggsave(file.path(FIG_DIR, "glmnet_fixed_calibration_train.png"), 
       p_cal_train, width = 6, height = 6, dpi = 300)
ggsave(file.path(FIG_DIR, "glmnet_fixed_calibration_test.png"), 
       p_cal_test, width = 6.2, height = 6, dpi = 300)


### 10. 混淆矩阵分析 ###
cat("\nPerforming cutoff analysis...\n")

# 计算最优截断值
roc_obj <- roc(y_train, train_pred)
cutoff_youden <- function(roc) {
  cutoff <- roc$thresholds[which.max(roc$sensitivities + roc$specificities)]
  return(round(cutoff, 4))
}

roc_c1 <- cutoff_youden(roc_obj)

# 应用截断值
test_data$pred_prob <- test_pred
test_data$pre_value_youden <- ifelse(test_data$pred_prob > roc_c1, "cancer", "control")
test_data$Truth <- test_data$group
test_data$pre_value_youden <- factor(test_data$pre_value_youden, levels = c("control", "cancer"))

# 计算混淆矩阵
c1 <- confusionMatrix(test_data$pre_value_youden, test_data$Truth, positive = "cancer")

# 绘制混淆矩阵
plot_confusion_matrix <- function(cm, title) {
  data <- as.data.frame(cm$table)
  colnames(data) <- c("Prediction", "Reference", "Count")
  ggplot(data, aes(x = Reference, y = Prediction, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Count), color = "black", 
              size = 6, fontface = "bold") +
    scale_fill_gradient(low = "#edf8fb", high = "#006d2c") +
    labs(title = title, x = "Actual Class", y = "Predicted Class") +
    theme_minimal(base_size = 14) +
    theme(
      panel.grid = element_blank(), 
      legend.position = "right",
      text = element_text(size = 16),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 16),
      plot.title = element_text(size = 18)
    )
}

p_cm_youden <- plot_confusion_matrix(c1, paste("Youden Index (Threshold =", roc_c1, ")"))

# 保存混淆矩阵
ggsave(file.path(FIG_DIR, "glmnet_fixed_confusion_matrix.pdf"), 
       p_cm_youden, width = 7, height = 6)

### 11. 预测概率分布 ###
cat("\nGenerating prediction probability distribution...\n")

performance_df <- data.frame(
  pred_prob = c(train_pred, test_pred),
  group = c(y_train, y_test),
  dataset = rep(c("Training", "Test"), c(length(train_pred), length(test_pred)))
)

p_distribution <- ggplot(performance_df, aes(x = group, y = pred_prob, fill = group)) +
  geom_boxplot(outlier.alpha = 0.0, alpha = 0.7) +
  geom_jitter(color = 'black', fill = 'white', position = position_jitter(0.12), 
              shape = 21, size = 1.5, alpha = 0.6) +
  geom_signif(comparisons = list(c("control", "cancer")), 
              textsize = 4, 
              map_signif_level = TRUE) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none") +
  facet_grid(~ dataset) +
  labs(
    title = "glmnet Fixed Parameters - Prediction Probability Distribution",
    x = "Group", 
    y = "Predicted Probability"
  )

print(p_distribution)
ggsave(file.path(FIG_DIR, "glmnet_fixed_prediction_distribution.pdf"), 
       p_distribution, width = 10, height = 6)

### 12. 瀑布图分析 ###
cat("\nGenerating waterfall plot...\n")

do_waterfall <- function(test, cut_off) {
  test$dif <- test$pred_prob - cut_off
  test <- test[order(test$dif), ]
  test$results <- ifelse(test$pre_value_youden == test$Truth, "correct", "wrong")
  
  ggplot(test, aes(x = seq_along(dif), y = dif, fill = results)) +
    geom_bar(stat = "identity", width = 1) +
    labs(
      x = 'Subjects', 
      y = 'Difference (Predicted Probability vs. Cutoff)',
      title = paste('Waterfall Plot (Cutoff =', round(cut_off, 4), ')')
    ) +
    theme_minimal(base_size = 18) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_manual(values = c("correct" = "#2ca25f", "wrong" = "#e34a33")) +
    theme(
      axis.text.x = element_blank(), 
      axis.ticks.x = element_blank(),
      plot.title = element_text(size = 18, face = "bold")
    )
}

# 使用你的cutoff值 0.330
waterfall_plot <- do_waterfall(test_data, cut_off = 0.330)
print(waterfall_plot)
ggsave(file.path(FIG_DIR, "glmnet_fixed_waterfall_plot.tiff"), 
       waterfall_plot, width = 10.5, height = 7.03, dpi = 300)



### 15. 打印最终总结 ###
cat("\n" , strrep("=", 60), "\n")
cat("glmnet Model with Fixed Parameters - Final Summary\n")
cat(strrep("=", 60), "\n")
cat("Parameters:\n")
cat("  alpha = 0.5281\n")
cat("  lambda = 0.004597\n\n")
cat("Performance:\n")
cat("  Youden cutoff:", roc_c1, "\n\n")
cat("Features:\n")
cat("  Total features:", ncol(x_train), "\n")
cat("  Non-zero coefficients:", nrow(feature_importance), "\n\n")
cat("Output Directories:\n")
cat("  Figures:", FIG_DIR, "\n")
cat("  Data:", DATA_DIR, "\n\n")

cat("Top 5 most important features (by coefficient):\n")
print(head(feature_importance, 5))

cat("\nTop 5 most important features (by permutation):\n")
print(head(perm_importance, 5))

cat("\nAnalysis complete! All results have been saved.\n")
cat(strrep("=", 60), "\n")

create_simple_waterfall <- function(shap_values, feature_values, feature_names,
                                    sample_idx, sample_type, actual_label, 
                                    predicted_prob, max_features = 6, 
                                    base_font_size = 14) {
  
  library(ggplot2)
  library(dplyr)
  
  # 获取数据
  shap_row <- as.numeric(shap_values[sample_idx, ])
  feature_row <- as.numeric(feature_values[sample_idx, ])
  
  # 选择最重要的特征
  important_idx <- order(abs(shap_row), decreasing = TRUE)[1:min(max_features, length(shap_row))]
  
  # 创建数据框
  df <- data.frame(
    Feature = feature_names[important_idx],
    SHAP = shap_row[important_idx],
    Value = feature_row[important_idx],
    stringsAsFactors = FALSE
  )
  
  # 按SHAP值排序（从大到小）
  df <- df[order(df$SHAP, decreasing = TRUE), ]
  
  # 关键修复：计算真实的对数几率
  # predicted_prob 是概率，需要转换为对数几率
  log_odds_final <- log(predicted_prob / (1 - predicted_prob))
  
  # 计算基线（平均预测的对数几率）
  # 通常 SHAP 基线是 0，但显示时应该是概率 0.5
  baseline <- 0  # 对数几率尺度的基线
  baseline_prob <- 0.5    # 概率尺度的基线
  
  # 累加SHAP值得到最终对数几率（应该接近 log_odds_final）
  df$Cumulative <- baseline + cumsum(df$SHAP)
  df$Start <- c(baseline, head(df$Cumulative, -1))
  
  # 添加最终预测
  final_pred <- tail(df$Cumulative, 1)
  
  # 由于只显示部分特征，final_pred 可能不等于 log_odds_final
  # 我们添加一个"Other features"项来补齐
  other_shap <- log_odds_final - final_pred
  if (abs(other_shap) > 0.001) {
    other_df <- data.frame(
      Feature = "Other features",
      SHAP = other_shap,
      Value = NA,
      Cumulative = log_odds_final,
      Start = final_pred,
      stringsAsFactors = FALSE
    )
    df <- rbind(df, other_df)
  }
  
  # 更新最终预测
  final_pred <- log_odds_final
  
  # 添加行类型和Y位置
  df$Type <- "Feature"
  df$Type[df$Feature == "Other features"] <- "Other"
  df$Y <- 1:nrow(df)
  
  # 添加基线和最终值
  baseline_df <- data.frame(
    Feature = "Baseline",
    SHAP = NA,
    Value = NA,
    Cumulative = baseline,
    Start = baseline,
    Type = "Baseline",
    Y = 0,
    stringsAsFactors = FALSE
  )
  
  final_df <- data.frame(
    Feature = "Final prediction",
    SHAP = NA,
    Value = NA,
    Cumulative = final_pred,
    Start = final_pred,
    Type = "Final",
    Y = nrow(df) + 1,
    stringsAsFactors = FALSE
  )
  
  # 合并所有数据
  plot_df <- rbind(baseline_df, df, final_df)
  plot_df$Y <- 1:nrow(plot_df)
  
  # 创建颜色
  plot_df$Color <- ifelse(plot_df$Type == "Baseline", "white",
                          ifelse(plot_df$Type == "Final", "#1f77b4",
                                 ifelse(plot_df$SHAP > 0, "#e34a33","#2ca25f")))
  
  # 计算每个条形的宽度和位置
  bar_width <- 0.4 
  # 创建标签
  # 动态计算标签位置
  plot_df <- plot_df %>%
    mutate(
      # 条形的Y轴位置（用于绘图）
      Y_pos = Y,
      # 条形的左侧和右侧位置
      X_left = Y - bar_width,
      X_right = Y + bar_width,
      # 条形的中心位置
      X_center = Y,
      X_offset = case_when(
        Type == "Baseline" ~ 0,  # 向右偏移
        Type == "Final" ~ 0,     # 向右偏移
        SHAP > 0 ~ 0,               # 正值标签居中或根据需要调整
        SHAP < 0 ~ 0,               # 负值标签居中或根据需要调整
        TRUE ~ 0                    # 其他情况
      ),
      Bar_Label_X_Adj = X_center + X_offset,
      # 条形的开始和结束位置（Y轴方向）
      Y_start = ifelse(Type == "Baseline", 0.3, Start),
      Y_end = Cumulative,
      # 条形的中心高度
      Y_center = (Y_start + Y_end) / 2,
      # 条形的高度（用于判断标签位置）
      Bar_Height = abs(Y_end - Y_start),
      
      # 条形内部标签（SHAP值）
      Bar_Label = case_when(
        Type == "Baseline" ~ sprintf("E[f(X)] = %.3f", exp(baseline)/(1+exp(baseline))),
        Type == "Final" ~ sprintf("f(X) = %.3f", exp(final_pred)/(1+exp(final_pred))),
        TRUE ~ sprintf("%+.3f", SHAP)
      ),
      
      
      # 确定标签位置 - 特征标签放在条形外部
      # 对于正SHAP值（绿色条形）：标签放在条形左侧
      # 对于负SHAP值（红色条形）：标签放在条形右侧
      
      
      # 特征标签的Y位置（垂直位置）
      
      
      # 特征标签的水平对齐方式
      Feature_Hjust = 0.5,
      
      # 特征标签的垂直对齐方式
      Feature_Vjust = 0.5,
      
      # 条形内部标签的位置（SHAP值）
      Bar_Label_X = Bar_Label_X_Adj,
      Bar_Label_Y = Y_center,
      
      # 条形内部标签的颜色
      Bar_Label_Color = ifelse(Type %in% c("Baseline", "Final"), "darkblue", 
                               ifelse(abs(SHAP) > 0.05, "black", "black")),
      
      # 条形内部标签的字体大小（根据条形高度调整）
      Bar_Label_Size = case_when(
        Type %in% c("Baseline", "Final") ~ base_font_size * 0.7,
        Bar_Height > 0.1 ~ base_font_size * 0.7,
        Bar_Height > 0.05 ~ base_font_size * 0.7,
        TRUE ~ base_font_size * 0.7
      ),
      
      # 特征标签的字体大小
      
      
      # 连接线的位置
      Line_Y = ifelse(Type == "Feature", Start, NA)
    )
  
  
  # 绘制瀑布图
  p <- ggplot(plot_df, aes(x = Y_pos)) +
    
    # 连接线（仅用于特征部分，连接相邻条形）
    geom_segment(data = plot_df[plot_df$Type == "Feature", ],
                 aes(x = X_left + 0.1, xend = X_right - 0.1, 
                     y = Line_Y, yend = Line_Y),
                 color = "gray70", 
                 linetype = "dotted", 
                 linewidth = 0.5) +
    # 瀑布条
    geom_rect(aes(xmin = X_left, 
                  xmax = X_right,
                  ymin = Y_start, 
                  ymax = Y_end,
                  fill = Color),
              alpha = 0.8) +
    
    # 条形内部标签（SHAP值）
    geom_text(aes(x = Bar_Label_X, 
                  y = Bar_Label_Y,
                  label = Bar_Label,
                  color = Bar_Label_Color),
              size = plot_df$Bar_Label_Size,
              fontface = "bold",
              show.legend = FALSE) +
    
    # 特征名称和值标签（在条形外部）
    
    
    # 设置颜色
    scale_fill_identity() +
    scale_color_identity() +
    
    # 坐标轴翻转
    coord_flip() +
    
    # 调整Y轴（特征名称）
    scale_x_continuous(
      breaks = plot_df$Y_pos,
      labels = plot_df$Feature,
      expand = expansion(mult = 0.15)  # 给外部标签留出空间
    ) +
    
    # 调整X轴（SHAP值）
    scale_y_continuous(
      expand = expansion(mult = 0.2)  # 给外部标签留出空间
    ) +
    
    # 标签和主题
    labs(
      title = sprintf("Waterfall Plot - %s Sample", sample_type),
      subtitle = sprintf("Actual: %s | Predicted Probability: %.3f", 
                         actual_label, predicted_prob),
      x = "",
      y = "SHAP Contribution to Log-Odds"
    ) +
    
    theme_minimal(base_size = base_font_size) +
    theme(
      plot.title = element_text(hjust = 0.5, 
                                face = "bold", 
                                size = base_font_size * 1.9,
                                margin = margin(b = 10)),
      plot.subtitle = element_text(hjust = 0.5, 
                                   size = base_font_size * 1.5,
                                   margin = margin(b = 15)),
      axis.title = element_text(face = "bold", 
                                size = base_font_size * 1.5),
      axis.text = element_text(size = base_font_size* 1.5),
      axis.text.y = element_text(face = "bold", 
                                 size = base_font_size * 1.5),
      axis.text.x = element_text(size = base_font_size* 1.7),
      panel.grid.major.x = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor.x = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.major.y = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor.y = element_line(color = "gray90", linewidth = 0.5),
      legend.position = "none",
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(20, 50, 20, 50)
    )
  
  return(p)
}
# 在您的循环中使用
cat("Creating custom waterfall plots...\n")

if (exists("shap_values") && exists("explain_data")) {
  # 创建PDF文件
  pdf(file.path(FIG_DIR, "glmnetAIC_custom_waterfall_plots.pdf"), 
      width = 21, height = 18)
  
  for (i in 1:3) {
    # 找到在explain_data中的索引
    explain_idx <- which(explain_indices == sample_indices[i])
    
    if (length(explain_idx) > 0) {
      actual_label <- ifelse(is.na(actual_labels[explain_idx]), 
                             "Unknown", 
                             actual_labels[explain_idx])
      
      cat(sprintf("  Creating waterfall plot for %s sample...\n", sample_names[i]))
      
      # 使用简化版本（推荐）
      waterfall_plot <- create_simple_waterfall(
        shap_values = shap_values,
        feature_values = explain_data,
        feature_names = colnames(explain_data),
        sample_idx = explain_idx,
        sample_type = sample_names[i],
        actual_label = actual_label,
        predicted_prob = pred_probs[explain_idx],
        max_features = 6,
        base_font_size = 14  # 控制字体大小
      )
      
      print(waterfall_plot)
      
      # 单独保存每个图
      
      ggsave(file.path(FIG_DIR, 
                       sprintf("glmnetAIC_waterfall_%s.png", 
                               gsub(" ", "_", sample_names[i]))),
             waterfall_plot, width = 12, height = 8, dpi = 300)
    }
  }
  
  dev.off()
}

# 更新您的原始代码，替换瀑布图部分
cat("Updating the original waterfall plot code...\n")

# 在循环中替换原有的瀑布图代码
for (i in 1:3) {
  idx <- which(explain_indices == sample_indices[i])
  actual_label <- ifelse(is.na(actual_labels[idx]), "Unknown", actual_labels[idx])
  
  # 替换原有的sv_waterfall调用
  cat(sprintf("  Generating custom waterfall plot for %s sample...\n", sample_names[i]))
  
  p_waterfall_custom <- create_simple_waterfall(
    shap_values = shap_values,
    feature_values = explain_data,
    feature_names = colnames(explain_data),
    sample_idx = idx,
    sample_type = sample_names[i],
    actual_label = actual_label,
    predicted_prob = pred_probs[idx],
    max_features = 6,
    base_font_size = 14  # 可以调大字体
  )
  
  print(p_waterfall_custom)
}
# 优化注释位置的自定义力力图函数
create_custom_force_diagram <- function(shap_values, feature_values, feature_names,
                                        sample_idx, sample_type, actual_label, 
                                        predicted_prob, max_features = 6,
                                        base_font_size = 14) {
  
  library(ggplot2)
  library(dplyr)
  library(stringr)
  
  # 获取该样本的数据
  shap_row <- as.numeric(shap_values[sample_idx, ])
  feature_row <- as.numeric(feature_values[sample_idx, ])
  
  # 按SHAP绝对值选择最重要的特征
  shap_abs <- abs(shap_row)
  top_indices <- order(shap_abs, decreasing = TRUE)[1:min(max_features, length(shap_row))]
  
  # 创建数据框
  plot_data <- data.frame(
    Feature = feature_names[top_indices],
    SHAP = shap_row[top_indices],
    Value = feature_row[top_indices],
    stringsAsFactors = FALSE
  )
  
  # 按SHAP值排序（从小到大）
  plot_data <- plot_data[order(plot_data$SHAP), ]
  rownames(plot_data) <- NULL
  
  # 计算累计位置
  baseline <- 0  # SHAP基线
  plot_data$Cumulative <- baseline + cumsum(plot_data$SHAP)
  plot_data$Start <- c(baseline, head(plot_data$Cumulative, -1))
  plot_data$End <- plot_data$Cumulative
  
  # 计算真实的对数几率
  log_odds_final <- log(predicted_prob / (1 - predicted_prob))
  
  # 添加"其他特征"项（如果需要）
  final_pred_from_shown <- tail(plot_data$Cumulative, 1)
  other_shap <- log_odds_final - final_pred_from_shown
  if (abs(other_shap) > 0.001) {
    other_df <- data.frame(
      Feature = "Other features",
      SHAP = other_shap,
      Value = NA,
      Cumulative = log_odds_final,
      Start = final_pred_from_shown,
      End = log_odds_final,
      stringsAsFactors = FALSE
    )
    plot_data <- rbind(plot_data, other_df)
  }
  
  # 更新最终预测
  final_prediction <- predicted_prob
  final_log_odds <- log_odds_final
  
  # 重新计算Y位置 - 这里我们为基线和最终预测留出位置
  # 基线在位置0，特征从1开始，最终预测在最后
  plot_data$Y <- 1:nrow(plot_data)
  plot_data$Bar_Center <- (plot_data$Start + plot_data$End) / 2
  
  # 添加基线和最终预测到数据中，用于Y轴标签
  baseline_y <- 0
  final_y <- nrow(plot_data) + 1
  
  # 创建包含基线和最终预测的完整标签向量
  y_positions <- c(baseline_y, plot_data$Y, final_y)
  y_labels <- c("baseline", plot_data$Feature, "Final prediction")
  
  # 调整绘图数据的Y位置（在Y轴标签中已经考虑了基线和最终预测）
  # 实际的条形图还是只画特征部分
  
  # 创建标签
  plot_data$SHAP_Label <- sprintf("%+.3f", plot_data$SHAP)
  
  # 确定颜色 - 与瀑布图保持一致：正=红色，负=绿色
  plot_data$Color <- ifelse(plot_data$SHAP > 0, "#e34a33", "#2ca25f")
  
  # 计算特征标签位置
  plot_data <- plot_data %>%
    mutate(
      # 特征名称和数值标签
      Feature_Value_Label = ifelse(!is.na(Value), 
                                   sprintf("%s\n%.2f", Feature, Value),
                                   Feature),
      
      # 特征标签位置（在条形左侧或右侧）
      Feature_Label_X = ifelse(SHAP > 0, Start - 0.02, End + 0.02),
      Feature_Label_Hjust = ifelse(SHAP > 0, 1, 0),  # 1=右对齐，0=左对齐
      
      # SHAP标签位置（在条形中心）
      SHAP_Label_X = Bar_Center,
      SHAP_Label_Y = Y,  # 使用调整后的Y位置
      
      # SHAP标签颜色
      SHAP_Label_Color = "black"
    )
  
  # 确定X轴范围（为标签留出空间）
  x_range <- c(baseline, plot_data$Start, plot_data$End, final_log_odds)
  data_range <- max(x_range) - min(x_range)
  x_min <- min(x_range) - data_range * 0.25
  x_max <- max(x_range) + data_range * 0.25
  
  # 绘制力力图
  p <- ggplot(plot_data, aes(y = Y)) +
    # 背景和基线
    geom_rect(xmin = x_min, xmax = x_max,
              ymin = min(y_positions) - 0.5, ymax = max(y_positions) + 0.5,
              fill = "white", color = NA, alpha = 0.1) +
    
    geom_vline(xintercept = baseline, 
               linetype = "dashed", 
               color = "gray40", 
               linewidth = 1,
               alpha = 0.7) +
    
    # SHAP贡献段（使用geom_segment创建力效果）
    geom_segment(aes(x = Start, xend = End, y = Y, yend = Y, color = Color),
                 linewidth = 18,  # 控制条的粗细
                 lineend = "round",
                 alpha = 0.8) +
    
    # SHAP值标签（在条形中心）
    geom_text(aes(x = SHAP_Label_X, 
                  y = SHAP_Label_Y,
                  label = SHAP_Label,
                  color = SHAP_Label_Color),
              size = base_font_size * 0.7,
              fontface = "bold") +
    
    # 添加最终预测值文本
    annotate("text",
             x = tail(plot_data$End, 1),
             y = final_y,  # 使用最终预测的Y位置
             label = sprintf("f(x) = %.3f", final_prediction),
             size = base_font_size * 0.7,
             fontface = "bold",
             color = "darkblue",
             hjust = 0.3) +
    
    # 添加基线文本
    annotate("text",
             x = baseline,
             y = baseline_y,  # 使用基线的Y位置
             label = sprintf("E[f(x)] = %.3f", 0.5),  # 直接显示概率0.5
             size = base_font_size * 0.7,
             fontface = "bold",
             color = "darkblue",
             hjust = 0.4) +
    
    # 设置颜色
    scale_color_identity() +
    
    # 坐标轴设置 - 关键修复：使用完整的y_labels
    scale_x_continuous(
      limits = c(x_min, x_max),
      expand = expansion(mult = 0.1),
      breaks = scales::pretty_breaks(n = 8),
      name = "SHAP Contribution to Log-Odds"
    ) +
    
    scale_y_continuous(
      breaks = y_positions,  # 包含基线和最终预测的位置
      labels = y_labels,     # 包含基线和最终预测的标签
      limits = c(min(y_positions) - 0.5, max(y_positions) + 0.5),
      expand = expansion(mult = 0.1),
      name = ""
    ) +
    
    # 标签和主题
    labs(
      title = sprintf("Force Diagram - %s Sample", sample_type),
      subtitle = sprintf("Actual: %s | Predicted Probability: %.3f", 
                         actual_label, predicted_prob),
      x = "SHAP Contribution to Log-Odds",
      y = ""
    ) +
    
    theme_minimal(base_size = base_font_size) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", 
                                size = base_font_size * 1.9,
                                margin = margin(b = 10)),
      plot.subtitle = element_text(hjust = 0.5, 
                                   size = base_font_size * 1.5,
                                   margin = margin(b = 15)),
      axis.title = element_text(face = "bold", size = base_font_size * 1.5),
      axis.text = element_text(size = base_font_size* 1.5),
      axis.text.y = element_text(face = "bold", size = base_font_size * 1.5),
      axis.text.x = element_text(size = base_font_size* 1.7),
      panel.grid.major.x = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor.x = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.major.y = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor.y = element_line(color = "gray90", linewidth = 0.5),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(20, 50, 20, 50)
    )
  
  return(p)
}
# 在循环中使用这个函数
cat("Creating custom force diagrams...\n")

# 确保我们有正确的数据
if (exists("shap_values") && exists("explain_data")) {
  
  # 创建单独的PDF用于力力图
  pdf(file.path(FIG_DIR, "glmnetAIC_custom_force_diagrams.pdf"), 
      width = 16, height = 12)
  
  for (i in 1:3) {
    # 找到在explain_data中的索引
    explain_idx <- which(explain_indices == sample_indices[i])
    
    if (length(explain_idx) > 0) {
      actual_label <- ifelse(is.na(actual_labels[explain_idx]), 
                             "Unknown", 
                             actual_labels[explain_idx])
      
      cat(sprintf("  Creating force diagram for %s sample...\n", sample_names[i]))
      
      # 创建自定义力力图
      force_diagram <- create_custom_force_diagram(
        shap_values = shap_values,
        feature_values = explain_data,
        feature_names = colnames(explain_data),
        sample_idx = explain_idx,  # 使用在explain_data中的索引
        sample_type = sample_names[i],
        actual_label = actual_label,
        predicted_prob = pred_probs[explain_idx],
        max_features = 6,
        base_font_size = 14  # 控制基础字体大小
      )
      
      print(force_diagram)
      
      # 也单独保存每个图
      
      ggsave(file.path(FIG_DIR, 
                       sprintf("glmnetAIC_force_diagram_%s.png", 
                               gsub(" ", "_", sample_names[i]))),
             force_diagram, width = 12, height = 8, dpi = 300)
    }
  }
  
  dev.off()
  
} else {
  cat("Warning: shap_values or explain_data not found. Using alternative approach...\n")
  
  # 备用方案：直接从shapviz对象提取数据
  if (exists("sv")) {
    pdf(file.path(FIG_DIR, "glmnetAIC_force_diagrams_from_sv.pdf"), 
        width = 16, height = 12)
    
    for (i in 1:3) {
      explain_idx <- which(explain_indices == sample_indices[i])
      
      if (length(explain_idx) > 0) {
        actual_label <- ifelse(is.na(actual_labels[explain_idx]), 
                               "Unknown", 
                               actual_labels[explain_idx])
        
        # 从shapviz对象提取数据
        shap_row <- sv$S[explain_idx, ]
        feature_row <- sv$X[explain_idx, ]
        
        # 创建简化版力力图
        create_simple_diagram <- function(shap_row, feature_row, title) {
          # 选择最重要的6个特征
          important_idx <- order(abs(shap_row), decreasing = TRUE)[1:6]
          
          df <- data.frame(
            Feature = names(shap_row)[important_idx],
            SHAP = as.numeric(shap_row[important_idx]),
            Value = as.numeric(feature_row[important_idx])
          )
          
          df <- df[order(df$SHAP), ]
          df$Y <- 1:nrow(df)
          df$Cumulative <- cumsum(df$SHAP)
          df$Start <- c(0, head(df$Cumulative, -1))
          
          ggplot(df, aes(y = Y)) +
            geom_segment(aes(x = Start, xend = Cumulative, y = Y, yend = Y,
                             color = ifelse(SHAP > 0, "#2ca25f", "#e34a33")),
                         size = 15, alpha = 0.7) +
            geom_text(aes(x = Start - 0.05, y = Y, 
                          label = paste(Feature, "=", round(Value, 2))),
                      hjust = 1, size = 5, fontface = "bold") +
            geom_text(aes(x = (Start + Cumulative)/2, y = Y,
                          label = sprintf("%+.3f", SHAP)),
                      color = "white", size = 4.5, fontface = "bold") +
            scale_color_identity() +
            labs(title = title, x = "SHAP Contribution", y = "") +
            theme_minimal(base_size = 14) +
            theme(axis.text.y = element_blank(),
                  axis.ticks.y = element_blank())
        }
        
        simple_diagram <- create_simple_diagram(
          shap_row = shap_row,
          feature_row = feature_row,
          title = sprintf("Force Diagram - %s\nActual: %s | Predicted: %.3f",
                          sample_names[i], actual_label, pred_probs[explain_idx])
        )
        
        print(simple_diagram)
      }
    }
    
    dev.off()
  }
}

cat("\nCustom force diagrams have been created successfully!\n")
cat("Files saved in:", FIG_DIR, "\n")
cat("Look for files named:\n")
cat("  - glmnetAIC_custom_force_diagrams.pdf (all diagrams in one file)\n")
cat("  - glmnetAIC_force_diagram_*.pdf (individual diagrams)\n")
cat("  - glmnetAIC_force_diagram_*.png (individual diagrams, high-res)\n")


