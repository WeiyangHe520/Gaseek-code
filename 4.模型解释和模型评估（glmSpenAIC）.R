rm(list = ls())
#################################################
## 临床预测模型开发教学代码（glmStepAIC版本）  ##
## 功能：多特征筛选+模型解释+可视化          ##
## 数据要求：CSV格式，最后一列为分组变量     ##
## 作者：基于罗怀超代码改编                   ##
## 版本：v1.0 (2024-05-25)                    ##
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
set.seed(1234)  # For reproducibility
FIG_DIR <- "figures_glmnetAIC/"    # Output directory for figures
DATA_DIR <- "data_glmnetAIC/"      # Output directory for data

# Create directories if they don't exist
if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR)
if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR)

### 1. Data Loading and Preprocessing ###
# Load datasets
load(file = ".left_data.rdata")



cat("Dataset Information:\n")
cat("Training samples:", nrow(train_data), "\n")
cat("Test samples:", nrow(test_data), "\n")
cat("Cancer prevalence - Training:", mean(train_data$group == "cancer"), 
    "Test:", mean(test_data$group == "cancer"), "\n")

### 2. glmnet Model Training with AIC ###
cat("Training glmnet model with AIC...\n")

# Prepare data for glmnet
x_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$group

x_test <- as.matrix(test_data[, -ncol(test_data)])
y_test <- test_data$group

# Fit glmnet model with cross-validation
cv_fit <- cv.glmnet(x_train, y_train, 
                    family = "binomial",
                    alpha = 1,  # LASSO regularization
                    nfolds = 5,
                    type.measure = "deviance")

# Get the lambda that gives minimum cross-validation error
lambda_min <- cv_fit$lambda.min

# Fit final model with optimal lambda
final_model <- glmnet(x_train, y_train, 
                      family = "binomial",
                      alpha = 1,
                      lambda = lambda_min)

### 3. Feature Importance Analysis ###
cat("Performing feature importance analysis...\n")

# Extract coefficients
coef_matrix <- coef(final_model, s = lambda_min)
feature_importance <- data.frame(
  feature = rownames(coef_matrix)[-1],  # Remove intercept
  coefficient = as.numeric(coef_matrix[-1, 1]),
  abs_coefficient = abs(as.numeric(coef_matrix[-1, 1])),
  stringsAsFactors = FALSE
)

# Remove zero coefficients
feature_importance <- feature_importance[feature_importance$coefficient != 0, ]
feature_importance <- feature_importance[order(-feature_importance$abs_coefficient), ]

# Visualize coefficient importance
p_coefficient_importance <- ggplot(head(feature_importance, 30), 
                                   aes(x = reorder(feature, coefficient), y = coefficient)) +
  geom_bar(stat = "identity", aes(fill = coefficient > 0), alpha = 0.8) +
  scale_fill_manual(values = c("TRUE" = "#2ca25f", "FALSE" = "#e34a33"), 
                    name = "Positive Effect") +
  geom_text(aes(label = sprintf("%.3f", coefficient), 
                hjust = ifelse(coefficient > 0, -0.1, 1.1)), 
            size = 6) +
  coord_flip() +
  labs(
    title = "glmStepAIC - Feature Coefficients",
    subtitle = paste("Top 30 features (lambda =", round(lambda_min, 4), ")"),
    x = "Feature",
    y = "Coefficient Value"
  ) +
  theme_minimal(base_size = 22) +
  theme(
    legend.position = "none",
    # 以下是字体大小调整的具体选项：
    plot.title = element_text(size = 26, face = "bold"),      # 主标题大小
    plot.subtitle = element_text(size = 22),                 # 副标题大小
    axis.title = element_text(size = 24),                    # 坐标轴标题大小
    axis.text.x = element_text(size = 21),                   # X轴刻度标签大小
    axis.text.y = element_text(size = 20)                    # Y轴刻度标签大小（特征名称）
  )

print(p_coefficient_importance)
ggsave(file.path(FIG_DIR, "glmnetAIC_coefficient_importance.pdf"), 
       p_coefficient_importance, width = 21, height = 20)

# Method 2: Permutation Importance
calculate_permutation_importance_glmnet <- function(model, x_data, y_data, n_permutations = 10) {
  baseline_pred <- predict(model, newx = x_data, type = "response")[, 1]
  baseline_auc <- auc(roc(y_data, baseline_pred))
  
  importance_df <- data.frame(
    feature = colnames(x_data),
    importance = 0,
    stringsAsFactors = FALSE
  )
  
  for (feature in colnames(x_data)) {
    auc_drops <- numeric(n_permutations)
    
    for (i in 1:n_permutations) {
      perm_data <- x_data
      perm_data[, feature] <- sample(perm_data[, feature])
      perm_pred <- predict(model, newx = perm_data, type = "response")[, 1]
      perm_auc <- auc(roc(y_data, perm_pred))
      auc_drops[i] <- baseline_auc - perm_auc
    }
    
    importance_df$importance[importance_df$feature == feature] <- mean(auc_drops)
  }
  
  importance_df <- importance_df[order(-importance_df$importance), ]
  return(importance_df)
}

# Calculate permutation importance
perm_importance <- calculate_permutation_importance_glmnet(final_model, x_train, y_train)

# Visualize permutation importance
p_perm_importance <- ggplot(head(perm_importance, 30), 
                            aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "#2ca25f", alpha = 0.8) +
  geom_text(aes(label = sprintf("%.4f", importance)), 
            hjust = -0.1, size = 4, color = "darkgreen") +
  coord_flip() +
  labs(
    title = "glmStepAIC - Permutation Importance",
    subtitle = "Decrease in AUC after feature permutation",
    x = "Feature",
    y = "AUC Decrease"
  ) +
  theme_minimal(base_size = 22) +  # 调整基础字体大小
  theme(
    plot.title = element_text(size = 26, face = "bold"),
    plot.subtitle = element_text(size = 22),
    axis.title.x = element_text(size = 24),
    axis.title.y = element_text(size = 24),
    axis.text.x = element_text(size = 21),
    axis.text.y = element_text(size = 20)  # Y轴特征名称通常稍小
  )

print(p_perm_importance)
ggsave(file.path(FIG_DIR, "glmnetAIC_permutation_importance.pdf"), 
       p_perm_importance, width = 21, height = 20)

### 4. Model Calibration ###
cat("Generating calibration curve...\n")

# Predict probabilities
train_pred <- predict(final_model, newx = x_train, type = "response")[, 1]
test_pred <- predict(final_model, newx = x_test, type = "response")[, 1]

# Create data frames for calibration
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

# Calibration plot function
plot_calibration <- function(data, title) {
  # Create calibration groups
  data$cal_group <- cut(data$pred_prob, 
                        breaks = seq(0, 1, 0.2),
                        include.lowest = TRUE)
  
  cal_summary <- data %>%
    group_by(cal_group) %>%
    summarise(
      mean_pred = mean(pred_prob),
      mean_actual = mean(as.numeric(group) - 1),
      n = n()
    ) %>%
    filter(n > 0)
  
  ggplot(cal_summary, aes(x = mean_pred, y = mean_actual)) +
    geom_point(aes(size = n), color = "#2ca25f") +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
    geom_smooth(method = "loess", se = TRUE, color = "#e34a33", alpha = 0.3) +
    labs(
      title = title,
      x = "Predicted Probability",
      y = "Actual Proportion",
      size = "Number of\nSamples"
    ) +
    theme_minimal(base_size = 12) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))
}

p_cal_train <- plot_calibration(train_cal_df, "glmStepAIC - Training Set Calibration")
p_cal_test <- plot_calibration(test_cal_df, "glmStepAIC - Test Set Calibration")

# Combine calibration plots
library(patchwork)
p_calibration_combined <- p_cal_train + p_cal_test +
  plot_annotation(title = "glmStepAIC - Calibration Analysis")

print(p_calibration_combined)
ggsave(file.path(FIG_DIR, "glmnetAIC_calibration.pdf"), 
       p_calibration_combined, width = 12, height = 6)
#### 5. SHAP Analysis for glmnet Model ###
cat("Performing SHAP analysis for glmnet model...\n")

# 为glmnet模型创建预测函数
glmnet_predict <- function(model, newdata) {
  if (!is.matrix(newdata)) {
    newdata <- as.matrix(newdata)
  }
  predictions <- predict(model, newx = newdata, type = "response", s = lambda_min)
  return(as.numeric(predictions))
}

# 确保安装了必要的包
if (!requireNamespace("shapviz", quietly = TRUE)) {
  install.packages("shapviz")
}
if (!requireNamespace("fastshap", quietly = TRUE)) {
  install.packages("fastshap")
}
library(shapviz)
library(fastshap)

# 选择代表性样本进行SHAP分析
set.seed(123)
n_explain <- min(100, nrow(x_train))  # 使用足够的样本量
explain_indices <- sample(nrow(x_train), n_explain)
explain_data <- x_train[explain_indices, , drop = FALSE]

# 确保数据有正确的列名
colnames(explain_data) <- colnames(x_train)

# 计算SHAP值 - 移除.progress参数
cat("Calculating SHAP values (this may take a while)...\n")
shap_values <- fastshap::explain(
  final_model,
  X = explain_data,
  pred_wrapper = glmnet_predict,
  nsim = 20,  # 适当减少以提高计算速度，对于初始分析足够了
  adjust = TRUE
)

# 创建shapviz对象
sv <- shapviz(shap_values, X = explain_data)

# 获取实际标签
actual_labels <- as.character(y_train[explain_indices])

# 计算预测概率
pred_probs <- glmnet_predict(final_model, explain_data)

# 1. 生成散点图/蜂群图（第一页）
cat("Generating SHAP scatter/beeswarm plot (Page 1)...\n")
pdf(file.path(FIG_DIR, "glmnetAIC_SHAP_beeswarm.pdf"), width = 21, height = 20)

p_beeswarm <- sv_importance(sv, kind = "bee", max_display = 30, 
                            fill = "#2ca25f", alpha = 0.7,
                            bee_width = 0.25) + 
  ggtitle("glmStepAIC - SHAP Importance (Top 30 Features)") +
  theme_classic(base_size = 14) +
  theme(plot.title = element_text(size = 26, face = "bold"),      # 主标题大小
        plot.subtitle = element_text(size = 22),                 # 副标题大小
        axis.title = element_text(size = 24),                    # 坐标轴标题大小
        axis.text = element_text(size = 24),  # 坐标轴刻度标签
        legend.title = element_text(size = 21),  # 图例标题
        legend.text = element_text(size = 20)  # 图例文本
       )+
  labs(x = "SHAP value", y = "Feature")

print(p_beeswarm)
dev.off()

# 2. 生成条形图重要性图（单独保存）
cat("Generating SHAP bar importance plot...\n")
pdf(file.path(FIG_DIR, "glmnetAIC_SHAP_bar_importance.pdf"), width = 21, height = 20)

p_bar <- sv_importance(sv, kind = "bar", max_display = 30, 
                       fill = "#2ca25f", color = "darkgreen") + 
  ggtitle("glmStepAIC - Mean |SHAP| Value (Top 30 Features)") +
  theme_classic(base_size = 14) +
  theme(plot.title = element_text(size = 26, face = "bold"),      # 主标题大小
        plot.subtitle = element_text(size = 22),                 # 副标题大小
        axis.title = element_text(size = 24),                    # 坐标轴标题大小
        axis.text = element_text(size = 21),  # 坐标轴刻度标签
        legend.title = element_text(size = 15),  # 图例标题
        legend.text = element_text(size = 14)  # 图例文本
  ) +
  labs(x = "Mean |SHAP| value", y = "Feature")

print(p_bar)
dev.off()

# 3. 选择代表性样本进行详细分析
# 根据预测概率选择3个代表性样本
cancer_indices_all <- explain_indices[actual_labels == "cancer"]
control_indices_all <- explain_indices[actual_labels == "control"]

if (length(cancer_indices_all) >= 2 && length(control_indices_all) >= 1) {
  # 获取预测概率
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
  pdf(file.path(FIG_DIR, "glmnetAIC_SHAP_detailed_analysis.pdf"), width = 21, height = 20)
  
  # 瀑布图和力图对每个样本
  for (i in 1:3) {
    idx <- which(explain_indices == sample_indices[i])
    
    # 获取实际标签（确保不是NA）
    actual_label <- ifelse(is.na(actual_labels[idx]), "Unknown", actual_labels[idx])
    
    # 瀑布图
    cat(sprintf("  Generating waterfall plot for %s sample...\n", sample_names[i]))
    # 瀑布图 - 专门修改核心文本
    p_waterfall <- sv_waterfall(
      sv, 
      row_id = idx, 
      max_display = 6, 
      fill_phi = ifelse(pred_probs[idx] > 0.5, "#e34a33", "#2ca25f"),
      # 控制核心文本的参数
      size = 15,                    # 基础文本大小
      size_waterfall = 15,            # 瀑布条文本
      size_annotation = 12,
      show_annotation = TRUE,  # 确保显示注释
      annotation_size = 10,      # 注释大小
      digits = 3,                    # 小数位数
      format_fun = function(x) sprintf("%.3f", x)  # 格式化函数
    ) +  
      ggtitle(sprintf("Waterfall Plot - %s\nActual: %s | Predicted: %.3f", 
                      sample_names[i], actual_label, pred_probs[idx])) +
      theme_classic(base_size = 14) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 26),  # 标题字体
        axis.title = element_text(size = 22),      # 坐标轴标题
        axis.text = element_text(size = 21),       # 坐标轴刻度
        axis.text.x = element_text(size = 21, angle = 0),  # x轴刻度
        axis.text.y = element_text(size = 21),     # y轴刻度
        legend.title = element_text(size = 22),    # 图例标题
        legend.text = element_text(size = 22)      # 图例文字
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
      # 控制文本的参数
      size = 0,
      size_force = 0,
      size_annotation = 0, 
      show_annotation = TRUE,  # 确保显示注释
      annotation_size = 10) +
      # 添加 geom_text_repel 的字体大小控制
      
      ggtitle(sprintf("Force Plot - %s\nActual: %s | Predicted: %.3f", 
                      sample_names[i], actual_label, pred_probs[idx])) +
      theme_classic(base_size = 14) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 26),  # 标题字体
        axis.title = element_text(size = 22),      # 坐标轴标题
        axis.text = element_text(size = 21),       # 坐标轴刻度
        axis.text.x = element_text(size = 21, angle = 0),  # x轴刻度
        axis.text.y = element_text(size = 21),     # y轴刻度
        legend.title = element_text(size = 22),    # 图例标题
        legend.text = element_text(size = 22)      # 图例文字
      )
    
    print(p_force)
  }
  
  dev.off()
  
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
  
  # 5. 保存SHAP数据
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
  
  saveRDS(shap_data, file.path(DATA_DIR, "glmnetAIC_SHAP_data.rds"))
  
  # 6. 打印样本信息
  cat("\n=== SHAP Analysis - Sample Information ===\n")
  cat(sprintf("Total samples analyzed: %d\n", n_explain))
  cat(sprintf("Cancer samples: %d, Control samples: %d\n", 
              sum(actual_labels == "cancer"), sum(actual_labels == "control")))
  cat("\nRepresentative samples selected:\n")
  
  for (i in 1:3) {
    idx <- which(explain_indices == sample_indices[i])
    actual_label <- actual_labels[idx]
    
    cat(sprintf("\nSample %d: %s\n", i, sample_names[i]))
    cat(sprintf("  Sample index: %d\n", sample_indices[i]))
    cat(sprintf("  Actual label: %s\n", actual_label))
    cat(sprintf("  Predicted probability: %.3f\n", pred_probs[idx]))
    
    # 获取该样本的SHAP贡献
    shap_contrib <- shap_values[idx, ]
    top_features_idx <- order(abs(shap_contrib), decreasing = TRUE)[1:5]
    
    cat(sprintf("  Top 5 contributing features:\n"))
    for (j in 1:5) {
      feat_idx <- top_features_idx[j]
      feat_name <- colnames(shap_values)[feat_idx]
      feat_value <- explain_data[idx, feat_name]
      shap_val <- shap_contrib[feat_idx]
      cat(sprintf("    %-15s = %7.2f  (SHAP: %7.4f)\n", 
                  feat_name, feat_value, shap_val))
    }
  }
  
  # 7. 生成汇总报告
  cat("\n=== SHAP Analysis Summary ===\n")
  cat("Generated PDF files:\n")
  cat("  1. glmnetAIC_SHAP_beeswarm.pdf      - Beeswarm plot (Page 1 from example)\n")
  cat("  2. glmnetAIC_SHAP_bar_importance.pdf - Bar importance plot\n")
  cat("  3. glmnetAIC_SHAP_detailed_analysis.pdf - Waterfall & force plots\n")
  cat("  4. glmnetAIC_SHAP_dependence.pdf     - Dependence plots\n")
  cat("\nSHAP data saved to: glmnetAIC_SHAP_data.rds\n")
  
} else {
  cat("\nWarning: Insufficient samples for detailed SHAP analysis.\n")
  cat(sprintf("  Cancer samples: %d (need at least 2)\n", length(cancer_indices_all)))
  cat(sprintf("  Control samples: %d (need at least 1)\n", length(control_indices_all)))
  cat("Generating basic SHAP plots only...\n")
  
  # 生成基本的SHAP图
  pdf(file.path(FIG_DIR, "glmnetAIC_SHAP_basic.pdf"), width = 12, height = 10)
  print(p_beeswarm)
  print(p_bar)
  dev.off()
}

cat("\nSHAP analysis completed successfully.\n")
### 5. Feature Correlation Analysis ###
cat("Analyzing feature correlations...\n")

# Select top features for correlation analysis
top_features <- head(perm_importance$feature, 44)

if(length(top_features) > 1) {
  # Create correlation matrix
  cor_data <- train_data[, top_features]
  cor_matrix <- cor(cor_data, method = "spearman", use = "complete.obs")
  
  pdf(file.path(FIG_DIR, "glmnetAIC_feature_correlations.pdf"), width = 21, height = 20)
  corrplot(cor_matrix, 
           type = "lower",
           tl.col = "black",
           tl.srt = 45,
           tl.cex = 1.2,
           method = "square",
           order = "AOE",
           addCoef.col = "black",
           number.cex = 0.85,
           col = COL2("BrBG"),
           diag = FALSE,
           mar = c(0, 0, 0, 0),
           title = "glmStepAIC - Feature Correlations")
  # 使用 title() 添加自定义标题
  title(main = "glmStepAIC - Feature Correlations",
        cex.main = 2,          # 标题字体大小，默认是1.2
        font.main = 2,           # 字体样式：1=常规，2=粗体，3=斜体，4=粗斜体
        line = -1)              # 标题位置，正值向上移动，负值向下移动
  dev.off()
}

### 6. Confounding Adjustment Analysis ###
cat("Performing confounding adjustment...\n")

# Prepare test data for confounding analysis
test_data_numeric <- test_data
test_data_numeric$group <- as.numeric(test_data_numeric$group) - 1
test_data_numeric$pred_prob <- test_pred

# Select top features for adjustment
top_features_confound <- head(perm_importance$feature, 30)

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
                         midpoint = 0, name = "Coefficient") +
    coord_flip() + 
    theme_minimal() + 
    labs(title = 'glmStepAIC - Logit Regression Adjustment', 
         subtitle = 'Adjusting for top 30 important features',
         x = 'Features', y = '-log10 P_value') +
    geom_hline(yintercept = 1.30103, linetype = "dashed", color = "red", size = 1.0) +
    geom_hline(yintercept = 1.30103, linetype = "dashed", color = "red", size = 1.0) +
    theme(
      text = element_text(size = 14),  # 全局字体大小（默认12）
      plot.title = element_text(size = 26, face = "bold"),      # 主标题大小
      plot.subtitle = element_text(size = 22),                 # 副标题大小
      axis.title = element_text(size = 24),                    # 坐标轴标题大小
      axis.text = element_text(size = 21),  # 坐标轴刻度标签
      legend.title = element_text(size = 15),  # 图例标题
      legend.text = element_text(size = 14)  # 图例文本
    ) + 
    annotate(geom = 'text', y = 1.35, x = 1, size = 4.0, label = 'p=0.05')
  
  print(p_confound)
  ggsave(file.path(FIG_DIR, "glmnetAIC_confounding_adjustment.pdf"), 
         p_confound, width = 21, height = 20)
}
### 7. ROC Analysis ###
cat("Performing ROC analysis...\n")

# Calculate ROC curves
roc_train <- roc(y_train, train_pred)
roc_test <- roc(y_test, test_pred)

# 创建ROC数据
roc_data <- list(
  Training = roc_train,
  `Internal Validation` = roc_test
)

# 方法1：使用ggroc的传统方法（适用于较新版本）
roc_plot <- ggroc(roc_data) +
  geom_line(size = 1.2) +  # 添加线条
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "gray") +
  labs(
    title = "glmStepAIC - ROC Curve Analysis",
    color = "Dataset"
  ) +
  scale_color_manual(values = c("Training" = "#4daf4a", 
                                "Internal Validation" = "#377eb8")) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = c(0.85, 0.25),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  annotate(
    "text", 
    x = 0.4, y = 0.25, 
    label = paste('Training AUC:', round(roc_train$auc, 3)),
    color = "#4daf4a", size = 5
  ) +
  annotate(
    "text", 
    x = 0.4, y = 0.15, 
    label = paste('Internal Validation AUC:', round(roc_test$auc, 3)),
    color = "#377eb8", size = 5
  )

# 或者，方法2：使用更基础的绘图方法
if (!exists("roc_plot")) {
  # 方法2：直接绘制ROC曲线
  roc_plot <- ggplot() +
    # 训练集ROC
    geom_line(data = data.frame(
      Specificity = roc_train$specificities,
      Sensitivity = roc_train$sensitivities,
      Dataset = "Training"
    ), aes(x = 1 - Specificity, y = Sensitivity, color = Dataset),
    size = 1.2) +
    
    # 测试集ROC
    geom_line(data = data.frame(
      Specificity = roc_test$specificities,
      Sensitivity = roc_test$sensitivities,
      Dataset = "Internal Validation"
    ), aes(x = 1 - Specificity, y = Sensitivity, color = Dataset),
    size = 1.2) +
    
    # 对角线
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
    
    # 样式设置
    scale_color_manual(values = c("Training" = "#4daf4a", 
                                  "Internal Validation" = "#377eb8")) +
    labs(
      title = "glmStepAIC - ROC Curve Analysis",
      x = "1 - Specificity (False Positive Rate)",
      y = "Sensitivity (True Positive Rate)",
      color = "Dataset"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = c(0.85, 0.25),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    ) +
    annotate(
      "text", 
      x = 0.4, y = 0.25, 
      label = paste('Training AUC:', round(roc_train$auc, 3)),
      color = "#4daf4a", size = 5
    ) +
    annotate(
      "text", 
      x = 0.4, y = 0.15, 
      label = paste('Internal Validation AUC:', round(roc_test$auc, 3)),
      color = "#377eb8", size = 5
    ) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))
}

print(roc_plot)
ggsave(file.path(FIG_DIR, "glmnetAIC_roc_curve.pdf"), roc_plot, width = 6, height = 6)
### 8. Performance Comparison Boxplot ###
performance_df <- data.frame(
  pred_prob = c(train_pred, test_pred),
  group = c(y_train, y_test),
  dataset = rep(c("Training", "Test"), c(length(train_pred), length(test_pred)))
)

final_eval_plot <- ggplot(performance_df, aes(x = group, y = pred_prob, fill = group)) +
  geom_boxplot(outlier.alpha = 0.0, alpha = 0.7) +
  geom_jitter(color = 'black', fill = 'white', position = position_jitter(0.12), 
              shape = 21, size = 1.5, alpha = 0.6) +
  geom_signif(comparisons = list(c("control", "cancer")), 
              textsize = 4, 
              map_signif_level = TRUE) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal() +
  theme(legend.position = "none") + 
  theme(text = element_text(size = 12)) +
  facet_grid(~ dataset) +
  labs(title = "glmStepAIC - Prediction Probability Distribution",
       x = "Group", y = "Predicted Probability")

print(final_eval_plot)
ggsave(file.path(FIG_DIR, "glmnetAIC_performance_comparison.pdf"), 
       final_eval_plot, width = 10, height = 6)

### 9. Cutoff Analysis and Confusion Matrices ###
cat("Performing cutoff analysis...\n")

# Calculate optimal cutoffs
roc_obj <- roc(y_train, train_pred)

cutoff_youden <- function(roc) {
  cutoff <- roc$thresholds[which.max(roc$sensitivities + roc$specificities)]
  return(round(cutoff, 4))
}

roc_c1 <- cutoff_youden(roc_obj)

# Apply cutoffs to test data
test_data$pred_prob <- test_pred
test_data$pre_value_youden <- ifelse(test_data$pred_prob > roc_c1, "cancer", "control")

# Ensure factor levels
test_data$Truth <- test_data$group
test_data$pre_value_youden <- factor(test_data$pre_value_youden, levels = c("control", "cancer"))

# Calculate confusion matrices
c1 <- confusionMatrix(test_data$pre_value_youden, test_data$Truth, positive = "cancer")

# Plot confusion matrices
plot_confusion_matrix <- function(cm, title) {
  data <- as.data.frame(cm$table)
  colnames(data) <- c("Prediction", "Reference", "Count")
  ggplot(data, aes(x = Reference, y = Prediction, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Count), color = "black", 
              size = 6, fontface = "bold") +  # 修改这里的 size 值
    scale_fill_gradient(low = "#edf8fb", high = "#006d2c") +
    labs(title = title, x = "Actual Class", y = "Predicted Class") +
    theme_minimal(base_size = 14) +  # 增加 base_size
    theme(panel.grid = element_blank(), 
          legend.position = "right",
          text = element_text(size = 16),  # 添加这个设置
          axis.title = element_text(size = 16),
          axis.text = element_text(size = 16),
          plot.title = element_text(size = 18))
}

p_cm_youden <- plot_confusion_matrix(c1, paste("Youden Index (Threshold =", roc_c1, ")"))

# 修改为JPG格式保存混淆矩阵
# Save confusion matrices as JPG
jpeg(file.path(FIG_DIR, "glmnetAIC_confusion_matrices.jpg"), 
     width = 7.2, height = 6, units = "in", res = 300)
gridExtra::grid.arrange(p_cm_youden)
dev.off()

### 10. Waterfall Plot ###
do_waterfall <- function(test, cut_off) {
  test$dif <- test$pred_prob - cut_off
  test <- test[order(test$dif), ]
  test$results <- ifelse(test$pre_value_youden == test$Truth, "correct", "wrong")
  
  ggplot(test, aes(x = seq_along(dif), y = dif, fill = results)) +
    geom_bar(stat = "identity", width = 1) +
    labs(x = 'Subjects', y = 'Difference (Predicted Probability vs. Cutoff)',
         title = paste('glmStepAIC - Waterfall Plot (Threshold =', cut_off, ')')) +
    theme_minimal() +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_manual(values = c("correct" = "#2ca25f", "wrong" = "#e34a33")) +
    theme(axis.text.x = element_blank(), 
          axis.ticks.x = element_blank(),
          text = element_text(size = 16),  # 主要文本大小
          axis.title = element_text(size = 16),  # 坐标轴标题
          axis.text.y = element_text(size = 16),  # Y轴文本
          plot.title = element_text(size = 18),  # 标题
          legend.text = element_text(size = 14),  # 图例文本
          legend.title = element_text(size = 15))  # 图例标题
}

waterfall_plot <- do_waterfall(test_data, cut_off = roc_c1)
print(waterfall_plot)

# 修改为JPG格式保存瀑布图
# Save waterfall plot as JPG
ggsave(file.path(FIG_DIR, "glmnetAIC_waterfall_plot.jpg"), 
       waterfall_plot, width = 10, height = 6, dpi = 300)


### 12. Model Summary and Coefficients ###
cat("Generating model summary...\n")

# Create coefficient table
coefficient_table <- feature_importance
coefficient_table$perm_importance <- perm_importance$importance[match(coefficient_table$feature, perm_importance$feature)]

# Save detailed results
write.csv(coefficient_table, file.path(DATA_DIR, "glmnetAIC_coefficients.csv"), row.names = FALSE)

# Model performance summary
performance_summary <- data.frame(
  Metric = c("Training AUC", "Test AUC", "Optimal Lambda", "Number of Features"),
  Value = c(
    round(roc_train$auc, 3),
    round(roc_test$auc, 3),
    round(lambda_min, 5),
    nrow(feature_importance)
  )
)

write.csv(performance_summary, file.path(DATA_DIR, "glmnetAIC_performance.csv"), row.names = FALSE)

### 13. Save Final Results ###
cat("Saving final results...\n")

save(final_model, cv_fit, feature_importance, perm_importance,
     roc_train, roc_test, performance_df,
     file = file.path(DATA_DIR, "glmnetAIC_final_results.rdata"))

# Print summary
cat("\n=== glmStepAIC Model Summary ===\n")
cat("Training AUC:", round(roc_train$auc, 3), "\n")
cat("Test AUC:", round(roc_test$auc, 3), "\n")
cat("Optimal lambda:", round(lambda_min, 5), "\n")
cat("Number of selected features:", nrow(feature_importance), "\n")
cat("Youden cutoff:", roc_c1, "\n")
cat("\nTop 10 important features:\n")
print(head(feature_importance, 10))
cat("\nAnalysis complete! Results saved to:", FIG_DIR, "and", DATA_DIR, "\n")

# Create a comprehensive summary plot
summary_plot <- (p_coefficient_importance + p_perm_importance) / 
  (roc_plot + p_cal_test) +
  plot_annotation(title = "glmStepAIC Model - Comprehensive Summary",
                  theme = theme(plot.title = element_text(size = 16, face = "bold")))

ggsave(file.path(FIG_DIR, "glmnetAIC_comprehensive_summary.pdf"), 
       summary_plot, width = 16, height = 12)


# 简化版本的瀑布图（更接近shapviz风格）
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
  
  # 计算累计值
  baseline <- 0  # SHAP基线
  df$Cumulative <- baseline + cumsum(df$SHAP)
  df$Start <- c(baseline, head(df$Cumulative, -1))
  
  # 添加最终预测
  final_pred <- tail(df$Cumulative, 1)
  
  # 添加行类型和Y位置
  df$Type <- "Feature"
  df$Y <- 1:nrow(df)
  
  # 添加基线和最终值
  baseline_df <- data.frame(
    Feature = "",
    SHAP = NA,
    Value = NA,
    Cumulative = baseline,
    Start = baseline,
    Type = "Baseline",
    Y = 0
  )
  
  final_df <- data.frame(
    Feature = "",
    SHAP = NA,
    Value = NA,
    Cumulative = final_pred,
    Start = final_pred,
    Type = "Final",
    Y = nrow(df) + 1
  )
  
  # 合并所有数据
  plot_df <- rbind(baseline_df, df, final_df)
  plot_df$Y <- 1:nrow(plot_df)  # 重新编号
  
  # 创建颜色
  plot_df$Color <- ifelse(plot_df$Type == "Baseline", "#999999",
                          ifelse(plot_df$Type == "Final", "#1f77b4",
                                 ifelse(plot_df$SHAP > 0, "#2ca25f", "#e34a33")))
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
      # 条形的开始和结束位置（Y轴方向）
      Y_start = ifelse(Type == "Baseline", 0, Start),
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
      
      # 特征标签（特征名称和值）
      Feature_Label = case_when(
        Type %in% c("Baseline", "Final") ~ Feature,
        TRUE ~ sprintf("%s = %.2f", str_wrap(Feature, 15), Value)
      ),
      
      # 确定标签位置 - 特征标签放在条形外部
      # 对于正SHAP值（绿色条形）：标签放在条形左侧
      # 对于负SHAP值（红色条形）：标签放在条形右侧
      Feature_Label_X = case_when(
        Type %in% c("Baseline", "Final") ~ X_center,
        SHAP > 0 ~ X_left - 0.15,   # 正SHAP：左侧
        SHAP <= 0 ~ X_right + 0.15  # 负SHAP：右侧
      ),
      
      # 特征标签的Y位置（垂直位置）
      Feature_Label_Y = Y_center,
      
      # 特征标签的水平对齐方式
      Feature_Hjust = case_when(
        Type %in% c("Baseline", "Final") ~ 0.5,  # 居中
        SHAP > 0 ~ 1,    # 正SHAP：右对齐（靠近条形）
        TRUE ~ 0         # 负SHAP：左对齐（靠近条形）
      ),
      
      # 特征标签的垂直对齐方式
      Feature_Vjust = 0.5,
      
      # 条形内部标签的位置（SHAP值）
      Bar_Label_X = X_center,
      Bar_Label_Y = Y_center,
      
      # 条形内部标签的颜色
      Bar_Label_Color = ifelse(Type %in% c("Baseline", "Final"), "darkblue", 
                               ifelse(abs(SHAP) > 0.05, "black", "black")),
      
      # 条形内部标签的字体大小（根据条形高度调整）
      Bar_Label_Size = case_when(
        Type %in% c("Baseline", "Final") ~ base_font_size * 0.8,
        Bar_Height > 0.1 ~ base_font_size * 0.7,
        Bar_Height > 0.05 ~ base_font_size * 0.7,
        TRUE ~ base_font_size * 0.7
      ),
      
      # 特征标签的字体大小
      Feature_Label_Size = base_font_size * 0.7,
      
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
    geom_text(aes(x = Feature_Label_X,
                  y = Feature_Label_Y,
                  label = Feature_Label,
                  hjust = Feature_Hjust,
                  vjust = Feature_Vjust),
              size = plot_df$Feature_Label_Size,
              fontface = "bold",
              lineheight = 0.8) +
    
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
                                margin = margin(b = 15)),
      plot.subtitle = element_text(hjust = 0.5, 
                                   size = base_font_size * 1.7,
                                   margin = margin(b = 20)),
      axis.title = element_text(face = "bold", 
                                size = base_font_size * 1.5),
      axis.text = element_text(size = base_font_size),
      axis.text.y = element_text(face = "bold", 
                                 size = base_font_size * 1.5),
      axis.text.x = element_text(size = base_font_size* 1.5),
      panel.grid.major.x = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor.x = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.major.y = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor.y = element_line(color = "gray90", linewidth = 0.5),
      legend.position = "none"
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
                       sprintf("glmnetAIC_waterfall_%s.pdf", 
                               gsub(" ", "_", sample_names[i]))),
             waterfall_plot, width = 21, height = 18)
      
      ggsave(file.path(FIG_DIR, 
                       sprintf("glmnetAIC_waterfall_%s.png", 
                               gsub(" ", "_", sample_names[i]))),
             waterfall_plot, width = 21, height = 18, dpi = 300)
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
  shap_row <- shap_values[sample_idx, ]
  feature_row <- feature_values[sample_idx, ]
  
  # 按SHAP绝对值选择最重要的特征
  shap_abs <- abs(shap_row)
  top_indices <- order(shap_abs, decreasing = TRUE)[1:min(max_features, length(shap_row))]
  
  # 创建数据框
  plot_data <- data.frame(
    Feature = feature_names[top_indices],
    SHAP = as.numeric(shap_row[top_indices]),
    Value = as.numeric(feature_row[top_indices]),
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
  
  # 添加y轴位置（从上到下）
  plot_data$Y <- 1:nrow(plot_data)
  
  # 计算条形的中心位置
  plot_data$Bar_Center <- (plot_data$Start + plot_data$End) / 2
  
  # 创建标签
  plot_data$Feature_Label <- sprintf("%s = %.2f", 
                                     str_wrap(plot_data$Feature, 15),
                                     plot_data$Value)
  plot_data$SHAP_Label <- sprintf("%+.3f", plot_data$SHAP)
  
  # 确定颜色
  plot_data$Color <- ifelse(plot_data$SHAP > 0, "#2ca25f", "#e34a33")
  
  # 创建最终的预测值标签
  final_prediction <- round(predicted_prob, 3)
  
  # 计算标签位置
  plot_data <- plot_data %>%
    mutate(
      # 特征标签位置（在条形左侧或右侧）
      Feature_Label_X = ifelse(SHAP > 0, Start - 0.03, End + 0.03),
      # 特征标签对齐方式
      Feature_Hjust = ifelse(SHAP > 0, 1, 0),  # 1=右对齐，0=左对齐
      # SHAP标签位置（在条形中心）
      SHAP_Label_X = Bar_Center,
      # SHAP标签垂直偏移（稍微偏离中心）
      SHAP_Label_Y = Y + 0.2,
      # 特征标签垂直位置
      Feature_Label_Y = Y
    )
  
  # 确定X轴范围（为标签留出空间）
  x_min <- min(baseline, plot_data$Start, plot_data$End, 
               plot_data$Feature_Label_X) - 0.15
  x_max <- max(baseline, plot_data$Start, plot_data$End, 
               plot_data$Feature_Label_X) + 0.15
  
  # 绘制力力图
  p <- ggplot(plot_data, aes(y = Y)) +
    # 背景和基线
    geom_rect(xmin = x_min, xmax = x_max,
              ymin = 0.5, ymax = nrow(plot_data) + 0.5,
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
    
    # 特征名称和值标签
    geom_text(aes(x = Feature_Label_X,
                  y = Feature_Label_Y,
                  label = Feature_Label,
                  hjust = Feature_Hjust),
              size = base_font_size * 0.7,
              fontface = "bold",
              color = "black",
              lineheight = 0.8) +
    
    # SHAP值标签（在条上方）
    geom_text(aes(x = SHAP_Label_X, 
                  y = SHAP_Label_Y,
                  label = SHAP_Label),
              size = base_font_size * 0.7,
              fontface = "bold",
              color = "black") +
    
    # 最终预测值标记
    geom_point(aes(x = End, y = Y, color = Color), 
               size = 15, 
               shape = 21, 
               fill = "white",
               stroke = 2) +
    
    # 添加最终预测值文本
    annotate("text",
             x = tail(plot_data$End, 1),
             y = tail(plot_data$Y, 1) + 0.4,
             label = sprintf("f(x) = %.3f", final_prediction),
             size = base_font_size * 0.7,
             fontface = "bold",
             color = "darkblue",
             hjust = ifelse(tail(plot_data$SHAP, 1) > 0, 0, 1)) +
    
    # 添加基线文本
    annotate("text",
             x = baseline,
             y = 0.7,
             label = sprintf("E[f(x)] = %.3f", exp(baseline)/(1+exp(baseline))),
             size = base_font_size * 0.7,
             fontface = "bold",
             color = "gray40",
             hjust = ifelse(baseline < mean(c(min(plot_data$Start), max(plot_data$End))), 
                            -0.1, 1.1)) +
    
    # 设置颜色
    scale_color_identity() +
    
    # 坐标轴设置
    scale_x_continuous(
      limits = c(x_min, x_max),
      expand = expansion(mult = 0.05),
      breaks = scales::pretty_breaks(n = 8)
    ) +
    
    scale_y_continuous(
      breaks = plot_data$Y,
      labels = plot_data$Feature,
      limits = c(0.5, nrow(plot_data) + 0.7),  # 为顶部标签留出空间
      expand = expansion(mult = 0.1)
    ) +
    
    # 标签和主题
    labs(
      title = sprintf("Force Diagram - %s Sample", sample_type),
      subtitle = sprintf("Actual: %s | Predicted Probability: %.3f", 
                         actual_label, predicted_prob),
      x = "SHAP Contribution to Log-Odds",
      y = "Feature"
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
      axis.text = element_text(size = base_font_size* 1.7),
      axis.text.y = element_text(face = "bold", size = base_font_size * 1.7),
      axis.text.x = element_text(size = base_font_size* 1.7),
      panel.grid.major.x = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor.x = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.major.y = element_line(color = "gray90", linewidth = 0.5),
      panel.grid.minor.y = element_line(color = "gray90", linewidth = 0.5),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(20, 50, 20, 50)  # 增加左右边距
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
                       sprintf("glmnetAIC_force_diagram_%s.pdf", 
                               gsub(" ", "_", sample_names[i]))),
             force_diagram, width = 21, height = 18)
      
      ggsave(file.path(FIG_DIR, 
                       sprintf("glmnetAIC_force_diagram_%s.png", 
                               gsub(" ", "_", sample_names[i]))),
             force_diagram, width = 21, height = 18, dpi = 300)
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

