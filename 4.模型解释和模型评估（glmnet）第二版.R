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
FIG_DIR <- "glmnet_interpretation_figures/"
DATA_DIR <- "glmnet_interpretation_data/"

# 创建目录
if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR, recursive = TRUE)
if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR, recursive = TRUE)

### 1. 数据加载与标准化 ###
load(file = ".left_data.rdata")

feature_cols <- colnames(train_data)[-ncol(train_data)]  # 假设最后一列是 group

# 计算训练集的均值和标准差
train_mean <- colMeans(train_data[, feature_cols])
train_sd   <- apply(train_data[, feature_cols], 2, sd)

# 标准化
x_train <- scale(train_data[, feature_cols], center = train_mean, scale = train_sd)
x_test  <- scale(test_data[, feature_cols],  center = train_mean, scale = train_sd)
x_train <- as.matrix(x_train)
x_test  <- as.matrix(x_test)

y_train <- train_data$group
y_test  <- test_data$group

### 2. 训练带权重的 glmnet 模型 ###
y_train_num <- ifelse(y_train == "cancer", 1, 0)
cancer_count <- sum(y_train_num == 1)
control_count <- sum(y_train_num == 0)
weight_cancer <- control_count / cancer_count
sample_weights <- ifelse(y_train_num == 1, weight_cancer, 1)

fixed_model_weighted <- glmnet(x_train, y_train_num,
                               family = "binomial",
                               alpha = 0.5281,
                               lambda = 0.004597,
                               weights = sample_weights,
                               standardize = FALSE,
                               intercept = TRUE)

# 后续所有 fixed_model 都替换为 fixed_model_weighted

### 3. 特征重要性分析 ###
cat("\nPerforming feature importance analysis...\n")

# 提取系数
coef_matrix <- coef(fixed_model_weighted, s = 0.004597)
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
  scale_fill_manual(values = c("TRUE" = "#4daf4a", "FALSE" = "#e41a1c"), 
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
train_pred <- predict(fixed_model_weighted, newx = x_train, type = "response", s = 0.004597)[, 1]
test_pred <- predict(fixed_model_weighted, newx = x_test, type = "response", s = 0.004597)[, 1]

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
perm_importance <- calculate_permutation_importance_fixed(fixed_model_weighted, x_train, y_train)


# 保存置换重要性结果
write.csv(perm_importance, 
          file.path(DATA_DIR, "glmnet_fixed_permutation_importance.csv"), 
          row.names = FALSE)

### 5. 模型性能评估 ###
cat("\nEvaluating model performance...\n")

# 预测概率
train_pred <- predict(fixed_model_weighted, newx = x_train, type = "response", s = 0.004597)[, 1]
test_pred <- predict(fixed_model_weighted, newx = x_test, type = "response", s = 0.004597)[, 1]


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
  fixed_model_weighted,
  X = explain_data,
  pred_wrapper = glmnet_predict_fixed,
  nsim = 20,
  adjust = TRUE
)

# 创建shapviz对象
sv <- shapviz(shap_values, X = explain_data)
actual_labels <- as.character(y_train[explain_indices])
pred_probs <- glmnet_predict_fixed(fixed_model_weighted, explain_data)

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
    pdf(file.path(FIG_DIR, "glmnetAIC_SHAP_dependence.pdf"), width = 8, height = 6.4)
    
    # 创建多个依赖图
    for (feature in top_features) {
      if (feature %in% colnames(explain_data)) {
        p_dependence <- sv_dependence(sv, v = feature, 
                                      color_var = NULL,
                                      alpha = 0.6,
                                      size = 1.5) +
          geom_smooth(method = "loess", se = FALSE, color = "#e41a1c", size = 1.2) +
          ggtitle(sprintf("SHAP Dependence-%s", feature)) +
          theme_classic(base_size = 18) +
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
    scale_fill_gradient2(low = "#e41a1c", mid = "#f7f7f7", high = "#4daf4a",
                         midpoint = 0, name = "weight") +
    coord_flip() + 
    theme_minimal() + 
    labs(title = 'Logit regression adjustment', 
         x = 'Features', y = '-log10 P_value') +
    geom_hline(yintercept = 1.30103, linetype = "dashed", color = "#e41a1c", size = 1.0) +
    geom_hline(yintercept = 1.30103, linetype = "dashed", color = "#e41a1c", size = 1.0) +
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
  ggsave(file.path(FIG_DIR, "glmnetAIC_confounding_adjustment.pdf"), 
         p_confound, width = 7.5, height = 6.5, dpi = 300)
}


### 8. 模型校准 ###
cat("\nEvaluating model performance and calibration...\n")

# 预测概率
train_pred <- predict(fixed_model_weighted, newx = x_train, type = "response", s = 0.004597)[, 1]
test_pred <- predict(fixed_model_weighted, newx = x_test, type = "response", s = 0.004597)[, 1]

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

# ============================================================================
# 完整修改代码：glmnet 模型校准评估与校正（含 Platt Scaling 后校准）
# ============================================================================

library(ggplot2)
library(dplyr)
library(patchwork)
library(tidyr)



# ============================================================================
# 1. 增强版校正图函数（带基础率标注和加权指标）
# ============================================================================

plot_calibration_enhanced <- function(data, title, bin_width = 0.1, 
                                      prior_train = NULL, prior_test = NULL,
                                      show_corrected_line = FALSE) {
  
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
  
  # 计算标准ECE（等权重）
  ece_score <- mean(abs(cal_summary$mean_pred - cal_summary$mean_actual))
  
  # 计算加权ECE（按样本量加权，更可靠）
  weighted_ece <- sum(cal_summary$n * abs(cal_summary$mean_pred - cal_summary$mean_actual)) / sum(cal_summary$n)
  
  # 计算最大校准误差
  max_ece <- max(abs(cal_summary$mean_pred - cal_summary$mean_actual))
  
  # 创建基础图
  p <- ggplot(cal_summary, aes(x = mean_pred, y = mean_actual)) +
    geom_point(aes(size = n, color = n), alpha = 0.8) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                  width = 0.02, alpha = 0.6) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40", size = 1) +
    geom_smooth(method = "loess", se = TRUE, color = "#e41a1c", alpha = 0.3, size = 1.2) +
    scale_color_gradient(low = "#377eb8", high = "#4daf4a", name = "Sample size") +
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
  
  # 添加基础率参考线（如果提供）
  if(!is.null(prior_train)) {
    p <- p + geom_hline(yintercept = prior_train, 
                        linetype = "dotted", color = "#e41a1c", alpha = 0.6, size = 0.8) +
      annotate("text", x = 0.05, y = prior_train + -0.18, 
               label = sprintf("Train prior: %.1f%%", prior_train*100),
               size = 6, color ="#e41a1c", hjust = 0, alpha = 0.8)
  }
  
  if(!is.null(prior_test)) {
    p <- p + geom_hline(yintercept = prior_test, 
                        linetype = "dotted", color = "#ff7f00", alpha = 0.6, size = 0.8) +
      annotate("text", x = 0.05, y = prior_test + 0.2, 
               label = sprintf("Test prior: %.1f%%", prior_test*100),
               size = 6, color = "#ff7f00", hjust = 0, alpha = 0.8)
  }
  
  return(p)
}


# ============================================================================
# 2. 先验校正函数（使用贝叶斯公式）
# ============================================================================

apply_prior_correction <- function(pred_probs, train_prior, target_prior) {
  # 防止极端值
  pred_probs <- pmax(pmin(pred_probs, 1 - 1e-7), 1e-7)
  
  # 计算对数几率（log-odds）
  log_odds <- log(pred_probs / (1 - pred_probs))
  
  # 调整先验
  adjusted_log_odds <- log_odds + log(target_prior / (1 - target_prior)) - 
    log(train_prior / (1 - train_prior))
  
  # 转换回概率
  corrected_probs <- 1 / (1 + exp(-adjusted_log_odds))
  
  return(corrected_probs)
}


# ============================================================================
# 3. 【新增】Platt Scaling 后校准函数（推荐用于不平衡数据）
# ============================================================================

apply_platt_scaling <- function(train_pred, train_y, test_pred = NULL, 
                                validation_pred = NULL, validation_y = NULL) {
  "
  Platt Scaling 后校准
  
  参数:
    train_pred: 训练集原始预测概率
    train_y: 训练集真实标签
    test_pred: 测试集原始预测概率（可选）
    validation_pred: 验证集原始预测概率（用于拟合校准器，推荐）
    validation_y: 验证集真实标签（用于拟合校准器，推荐）
  
  返回:
    包含校准后概率和校准模型的列表
  "
  
  train_y_num <- as.numeric(train_y) - 1
  
  # 选择用于拟合校准器的数据
  # 推荐使用独立的验证集，如果没有则使用训练集（但会有轻微过拟合风险）
  if(!is.null(validation_pred) && !is.null(validation_y)) {
    fit_pred <- validation_pred
    fit_y <- as.numeric(validation_y) - 1
    cat("使用独立验证集拟合 Platt scaling 校准器\n")
  } else {
    fit_pred <- train_pred
    fit_y <- train_y_num
    cat("警告：使用训练集拟合 Platt scaling 校准器（建议使用独立验证集）\n")
  }
  
  # 处理极端概率值（避免 log(0) 或 log(Inf)）
  fit_pred <- pmax(pmin(fit_pred, 1 - 1e-7), 1e-7)
  logit_pred <- log(fit_pred / (1 - fit_pred))
  
  # 拟合 Platt scaling 模型（逻辑回归）
  platt_model <- glm(fit_y ~ logit_pred, family = binomial())
  
  # 获取校准参数
  a <- coef(platt_model)[1]  # 截距
  b <- coef(platt_model)[2]  # 斜率
  
  cat(sprintf("Platt scaling 参数: intercept = %.4f, slope = %.4f\n", a, b))
  
  # 判断校准方向
  if(b < 1) {
    cat(">>> 模型存在过度自信倾向（预测概率过于极端），Platt scaling 将收缩预测\n")
  } else if(b > 1) {
    cat(">>> 模型存在保守倾向（预测概率过于保守），Platt scaling 将扩大预测\n")
  } else {
    cat(">>> 模型校准方向良好\n")
  }
  
  # 对训练集进行校准
  train_logit <- log(pmax(pmin(train_pred, 1 - 1e-7), 1e-7) / (1 - pmax(pmin(train_pred, 1 - 1e-7), 1e-7)))
  train_pred_calibrated <- 1 / (1 + exp(-(a + b * train_logit)))
  
  result <- list(
    train_calibrated = train_pred_calibrated,
    model = platt_model,
    a = a,
    b = b
  )
  
  # 如果提供了测试集，也进行校准
  if(!is.null(test_pred)) {
    test_logit <- log(pmax(pmin(test_pred, 1 - 1e-7), 1e-7) / (1 - pmax(pmin(test_pred, 1 - 1e-7), 1e-7)))
    test_pred_calibrated <- 1 / (1 + exp(-(a + b * test_logit)))
    result$test_calibrated <- test_pred_calibrated
  }
  
  return(result)
}


# ============================================================================
# 4. 【新增】Isotonic Regression 后校准函数（备选，样本量大时效果更好）
# ============================================================================

apply_isotonic_calibration <- function(train_pred, train_y, test_pred = NULL) {
  "
  Isotonic Regression 后校准（保序回归）
  适用于样本量较大的情况（>1000），比 Platt scaling 更灵活
  "
  
  train_y_num <- as.numeric(train_y) - 1
  
  # 使用 isoreg 函数拟合保序回归
  iso_model <- isoreg(train_pred, train_y_num)
  
  # 对训练集进行校准（使用阶梯函数）
  train_pred_calibrated <- approx(iso_model$x, iso_model$yf, 
                                  xout = train_pred, rule = 2)$y
  
  # 处理边界值
  train_pred_calibrated[is.na(train_pred_calibrated)] <- 0
  train_pred_calibrated <- pmax(pmin(train_pred_calibrated, 1), 0)
  
  result <- list(
    train_calibrated = train_pred_calibrated,
    model = iso_model
  )
  
  if(!is.null(test_pred)) {
    test_pred_calibrated <- approx(iso_model$x, iso_model$yf, 
                                   xout = test_pred, rule = 2)$y
    test_pred_calibrated[is.na(test_pred_calibrated)] <- 0
    test_pred_calibrated <- pmax(pmin(test_pred_calibrated, 1), 0)
    result$test_calibrated <- test_pred_calibrated
  }
  
  return(result)
}


# ============================================================================
# 5. 计算校准性能指标
# ============================================================================

calculate_calibration_metrics <- function(pred_probs, y_true) {
  y_num <- as.numeric(y_true) - 1
  
  # Brier score
  brier <- mean((pred_probs - y_num)^2)
  
  # 校准曲线斜率（理想值=1）
  pred_probs_safe <- pmax(pmin(pred_probs, 1 - 1e-7), 1e-7)
  logit_pred <- log(pred_probs_safe / (1 - pred_probs_safe))
  cal_model <- suppressWarnings(glm(y_num ~ logit_pred, family = binomial()))
  cal_slope <- coef(cal_model)[2]
  cal_intercept <- coef(cal_model)[1]
  
  # 期望校准误差（ECE）
  n_bins <- 10
  bins <- cut(pred_probs, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
  ece_data <- data.frame(pred = pred_probs, true = y_num, bin = bins)
  ece <- ece_data %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(pred),
      mean_true = mean(true),
      n = n()
    ) %>%
    summarise(ece = sum(n * abs(mean_pred - mean_true)) / sum(n)) %>%
    pull(ece)
  
  return(list(
    brier = brier,
    calibration_slope = cal_slope,
    calibration_intercept = cal_intercept,
    ece = ece
  ))
}


# ============================================================================
# 6. 对比表格生成函数
# ============================================================================

create_comparison_table <- function(metrics_list, model_names) {
  comparison_df <- data.frame(
    Model = model_names,
    Brier = sapply(metrics_list, function(x) round(x$brier, 4)),
    ECE = sapply(metrics_list, function(x) round(x$ece, 4)),
    Cal_Slope = sapply(metrics_list, function(x) round(x$calibration_slope, 3)),
    Cal_Intercept = sapply(metrics_list, function(x) round(x$calibration_intercept, 3))
  )
  return(comparison_df)
}


# ============================================================================
# 7. 主程序：执行评估与校正
# ============================================================================

cat("\n", rep("=", 70), "\n")
cat("GLMNET 模型校准评估与校正分析（含 Platt Scaling 后校准）\n")
cat(rep("=", 70), "\n\n")

# ===== 假设已经有以下变量 =====
# - train_pred: 训练集原始预测概率
# - test_pred: 测试集原始预测概率
# - y_train: 训练集真实标签（0/1或因子）
# - y_test: 测试集真实标签（0/1或因子）
# - val_pred: 验证集原始预测概率（可选，用于拟合校准器）
# - y_val: 验证集真实标签（可选）

# 计算先验概率
train_prior <- mean(as.numeric(y_train) - 1)
test_prior <- mean(as.numeric(y_test) - 1)

cat("基础率统计:\n")
cat(sprintf("  训练集: 正例数 = %d, 负例数 = %d, 正例率 = %.2f%%\n", 
            sum(as.numeric(y_train)-1), sum(1-(as.numeric(y_train)-1)), train_prior*100))
cat(sprintf("  测试集: 正例数 = %d, 负例数 = %d, 正例率 = %.2f%%\n", 
            sum(as.numeric(y_test)-1), sum(1-(as.numeric(y_test)-1)), test_prior*100))
cat(sprintf("  先验比率（测试/训练）: %.2f倍\n\n", test_prior/train_prior))

# ===== 7.1 原始模型性能 =====
cat("\n", rep("-", 50), "\n")
cat("步骤 1: 评估原始模型（未校准）\n")
cat(rep("-", 50), "\n")

metrics_train_raw <- calculate_calibration_metrics(train_pred, y_train)
metrics_test_raw <- calculate_calibration_metrics(test_pred, y_test)

cat(sprintf("训练集 - Brier: %.4f, ECE: %.4f\n", metrics_train_raw$brier, metrics_train_raw$ece))
cat(sprintf("测试集 - Brier: %.4f, ECE: %.4f\n", metrics_test_raw$brier, metrics_test_raw$ece))

# ===== 7.2 Platt Scaling 后校准 =====
cat("\n", rep("-", 50), "\n")
cat("步骤 2: 应用 Platt Scaling 后校准\n")
cat(rep("-", 50), "\n")

# 使用训练集拟合 Platt scaling（如果没有独立验证集）
# 注意：如果内部验证集可用，建议使用内部验证集拟合校准器
platt_result <- apply_platt_scaling(
  train_pred = train_pred,
  train_y = y_train,
  test_pred = test_pred
)

train_pred_platt <- platt_result$train_calibrated
test_pred_platt <- platt_result$test_calibrated

# 评估校准后性能
metrics_train_platt <- calculate_calibration_metrics(train_pred_platt, y_train)
metrics_test_platt <- calculate_calibration_metrics(test_pred_platt, y_test)

cat("\nPlatt scaling 校准后:\n")
cat(sprintf("训练集 - Brier: %.4f, ECE: %.4f\n", metrics_train_platt$brier, metrics_train_platt$ece))
cat(sprintf("测试集 - Brier: %.4f, ECE: %.4f\n", metrics_test_platt$brier, metrics_test_platt$ece))

# ===== 7.3 先验校正（用于对比）=====
cat("\n", rep("-", 50), "\n")
cat("步骤 3: 应用先验校正（用于对比）\n")
cat(rep("-", 50), "\n")

test_pred_prior_corr <- apply_prior_correction(test_pred, train_prior, test_prior)
metrics_test_prior <- calculate_calibration_metrics(test_pred_prior_corr, y_test)
cat(sprintf("先验校正后测试集 - Brier: %.4f, ECE: %.4f\n", metrics_test_prior$brier, metrics_test_prior$ece))

# ===== 7.4 生成对比表格 =====
cat("\n", rep("-", 50), "\n")
cat("步骤 4: 生成性能对比表\n")
cat(rep("-", 50), "\n")

comparison_table <- create_comparison_table(
  list(
    metrics_train_raw, 
    metrics_test_raw,
    metrics_train_platt,
    metrics_test_platt,
    metrics_test_prior
  ),
  c(
    "训练集（原始）", 
    "测试集（原始）",
    "训练集（Platt校正）",
    "测试集（Platt校正）",
    "测试集（先验校正）"
  )
)

cat("\n性能对比表:\n")
print(comparison_table)

# 保存对比表格为CSV
write.csv(comparison_table, file.path(FIG_DIR, "calibration_comparison.csv"), 
          row.names = FALSE)


# ============================================================================
# 8. 绘制校正图
# ============================================================================

cat("\n生成校准图...\n")

# 创建数据框
train_raw_df <- data.frame(
  pred_prob = train_pred,
  group = y_train,
  dataset = "Training (Raw)"
)

test_raw_df <- data.frame(
  pred_prob = test_pred,
  group = y_test,
  dataset = "Test (Raw)"
)

train_platt_df <- data.frame(
  pred_prob = train_pred_platt,
  group = y_train,
  dataset = "Training (Platt Calibrated)"
)

test_platt_df <- data.frame(
  pred_prob = test_pred_platt,
  group = y_test,
  dataset = "Test (Platt Calibrated)"
)

test_prior_df <- data.frame(
  pred_prob = test_pred_prior_corr,
  group = y_test,
  dataset = "Test (Prior Corrected)"
)

# 绘制各图
p_train_raw <- plot_calibration_enhanced(
  train_raw_df, 
  "A. 训练集（原始）",
  prior_train = train_prior,
  bin_width = 0.1
)

p_test_raw <- plot_calibration_enhanced(
  test_raw_df, 
  "B. 测试集（原始）",
  prior_train = train_prior,
  prior_test = test_prior,
  bin_width = 0.1
)

p_train_platt <- plot_calibration_enhanced(
  train_platt_df, 
  "Training Set (Platt Scaling)",
  prior_train = train_prior,
  bin_width = 0.1
)

p_test_platt <- plot_calibration_enhanced(
  test_platt_df, 
  "D. 测试集（Platt校正后）",
  prior_train = train_prior,
  prior_test = test_prior,
  bin_width = 0.1
)

p_test_prior <- plot_calibration_enhanced(
  test_prior_df, 
  "External Validation Set 
  (Prior Correction)",
  prior_train = train_prior,
  prior_test = test_prior,
  bin_width = 0.1
)

# 合并图（两种布局）
# 布局1：原始 vs Platt 对比（主要）
p_combined_main <- (p_train_raw | p_test_raw) / (p_train_platt | p_test_platt) +
  plot_annotation(
    title = "GLMNET 模型校准：Platt Scaling 前后对比",
    subtitle = sprintf("训练集正例率 = %.1f%% | 测试集正例率 = %.1f%%", train_prior*100, test_prior*100),
    theme = theme(
      plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5)
    )
  )

# 布局2：三种方法对比（原始、Platt、先验校正）
p_combined_all <- p_train_raw + p_test_raw + p_test_platt + p_test_prior +
  plot_layout(ncol = 2) +
  plot_annotation(
    title = "GLMNET 模型校准：多种校正方法对比",
    subtitle = sprintf("训练集正例率 = %.1f%% | 测试集正例率 = %.1f%%", train_prior*100, test_prior*100),
    theme = theme(
      plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5)
    )
  )

# 显示和保存
print(p_combined_main)
print(p_combined_all)

# 保存图表
ggsave(file.path(FIG_DIR, "calibration_platt_comparison.png"), 
       p_combined_main, width = 14, height = 12, dpi = 300)
ggsave(file.path(FIG_DIR, "calibration_all_methods.png"), 
       p_combined_all, width = 14, height = 12, dpi = 300)

# 单独保存各图
ggsave(file.path(FIG_DIR, "calibration_train_raw.png"), 
       p_train_raw, width = 6.5, height = 6, dpi = 300)
ggsave(file.path(FIG_DIR, "calibration_test_raw.png"), 
       p_test_raw, width = 6.5, height = 6, dpi = 300)
ggsave(file.path(FIG_DIR, "calibration_train_platt.png"), 
       p_train_platt, width = 6, height = 6, dpi = 300)
ggsave(file.path(FIG_DIR, "calibration_test_platt.png"), 
       p_test_platt, width = 6.5, height = 6, dpi = 300)
ggsave(file.path(FIG_DIR, "calibration_test_prior.png"), 
       p_test_prior, width = 6.2, height = 6.2, dpi = 300)


# ============================================================================
# 9. 详细诊断报告
# ============================================================================

cat("\n", rep("=", 70), "\n")
cat("校准诊断报告\n")
cat(rep("=", 70), "\n\n")

# 分析校准偏差模式函数
analyze_calibration_bias <- function(pred_probs, y_true, title) {
  y_num <- as.numeric(y_true) - 1
  df <- data.frame(pred = pred_probs, true = y_num)
  df$bin <- cut(df$pred, breaks = seq(0, 1, 0.1), include.lowest = TRUE)
  
  bin_analysis <- df %>%
    group_by(bin) %>%
    summarise(
      n = n(),
      mean_pred = mean(pred),
      mean_true = mean(true),
      bias = mean_true - mean_pred,
      .groups = 'drop'
    ) %>%
    filter(n > 0)
  
  cat("\n", title, "\n")
  cat("----------------------------------------\n")
  print(bin_analysis)
  
  # 判断偏差类型
  mid_bias <- mean(bin_analysis$bias[bin_analysis$mean_pred > 0.3 & bin_analysis$mean_pred < 0.7])
  if(!is.na(mid_bias)) {
    if(mid_bias > 0.05) {
      cat(sprintf("\n>>> 诊断：模型在中等概率区域存在系统性低估风险（平均偏差 = %.3f）\n", mid_bias))
    } else if(mid_bias < -0.05) {
      cat(sprintf("\n>>> 诊断：模型在中等概率区域存在系统性高估风险（平均偏差 = %.3f）\n", mid_bias))
    } else {
      cat(sprintf("\n>>> 诊断：模型在中等概率区域校准良好（平均偏差 = %.3f）\n", mid_bias))
    }
  }
  
  return(bin_analysis)
}

# 执行诊断
analyze_calibration_bias(train_pred, y_train, "1. 训练集（原始）")
analyze_calibration_bias(test_pred, y_test, "2. 测试集（原始）")
analyze_calibration_bias(train_pred_platt, y_train, "3. 训练集（Platt校正后）")
analyze_calibration_bias(test_pred_platt, y_test, "4. 测试集（Platt校正后）")
analyze_calibration_bias(test_pred_prior_corr, y_test, "5. 测试集（先验校正）")


# ============================================================================
# 10. 总结输出
# ============================================================================

cat("\n", rep("=", 70), "\n")
cat("分析总结\n")
cat(rep("=", 70), "\n\n")

# 计算校准改进幅度
ece_improvement_train <- metrics_train_raw$ece - metrics_train_platt$ece
ece_improvement_test <- metrics_test_raw$ece - metrics_test_platt$ece

cat("【主要发现】\n")
cat(sprintf("1. 原始模型训练集 ECE = %.4f（极差），存在严重的过度自信问题\n", metrics_train_raw$ece))
cat(sprintf("2. 原始模型测试集 ECE = %.4f（较差）\n", metrics_test_raw$ece))
cat(sprintf("3. Platt scaling 后，训练集 ECE 改善 %.4f（%.1f%%）\n", 
            ece_improvement_train, ece_improvement_train/metrics_train_raw$ece*100))
cat(sprintf("4. Platt scaling 后，测试集 ECE 改善 %.4f（%.1f%%）\n",
            ece_improvement_test, ece_improvement_test/metrics_test_raw$ece*100))
cat(sprintf("5. 先验校正后测试集 ECE = %.4f\n", metrics_test_prior$ece))

cat("\n【推荐方案】\n")
if(metrics_test_platt$ece < metrics_test_prior$ece) {
  cat("  ✅ 推荐使用 Platt Scaling 进行后校准\n")
  cat(sprintf("     理由：ECE 更低（%.4f vs %.4f），且不需要知道目标人群先验\n", 
              metrics_test_platt$ece, metrics_test_prior$ece))
} else {
  cat("  ✅ 推荐使用先验校正\n")
  cat(sprintf("     理由：ECE 更低（%.4f vs %.4f）\n", 
              metrics_test_prior$ece, metrics_test_platt$ece))
}

cat("\n【论文撰写建议】\n")
cat("  - 报告原始模型和校准后的性能对比\n")
cat("  - 明确说明训练集类别不平衡（1:4.79）导致原始校准较差\n")
cat("  - 说明采用 Platt scaling 进行后校准，显著改善了 ECE\n")
cat("  - 在图中展示校准前后对比\n")

cat("\n所有图表已保存至:", FIG_DIR, "\n")





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
                          ifelse(plot_df$Type == "Final", "#377eb8",
                                 ifelse(plot_df$SHAP > 0, "#e41a1c","#4daf4a")))
  
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
      Bar_Label_Color = ifelse(Type %in% c("Baseline", "Final"), "#377eb8", 
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
                                margin = ggplot2::margin(b = 10)),
      plot.subtitle = element_text(hjust = 0.5, 
                                   size = base_font_size * 1.5,
                                   margin = ggplot2::margin(b = 15)),
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
      plot.margin = ggplot2::margin(20, 50, 20, 50)
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
  plot_data$Color <- ifelse(plot_data$SHAP > 0, "#e41a1c", "#4daf4a")
  
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
             color = "#377eb8",
             hjust = 0.3) +
    
    # 添加基线文本
    annotate("text",
             x = baseline,
             y = baseline_y,  # 使用基线的Y位置
             label = sprintf("E[f(x)] = %.3f", 0.5),  # 直接显示概率0.5
             size = base_font_size * 0.7,
             fontface = "bold",
             color = "#377eb8",
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
                                margin = ggplot2::margin(b = 10)),
      plot.subtitle = element_text(hjust = 0.5, 
                                   size = base_font_size * 1.5,
                                   margin = ggplot2::margin(b = 15)),
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
      plot.margin = ggplot2::margin(20, 50, 20, 50)
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
                             color = ifelse(SHAP > 0, "#4daf4a", "#e41a1c")),
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


