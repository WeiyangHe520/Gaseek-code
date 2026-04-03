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

### 9. 绘制模型预测分数与临床特征相关性热图（三角热图）###
cat("\nCreating triangular correlation heatmap...\n")

# 首先获取模型在训练集和测试集的预测分数
train_pred <- predict(fixed_model, newx = x_train, type = "response", s = 0.004597)[, 1]
test_pred <- predict(fixed_model, newx = x_test, type = "response", s = 0.004597)[, 1]

# 合并预测分数和临床特征
train_data_with_pred <- train_data
train_data_with_pred$pred_prob <- train_pred

# 使用你实际有的临床特征（根据你的热图）
clinical_features <- c("pred_prob", "age", "GLB", "ALB", "IBIL", "HGB", "gender")

# 检查哪些特征在数据中存在
available_features <- clinical_features[clinical_features %in% colnames(train_data_with_pred)]
cat("Available features for correlation analysis:", paste(available_features, collapse = ", "), "\n")

# 使用训练集数据计算相关性
if (length(available_features) > 1) {
  # 提取相关数据
  cor_data <- train_data_with_pred[, available_features]
  
  # 确保数值型
  for (col in available_features) {
    if (is.factor(cor_data[[col]])) {
      cor_data[[col]] <- as.numeric(cor_data[[col]])
    } else if (is.character(cor_data[[col]])) {
      cor_data[[col]] <- as.numeric(factor(cor_data[[col]]))
    }
  }
  
  # 计算Spearman相关性矩阵
  cor_matrix <- cor(cor_data, method = "spearman", use = "complete.obs")
  
  # 重命名行列以增加可读性
  feature_names <- c(
    "pred_prob",
    "age",
    "GLB",
    "ALB",
    "IBIL",
    "HGB",
    "gender"
  )[1:length(available_features)]
  
  rownames(cor_matrix) <- colnames(cor_matrix) <- feature_names
  
  # 方法1：使用corrplot直接绘制三角热图
  tiff(file.path(FIG_DIR, "glmnet_fixed_correlation_triangle_corrplot.tiff"), 
    width = 6, height = 6, units = "in", res = 300, compression = "lzw")
  
  # 绘制下三角热图
  corrplot(cor_matrix,
           method = "square",
           type = "lower",           # 只显示下三角
           order = "original",       # 保持原始顺序
           tl.col = "black",
           tl.srt = 45,              # 标签旋转角度
           tl.cex = 1.4,             # 标签字体大小
           cl.cex = 1.4,             # 图例字体大小
           addCoef.col = "black",    # 系数颜色
           number.cex = 1.2,         # 系数字体大小
           diag = TRUE,              # 显示对角线
           col = COL2("BrBG"),       # 颜色方案
           mar = c(0, 0, 2, 0),      # 边距
           title = "")
  
  dev.off()
  
  # 方法2：使用ggplot2绘制自定义三角热图
  library(reshape2)
  
  # 将相关矩阵转换为长格式
  cor_melted <- melt(cor_matrix)
  colnames(cor_melted) <- c("Var1", "Var2", "value")
  
  # 只保留下三角部分（包括对角线）
  cor_melted$row_idx <- match(cor_melted$Var1, feature_names)
  cor_melted$col_idx <- match(cor_melted$Var2, feature_names)
  cor_melted_tri <- cor_melted[cor_melted$row_idx >= cor_melted$col_idx, ]
  
  # 设置因子顺序（保持与矩阵一致）
  cor_melted_tri$Var1 <- factor(cor_melted_tri$Var1, levels = feature_names)
  cor_melted_tri$Var2 <- factor(cor_melted_tri$Var2, levels = rev(feature_names))  # 反转Y轴
  
  # 创建三角热图
  p_tri_heatmap <- ggplot(cor_melted_tri, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white", size = 1) +
    # 对角线上的值为1，显示为空或"1"
    geom_text(data = subset(cor_melted_tri, as.character(Var1) == as.character(Var2)),
              aes(label = "1"), color = "black", size = 5, fontface = "bold") +
    # 非对角线显示相关系数
    geom_text(data = subset(cor_melted_tri, as.character(Var1) != as.character(Var2)),
              aes(label = sprintf("%.2f", value)), 
              color = "black", size = 5, fontface = "bold") +
    scale_fill_gradient2(
      low = "#377eb8",   # 蓝色表示负相关
      mid = "white",     # 白色表示无相关
      high = "#e41a1c",  # 红色表示正相关
      midpoint = 0,
      limits = c(-1, 1),
      name = "Spearman\nCorrelation"
    ) +
    labs(
      title = "",
      x = "",
      y = ""
    ) +
    theme_minimal(base_size = 18) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 18, face = "bold"),
      axis.text.y = element_text(angle = 45,size = 18, face = "bold", hjust = 1),
      axis.title = element_blank(),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 14),
      legend.position = "right",
      panel.grid = element_blank(),
      plot.margin = margin(1, 1, 1, 1, "cm")
    ) +
    coord_fixed(ratio = 1)
  
  print(p_tri_heatmap)
  
  # 保存ggplot2三角热图
  ggsave(file.path(FIG_DIR, "glmnet_fixed_correlation_triangle_ggplot.pdf"), 
         p_tri_heatmap, width = 10, height = 10)
  ggsave(file.path(FIG_DIR, "glmnet_fixed_correlation_triangle_ggplot.tiff"), 
         p_tri_heatmap, width = 10, height = 10, dpi = 300)
  
  # 方法3：绘制更简洁的版本（只显示非对角线）
  p_tri_heatmap_simple <- ggplot(cor_melted_tri, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white", size = 1) +
    # 只在下三角显示系数（不包括对角线）
    geom_text(data = subset(cor_melted_tri, as.character(Var1) != as.character(Var2)),
              aes(label = sprintf("%.2f", value)), 
              color = "black", size = 5, fontface = "bold") +
    scale_fill_gradient2(
      low = "#377eb8",
      mid = "white",
      high = "#e41a1c",
      midpoint = 0,
      limits = c(-1, 1),
      name = "Correlation"
    ) +
    labs(
      x = "",
      y = ""
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
      axis.text.y = element_text(size = 12),
      panel.grid = element_blank(),
      legend.position = "right"
    ) +
    coord_fixed()
  
  # 保存简洁版
  ggsave(file.path(FIG_DIR, "glmnet_fixed_correlation_triangle_simple.pdf"), 
         p_tri_heatmap_simple, width = 8, height = 8)
  
  # 保存相关性矩阵数据
  write.csv(cor_matrix, 
            file.path(DATA_DIR, "glmnet_fixed_correlation_matrix.csv"), 
            row.names = TRUE)
  
  # 保存下三角矩阵
  lower_tri_matrix <- cor_matrix
  lower_tri_matrix[upper.tri(lower_tri_matrix)] <- NA  # 将上三角设为NA
  
  write.csv(lower_tri_matrix, 
            file.path(DATA_DIR, "glmnet_fixed_correlation_matrix_lower_triangle.csv"), 
            row.names = TRUE, na = "")
  
  cat("Triangular correlation heatmaps saved.\n")
  
} else {
  cat("Warning: Not enough features available for correlation analysis.\n")
  cat("Available features:", paste(available_features, collapse = ", "), "\n")
}