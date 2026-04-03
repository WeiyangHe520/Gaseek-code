###############################################
## 变量共线性分析（完整调试版）
###############################################
rm(list = ls())
set.seed(278)
FIG_DIR <- "figures共线/"
DATA_DIR <- "data共线/"
dir.create(FIG_DIR, showWarnings = FALSE)
dir.create(DATA_DIR, showWarnings = FALSE)

# 加载必要的包
library(caret)
library(tidyverse)
library(viridis)
library(ggprism)
library(pROC)
library(ggplot2)
load(file = ".left_data.rdata")

cat("\n=== 变量共线性分析 ===\n")

# 1. 先检查原始数据结构
cat("\n1. 检查原始数据结构...\n")
cat("训练数据维度:", dim(train_data), "\n")
cat("测试数据维度:", dim(test_data), "\n")
cat("训练数据列名:\n")
print(colnames(train_data))
cat("测试数据列名:\n")
print(colnames(test_data))

# 2. 查看数据前几行
cat("\n训练数据前3行:\n")
print(head(train_data, 3))

# 3. 确定class列的位置
cat("\n确定class列的位置...\n")
class_col_train <- which(colnames(train_data) == "group")
class_col_test <- which(colnames(test_data) == "group")

cat("训练数据group列位置:", class_col_train, "\n")
cat("测试数据class列位置:", class_col_test, "\n")

if (length(class_col_train) == 0) {
  stop("错误：在训练数据中未找到class列！")
}
if (length(class_col_test) == 0) {
  stop("错误：在测试数据中未找到class列！")
}

# 4. 提取特征变量（正确的做法）
cat("\n提取特征变量...\n")
# 方法1：使用所有非class列
features_train <- train_data[, -class_col_train, drop = FALSE]
features_test <- test_data[, -class_col_test, drop = FALSE]

# 方法2：或者使用列名排除
# features_train <- subset(train_data, select = -class)
# features_test <- subset(test_data, select = -class)

cat("提取后的特征数据维度 - 训练集:", dim(features_train), "\n")
cat("提取后的特征数据维度 - 测试集:", dim(features_test), "\n")

if (ncol(features_train) == 0) {
  stop("错误：提取的特征变量为空！")
}

# 5. 检查提取的特征数据
cat("\n特征数据前3行:\n")
print(head(features_train, 3))

cat("\n特征数据结构:\n")
str(features_train)

# 6. 加载必要的包
cat("\n加载共线性分析所需包...\n")
required_packages <- c("car", "corrplot", "Hmisc", "ggplot2")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  } else {
    library(pkg, character.only = TRUE)
  }
}

# 7. 数据预处理
cat("\n=== 数据预处理 ===\n")

# 7.1 检查是否存在常数列
cat("检查常数变量...\n")
constant_cols <- sapply(features_train, function(x) {
  if (is.numeric(x)) {
    length(unique(x)) == 1
  } else {
    FALSE
  }
})

if (any(constant_cols)) {
  cat("发现常数变量:", names(features_train)[constant_cols], "\n")
  cat("已移除常数变量\n")
  features_train <- features_train[, !constant_cols, drop = FALSE]
  features_test <- features_test[, !constant_cols, drop = FALSE]
}

# 7.2 检查是否有缺失值
cat("检查缺失值...\n")
missing_count <- sum(is.na(features_train))
if (missing_count > 0) {
  cat("发现缺失值，数量:", missing_count, "\n")
  cat("使用列均值填充数值变量...\n")
  
  for (col in colnames(features_train)) {
    if (is.numeric(features_train[[col]])) {
      if (any(is.na(features_train[[col]]))) {
        col_mean <- mean(features_train[[col]], na.rm = TRUE)
        features_train[[col]][is.na(features_train[[col]])] <- col_mean
      }
    }
  }
}

# 8. 识别数值型变量
cat("\n=== 识别变量类型 ===\n")

# 方法1：使用sapply识别数值型变量
numeric_cols <- sapply(features_train, function(x) {
  is.numeric(x) || is.integer(x)
})

# 检查numeric_cols的类型
cat("numeric_cols的类型:", class(numeric_cols), "\n")
cat("numeric_cols的长度:", length(numeric_cols), "\n")

if (is.list(numeric_cols)) {
  cat("警告：numeric_cols是list类型，转换为向量...\n")
  numeric_cols <- unlist(numeric_cols)
}

cat("数值型变量数量:", sum(numeric_cols), "\n")
cat("非数值型变量数量:", sum(!numeric_cols), "\n")

if (sum(numeric_cols) == 0) {
  cat("\n⚠️ 警告：没有找到数值型变量！\n")
  cat("检查变量类型:\n")
  var_types <- sapply(features_train, class)
  print(var_types)
  
  # 尝试转换因子变量为数值
  cat("\n尝试转换因子变量为数值...\n")
  for (col in colnames(features_train)) {
    if (is.factor(features_train[[col]])) {
      cat("转换因子变量:", col, "\n")
      features_train[[col]] <- as.numeric(as.character(features_train[[col]]))
      features_test[[col]] <- as.numeric(as.character(features_test[[col]]))
    }
  }
  
  # 重新识别数值型变量
  numeric_cols <- sapply(features_train, function(x) is.numeric(x))
  cat("转换后数值型变量数量:", sum(numeric_cols), "\n")
}

if (sum(numeric_cols) > 0) {
  # 提取数值型变量
  numeric_features <- features_train[, numeric_cols, drop = FALSE]
  cat("\n数值型特征数据维度:", dim(numeric_features), "\n")
  
  # 显示前几个数值型变量名
  cat("\n前10个数值型变量名:\n")
  print(head(colnames(numeric_features), 10))
  
  # 9. 方差膨胀因子（VIF）分析
  cat("\n=== 方差膨胀因子（VIF）分析 ===\n")
  
  if (ncol(numeric_features) >= 2) {
    # 限制变量数量，避免计算问题
    max_vars_vif <- min(30, ncol(numeric_features))
    
    if (ncol(numeric_features) > max_vars_vif) {
      cat("变量数量较多，使用前", max_vars_vif, "个变量进行VIF分析\n")
      vif_data <- numeric_features[, 1:max_vars_vif]
    } else {
      vif_data <- numeric_features
    }
    
    # 确保有足够的样本
    if (nrow(vif_data) > ncol(vif_data) + 5) {
      tryCatch({
        # 创建虚拟响应变量
        set.seed(278)
        dummy_response <- rnorm(nrow(vif_data))
        
        # 创建模型
        vif_model <- lm(dummy_response ~ ., data = vif_data)
        vif_values <- vif(vif_model)
        
        # 创建VIF数据框
        vif_df <- data.frame(
          Variable = names(vif_values),
          VIF = round(vif_values, 3),
          stringsAsFactors = FALSE
        )
        vif_df <- vif_df[order(-vif_df$VIF), ]
        
        cat("\nVIF分析结果（前20个变量）:\n")
        print(head(vif_df, 20))
        
        # 识别高VIF变量
        high_vif <- vif_df[vif_df$VIF >= 10, ]
        if (nrow(high_vif) > 0) {
          cat("\n⚠️ 警告：发现高VIF变量（VIF ≥ 10）:\n")
          print(high_vif)
        } else {
          cat("\n✓ 没有发现VIF ≥ 10的变量\n")
        }
        
        # 可视化VIF结果
        if (nrow(vif_df) > 0) {
          png(file.path(FIG_DIR, "vif_analysis.png"), width = 12, height = 8, units = "in", res = 300)
          par(mar = c(10, 4, 4, 2) + 0.1)  # 增加底部边距
          
          # 只显示前30个变量以避免图形过于拥挤
          n_show <- min(30, nrow(vif_df))
          vif_show <- head(vif_df, n_show)
          
          bar_colors <- ifelse(vif_show$VIF >= 10, "red", 
                               ifelse(vif_show$VIF >= 5, "orange", "skyblue"))
          
          barplot(vif_show$VIF, names.arg = vif_show$Variable, 
                  las = 2, col = bar_colors,
                  ylab = "VIF Value", main = "方差膨胀因子（VIF）分析",
                  ylim = c(0, max(vif_show$VIF) * 1.1))
          
          abline(h = 5, lty = 2, col = "orange", lwd = 2)
          abline(h = 10, lty = 2, col = "red", lwd = 2)
          
          legend("topright", 
                 legend = c("VIF < 5", "5 ≤ VIF < 10", "VIF ≥ 10"),
                 fill = c("skyblue", "orange", "red"),
                 cex = 0.8)
          
          dev.off()
          cat("\nVIF分析图已保存至:", file.path(FIG_DIR, "vif_analysis.png"), "\n")
        }
        
      }, error = function(e) {
        cat("VIF计算错误:", e$message, "\n")
        vif_df <- NULL
      })
    } else {
      cat("样本量不足，跳过VIF分析\n")
      vif_df <- NULL
    }
  } else {
    cat("数值型变量数量不足（<2），跳过VIF分析\n")
    vif_df <- NULL
  }
  
  # 10. Pearson相关性分析
  cat("\n=== Pearson相关性分析 ===\n")
  
  if (ncol(numeric_features) >= 2) {
    # 限制变量数量
    max_vars_corr <- min(50, ncol(numeric_features))
    
    if (ncol(numeric_features) > max_vars_corr) {
      cat("变量数量较多，使用前", max_vars_corr, "个变量进行相关性分析\n")
      corr_data <- numeric_features[, 1:max_vars_corr]
    } else {
      corr_data <- numeric_features
    }
    
    # 计算相关系数矩阵
    cor_matrix <- cor(corr_data, use = "pairwise.complete.obs")
    
    cat("\n相关系数矩阵维度:", dim(cor_matrix), "\n")
    
    # 找出高度相关的变量对
    cat("\n寻找高度相关的变量对（|r| ≥ 0.8）...\n")
    
    high_corr_indices <- which(abs(cor_matrix) >= 0.8 & upper.tri(cor_matrix), arr.ind = TRUE)
    
    if (length(high_corr_indices) > 0 && nrow(high_corr_indices) > 0) {
      high_corr_df <- data.frame(
        Var1 = rownames(cor_matrix)[high_corr_indices[, 1]],
        Var2 = colnames(cor_matrix)[high_corr_indices[, 2]],
        Correlation = round(cor_matrix[high_corr_indices], 3),
        stringsAsFactors = FALSE
      )
      high_corr_df <- high_corr_df[order(-abs(high_corr_df$Correlation)), ]
      
      cat("发现", nrow(high_corr_df), "个高度相关的变量对:\n")
      print(head(high_corr_df, 10))  # 只显示前10个
    } else {
      cat("未发现高度相关（|r| ≥ 0.8）的变量对\n")
      high_corr_df <- data.frame(Var1 = character(), 
                                 Var2 = character(), 
                                 Correlation = numeric(),
                                 stringsAsFactors = FALSE)
    }
    
    # 中等相关性的变量对
    cat("\n寻找中等相关的变量对（0.5 ≤ |r| < 0.8）...\n")
    
    moderate_corr_indices <- which(abs(cor_matrix) >= 0.5 & abs(cor_matrix) < 0.8 & 
                                     upper.tri(cor_matrix), arr.ind = TRUE)
    
    if (length(moderate_corr_indices) > 0 && nrow(moderate_corr_indices) > 0) {
      moderate_corr_df <- data.frame(
        Var1 = rownames(cor_matrix)[moderate_corr_indices[, 1]],
        Var2 = colnames(cor_matrix)[moderate_corr_indices[, 2]],
        Correlation = round(cor_matrix[moderate_corr_indices], 3),
        stringsAsFactors = FALSE
      )
      moderate_corr_df <- moderate_corr_df[order(-abs(moderate_corr_df$Correlation)), ]
      
      cat("发现", nrow(moderate_corr_df), "个中等相关的变量对（显示前10个）:\n")
      print(head(moderate_corr_df, 10))
    } else {
      cat("未发现中等相关（0.5 ≤ |r| < 0.8）的变量对\n")
      moderate_corr_df <- data.frame(Var1 = character(), 
                                     Var2 = character(), 
                                     Correlation = numeric(),
                                     stringsAsFactors = FALSE)
    }
    
    # 可视化相关性矩阵（仅当变量数量合理时）
    if (ncol(corr_data) <= 30) {
      cat("\n生成相关性矩阵图...\n")
      
      png(file.path(FIG_DIR, "correlation_matrix.png"), 
          width = 12, height = 10, units = "in", res = 300)
      
      tryCatch({
        corrplot(cor_matrix, method = "color", type = "upper",
                 tl.col = "black", tl.srt = 45,
                 addCoef.col = "black", number.cex = 0.7,
                 col = colorRampPalette(c("blue", "white", "red"))(100),
                 title = "Pearson相关系数矩阵",
                 mar = c(0, 0, 2, 0))
      }, error = function(e) {
        plot.new()
        title(main = "相关性矩阵")
        text(0.5, 0.5, paste("无法显示图形:", e$message), cex = 1.2)
      })
      
      dev.off()
      cat("相关性矩阵图已保存至:", file.path(FIG_DIR, "correlation_matrix.png"), "\n")
    } else {
      cat("变量数量太多（>30），跳过相关性矩阵图\n")
    }
    
    # 相关性分布直方图
    cat("\n生成相关性分布图...\n")
    
    png(file.path(FIG_DIR, "correlation_distribution.png"), 
        width = 10, height = 6, units = "in", res = 300)
    
    tryCatch({
      # 提取上三角的相关系数
      cor_values <- cor_matrix[upper.tri(cor_matrix)]
      
      if (length(cor_values) > 0) {
        hist_data <- data.frame(Correlation = cor_values)
        
        p <- ggplot(hist_data, aes(x = Correlation)) +
          geom_histogram(binwidth = 0.05, fill = "skyblue", 
                         color = "black", alpha = 0.7) +
          geom_vline(xintercept = c(-0.8, -0.5, 0.5, 0.8), 
                     linetype = "dashed", 
                     color = c("red", "orange", "orange", "red")) +
          labs(title = "Pearson相关系数分布",
               x = "相关系数", 
               y = "频数") +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, size = 14),
                axis.title = element_text(size = 12))
        
        print(p)
      } else {
        plot.new()
        title(main = "无相关系数数据")
      }
    }, error = function(e) {
      plot.new()
      title(main = "相关性分布图")
      text(0.5, 0.5, paste("无法生成图形:", e$message))
    })
    
    dev.off()
    cat("相关性分布图已保存至:", file.path(FIG_DIR, "correlation_distribution.png"), "\n")
    
  } else {
    cat("数值型变量数量不足（<2），跳过相关性分析\n")
    cor_matrix <- NULL
    high_corr_df <- NULL
    moderate_corr_df <- NULL
  }
  
} else {
  cat("\n⚠️ 错误：没有数值型变量可用于共线性分析！\n")
  cat("请检查数据是否包含数值型特征。\n")
  cat("当前特征数据类型分布:\n")
  print(table(sapply(features_train, class)))
  
  vif_df <- NULL
  cor_matrix <- NULL
  high_corr_df <- NULL
  moderate_corr_df <- NULL
}

# 11. 保存分析结果
cat("\n=== 保存分析结果 ===\n")

if (exists("vif_df") || exists("cor_matrix")) {
  tryCatch({
    save(features_train, numeric_cols, vif_df, cor_matrix, 
         high_corr_df, moderate_corr_df,
         file = file.path(DATA_DIR, "collinearity_analysis_results.rdata"))
    
    cat("分析结果已保存至:", file.path(DATA_DIR, "collinearity_analysis_results.rdata"), "\n")
    
    # 也保存为CSV文件以便查看
    if (!is.null(vif_df)) {
      write.csv(vif_df, file.path(DATA_DIR, "vif_results.csv"), row.names = FALSE)
      cat("VIF结果已保存至:", file.path(DATA_DIR, "vif_results.csv"), "\n")
    }
    
    if (!is.null(high_corr_df) && nrow(high_corr_df) > 0) {
      write.csv(high_corr_df, file.path(DATA_DIR, "high_correlation_pairs.csv"), row.names = FALSE)
      cat("高相关性变量对已保存至:", file.path(DATA_DIR, "high_correlation_pairs.csv"), "\n")
    }
    
  }, error = function(e) {
    cat("保存结果时出错:", e$message, "\n")
  })
} else {
  cat("没有分析结果可保存\n")
}

# 12. 分析总结
cat("\n=== 共线性分析总结 ===\n")
cat("1. 原始数据维度:", dim(train_data), "\n")
cat("2. 特征变量数量:", ncol(features_train), "\n")
cat("3. 数值型变量数量:", sum(numeric_cols), "\n")

if (!is.null(vif_df)) {
  cat("4. VIF分析变量数量:", nrow(vif_df), "\n")
  if (nrow(vif_df) > 0) {
    cat("   最高VIF:", max(vif_df$VIF), "\n")
    cat("   平均VIF:", mean(vif_df$VIF), "\n")
  }
}

if (!is.null(high_corr_df) && nrow(high_corr_df) > 0) {
  cat("5. 高相关性变量对数量:", nrow(high_corr_df), "\n")
  cat("   最高相关系数:", max(abs(high_corr_df$Correlation)), "\n")
}

# 13. 处理建议
cat("\n=== 共线性处理建议 ===\n")
cat("1. 如果VIF > 10或相关系数 |r| > 0.8，考虑移除冗余变量\n")
cat("2. 处理方法选项：\n")
cat("   a) 基于领域知识选择保留哪个变量\n")
cat("   b) 使用主成分分析（PCA）\n")
cat("   c) 使用正则化回归（LASSO、Ridge）\n")
cat("   d) 创建复合特征或交互项\n")
cat("3. 对于临床模型，优先考虑临床意义和可解释性\n")

cat("\n=== 共线性分析完成 ===\n")

