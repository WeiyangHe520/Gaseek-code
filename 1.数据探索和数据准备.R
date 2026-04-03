rm(list = ls())
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
FIG_DIR <- "figures/"    # Output directory for figures
DATA_DIR <- "data/"      # Output directory for data
dir.create(FIG_DIR, showWarnings = FALSE)
dir.create(DATA_DIR, showWarnings = FALSE)

##数据探索
###--------------###--------------###--------------###--------------###--------------
library(DataExplorer)
library(ggplot2)
# Load data
cancer_data <- read.csv("./四川省肿瘤医院合并数据.csv")
# Calculate missing rate for each variable
# Calculate missing rate for each variable
missing_summary <- data.frame(
  variable = names(cancer_data),
  missing_rate = colMeans(is.na(cancer_data)) * 100
) %>%
  filter(missing_rate > 0) %>%  # Only show variables with missing values
  arrange(desc(missing_rate))

# Save missing value statistics
write.csv(missing_summary, 
          paste0(DATA_DIR, "missing_summary.csv"),
          row.names = FALSE)

# Create beautiful bar chart
tiff_file <- paste0(FIG_DIR, "missing_values_ggplot.tiff")

tiff(tiff_file, 
     width = 9, 
     height = 5.17, 
     units = "in", 
     res = 300, 
     compression = "lzw")

# Create ggplot
p <- ggplot(missing_summary, 
            aes(x = reorder(variable, missing_rate), 
                y = missing_rate,
                fill = missing_rate)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = sprintf("%.2f%%", missing_rate)), 
            hjust = -0.1, 
            size = 4.5,
            color = "black") +
  scale_fill_gradientn(colors = c("#4393c3", "#ffd966", "#d6604d"),
                       name = "Missing Rate (%)") +
  coord_flip() +  # Horizontal bar chart, easier to read
  labs(title = "Missing Values Analysis by Variable",
       subtitle = paste(nrow(missing_summary), "variables have missing values"),
       x = "Variable",
       y = "Missing Rate (%)") +
  theme_minimal(base_size = 18) +
  theme(
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
    axis.title.x = element_text(size = 20, face = "bold", margin = margin(t = 10)),
    axis.title.y = element_text(size = 20, face = "bold", margin = margin(r = 10)),
    axis.text.x = element_text(size = 18, color = "black"),
    axis.text.y = element_text(size = 18, color = "black"),
    legend.title = element_text(size = 20, face = "bold"),
    legend.text = element_text(size = 18),
    panel.grid.major = element_line(color = "gray90", size = 0.5),
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(5, 5, 5, 5)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)),
                     limits = c(0, max(missing_summary$missing_rate) * 1.2))

print(p)
dev.off()

##数据准备
###--------------###--------------###--------------###--------------###--------------
# Split into training and test sets (80%/20%)
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

##分组不平衡
table(train_data$group)
library(imbalance)
#Generate synthetic positive instances using ADASYN algorithm
imbalanceRatio(train_data, classAttr = "group")
adasyn_data <- imbalance::oversample(
  train_data, classAttr = "group",
  method = "MWMOTE", 
  ratio = 1# 平衡两类
)
imbalanceRatio(adasyn_data, classAttr = "group")
train_data=adasyn_data
##标准化
#"YeoJohnson" 正态化
train_pre<- preProcess(train_data, method = c("center", "scale", "YeoJohnson"))
train_data=predict(train_pre,train_data)
test_pre<- preProcess(test_data, method = c("center", "scale", "YeoJohnson"))
test_data=predict(test_pre,test_data)
write.csv(train_data, file = 'train_data（共线分析）.csv')
write.csv(test_data, file = 'test_data（共线分析）.csv')
train_data = read.csv('train_data（共线分析）.csv')
test_data = read.csv('test_data（共线分析）.csv')
train_data$group <- factor(train_data$group, 
                           levels = c("control", "cancer"))
test_data$group <- factor(test_data$group, 
                          levels = c("control", "cancer"))
sum(is.na(train_data))
sum(is.na(test_data))

save(train_data,test_data,file = ".left_data.rdata")
####-----
#特征筛选
##过滤法
#https://github.com/Huaichao2018/Clabomic
source("./source.R")
# Wilcoxon rank sum test
wilcox_res <- do_batch_Wilcoxon(mat = train_data, p_value = 1)
# ROC analysis
roc_res <- do_batch_roc(train_data)
#RF 随机森林
set.seed(825)
traincontrol <- trainControl(method="repeatedcv",number=5,repeats=1,classProbs = TRUE)
rf_filter <- train(group~., data=train_data, method="rf",  trControl=traincontrol)
rf_imp=varImp(rf_filter)
plot(rf_imp)
#glmboost
set.seed(825)
traincontrol <- trainControl(method="repeatedcv",number=5,repeats=1,classProbs = TRUE)
glmboost_filter <- train(group~., data=train_data, method="glmboost",  trControl=traincontrol)
glmboost_imp=varImp(glmboost_filter)
plot(glmboost_imp)

##Selection by filter (SBF)
# Set up our control scheme for the sbf. Will use ready-made random forest i.e. rfSBF
## Selection by filter (SBF)

# 确保 train_data 中的 group 是正确的因子
train_data$group <- factor(train_data$group, levels = c("control", "cancer"))

# 检查数据结构
str(train_data)

# 设置 sbf 的控制参数
filterCtrl <- sbfControl(functions = rfSBF, # 使用预定义的随机森林函数
                         method = "repeatedcv", # 外部/外层循环的重采样方法
                         number = 5, # 5折交叉验证
                         repeats = 5, # 重复5次
                         verbose = TRUE) # 添加详细输出以便调试

# 执行 SBF
sbf_rf <- sbf(
  group ~ ., 
  data = train_data, 
  # 对于分类问题，使用合适的指标，例如：
  metric = "Accuracy",  # 或者 "Kappa", "ROC"
  sbfControl = filterCtrl  # 注意：这里应该是 sbfControl，不是 sbfcontrol
)

# 查看结果
sbf_rf

# 查看选中的变量
sbf_rf$optVariables

# 查看性能指标
sbf_rf$results
#top3取交集
roc_res$id <- rownames(roc_res)
merged <- merge(wilcox_res, roc_res, by = "id")
merged$auc <- merged$res
merged$neg_log_p <- -log10(merged$p)
# Merge results
library (UpSetR)
#
marker_lists <- list (
  wilcox = wilcox_res$id[1:3] ,
  roc = roc_res[order(roc_res$res,decreasing = T),]$id[1:3] ,
  RF = rf_imp$importance%>%arrange(desc(Overall))%>% top_n(3) %>% rownames(), 
  glmboost = glmboost_imp$importance%>%arrange(desc(Overall))%>% top_n(3) %>% rownames()
  )

upset_data=fromList(marker_lists)
p1=upset(upset_data, sets = names (marker_lists), order.by = "freq" , 
      text.scale = 1.2 , mainbar.y.label = "number intersected" , 
      sets.x.label = "number selected" )
p1

dput(colnames(train_data))
merged$group <- ifelse(merged$id %in% c("HGB",  "PLT", "RBC", "WBC"), "Selected", "Not Selected")
plot_feature_selection <- function(merged_results, selected_features) {
  ggplot(merged_results, aes(x = auc, y = neg_log_p)) +
    geom_point(alpha = 0.6, stroke = 0.29, shape = 21,
               aes(fill = group, size = neg_log_p)) +
    geom_vline(xintercept = 0.55, linetype = 2, color = "black", linewidth = 0.8, alpha = 0.5) +
    geom_hline(yintercept = -log10(0.05), linetype = 2, color = "black", linewidth = 0.8, alpha = 0.5) +
    scale_fill_manual(values = c("#7f8fa6", "#e84118")) +
    labs(x = "AUC", y = "-log10(p-value)") +
    theme_prism(base_size = 16) +
    ggrepel::geom_text_repel(
      aes(label = id),
      data = merged_results[merged_results$id %in% selected_features, ],
      color = "black",
      min.segment.length = 0,
      force = 0.1,
      nudge_x = 0.5,
      direction = "y",
      hjust = 0,
      alpha = 0.8,
      size = 3,
      segment.size = 0.3,
      segment.curvature = -0.05
    )
}

# Generate and save volcano plot
p2<- plot_feature_selection(merged, c("HGB",  "PLT", "RBC", "WBC"))
p2

# Feature distribution visualization module
# ----------------------------------------
#' Plot distribution of selected features
plot_feature_distribution <- function(data, feature, group_col = "group") {
  group_levels <- unique(data[[group_col]])
  comparisons <- list(group_levels)
  
  ggplot(data, aes_string(x = group_col, y = feature, fill = group_col)) +
    geom_signif(comparisons = comparisons, textsize = 4) +
    geom_boxplot(outlier.alpha = 0) +
    geom_jitter(color = "black", fill = "white",
                position = position_jitter(0.22),
                shape = 21, size = 3, alpha = 1) +
    scale_fill_manual(values = c("#999999", "#d01c8b")) +
    theme_classic() +
    theme(legend.position = "none", text = element_text(size = 10)) +
    scale_y_continuous(labels = function(x) format(x, scientific = TRUE))
}

# Example feature visualizations
p_plt <- plot_feature_distribution(train_data, "PLT")
p_rbc <- plot_feature_distribution(train_data, "RBC")

 p_plt 
 p_rbc

##包裹式
###--------------###--------------###--------------###--------------###--------------
# Define the control using a random forest selection function
control <- rfeControl(functions = rfFuncs, # random forest
                      method = "repeatedcv", # repeated cv
                      repeats = 1, # number of repeats
                      number = 5) # number of folds

#Run RFE
result_rfe1 <- rfe(group~.,data=train_data,
                   sizes = c(1:(ncol(train_data)-1)),
                   rfeControl = control)

# Print the results
result_rfe1

# Print the selected features
predictors(result_rfe1)

# Print the results visually
# 创建图形对象
p <- ggplot(data = result_rfe1, metric = "Accuracy") +
  geom_point(size = 4, shape = 19, color = "steelblue") +
  geom_line(linewidth = 1.4, color = "darkred") +
  scale_y_continuous(
    labels = scales::label_number(accuracy = 0.01)  # 保留2位小数
  ) +
  labs(
    title = "Recursive feature elimination (RFE)",
    x = "Variables",
    y = "Accuracy"
  ) +
  theme_bw() +
  theme(
    # 坐标轴文字
    axis.text = element_text(size = 18),                    # 刻度标签
    axis.text.x = element_text(size = 20, angle = 0),      # 单独设置X轴刻度
    axis.text.y = element_text(size = 20),                 # 单独设置Y轴刻度
    
    # 坐标轴标题
    axis.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 20, face = "bold"), # 单独设置X轴标题
    axis.title.y = element_text(size = 20, face = "bold"), # 单独设置Y轴标题
    
    # 图表标题
    plot.title = element_text(size = 20, hjust = 0.5, face = "bold"),
    
    # 图例（如果有的话）
    legend.title = element_text(size = 16),                # 图例标题
    legend.text = element_text(size = 14),                 # 图例文字
    
    # 网格线等其他元素
    panel.grid.major = element_line(linewidth = 0.5),
    panel.grid.minor = element_line(linewidth = 0.25)
  )

# 显示图形
print(p)

# 保存为TIFF格式（高分辨率）
ggsave(
  filename = "RFE_Accuracy_Plot.tiff",  # 文件名
  plot = p,                             # 图形对象
  device = "tiff",                      # 格式：TIFF
  width = 6,                            # 宽度（英寸）
  height = 4.5,                           # 高度（英寸）
  units = "in",                         # 单位：英寸
  dpi = 300,                            # 分辨率：300 dpi
  compression = "lzw"                   # 压缩算法（减小文件大小）
)

varimp_data <- data.frame(feature = row.names(varImp(result_rfe1))[1:19],
                          importance = varImp(result_rfe1)[1:19, 1])
p3_rotated <- ggplot(data = varimp_data, 
                     aes(x = reorder(feature, importance),  # 按重要性升序
                         y = importance, 
                         fill = importance)) +
  geom_bar(stat = "identity", width = 0.7) +
  coord_flip() +
  labs(
    x = NULL,
    y = "Variable Importance",
    title = "Features Importance"
  ) +
  geom_text(
    aes(label = sprintf("%.2f", importance)), 
    hjust = -0.1,
    size = 5
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
    axis.title.x = element_text(size = 20, face = "bold"),
    axis.text.y = element_text(size = 16, face = "bold", 
                               margin = margin(r = 10)),  # 增加右边距
    axis.text.x = element_text(size = 18),
    legend.position = "none",
    panel.grid.major.y = element_blank(),
    plot.margin = margin(l = 0, r = 0, t = 20, b = 20)  # 增加整体边距
  ) +
  scale_fill_gradient(low = "#3498db", high = "#e74c3c") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  
  # 改进的换行函数
  scale_x_discrete(
    labels = function(x) {
      # 创建换行函数，每部分最多3个单词或15个字符
      wrap_labels <- function(label, max_words = 3, max_chars = 15) {
        # 替换下划线为空格
        label <- gsub("_", " ", label)
        
        # 分割单词
        words <- strsplit(label, " ")[[1]]
        
        if (length(words) <= max_words && nchar(label) <= max_chars) {
          return(label)
        }
        
        # 构建换行文本
        lines <- character()
        current_line <- character()
        current_chars <- 0
        
        for (word in words) {
          if (current_chars + nchar(word) + ifelse(length(current_line) > 0, 1, 0) <= max_chars && 
              length(current_line) < max_words) {
            current_line <- c(current_line, word)
            current_chars <- current_chars + nchar(word) + ifelse(length(current_line) > 1, 1, 0)
          } else {
            if (length(current_line) > 0) {
              lines <- c(lines, paste(current_line, collapse = " "))
            }
            current_line <- word
            current_chars <- nchar(word)
          }
        }
        
        if (length(current_line) > 0) {
          lines <- c(lines, paste(current_line, collapse = " "))
        }
        
        return(paste(lines, collapse = "\n"))
      }
      
      sapply(x, wrap_labels)
    }
  )

print(p3_rotated)
ggsave(file.path(FIG_DIR, "glmnet_fixed_calibration.TIFF"), 
       p3_rotated, width = 8, height = 8.5, dpi = 300)  # 增加宽度和高度

##嵌入式
library(glmnet)
set.seed(123)
lasso_fit <- cv.glmnet(x=as.matrix(train_data[,-ncol(train_data)]),y=train_data[,ncol(train_data)],
                       family = "binomial", type.measure = 'deviance')
plot(lasso_fit)
coefficient <- coef(lasso_fit, s=lasso_fit$lambda.min)
Active.Index<-coefficient[as.numeric(coefficient) != 0,]

mod<-glmnet(x=as.matrix(train_data[,-ncol(train_data)]),y=train_data[,ncol(train_data)],
            family = "binomial")

plot(mod)
coef.increase<-dimnames(coefficient[coefficient[,1]>0,0])[[1]]
coef.decrease<-dimnames(coefficient[coefficient[,1]<0,0])[[1]]
#get ordered list of variables as they appear at smallest lambda
cof=coef(mod)
allnames<-names(cof[,ncol(cof)][order(cof[, ncol(cof)],decreasing=TRUE)])
allnames<-setdiff(allnames,allnames[grep("Intercept",allnames)])

#assign colors"#00AFBB", "#E7B800", "#FC4E07"
cols<-rep("#E7B800",length(allnames))
cols[allnames %in% coef.increase]<-"#FC4E07"      
cols[allnames %in% coef.decrease]<-"#00AFBB"        
#install.packages("plotmo")
library(plotmo)
p1=plot_glmnet(mod,label=TRUE,s=lasso_fit$lambda.min,col=cols)
p1

#######
train_data$group=factor(train_data$group,levels = c('control', 'cancer'))
test_data$group=factor(test_data$group,levels = c('control', 'cancer'))
save(train_data,test_data,file = "./4/left_data.rdata")

