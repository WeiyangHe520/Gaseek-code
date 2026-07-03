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
FIG_DIR <- "figures（肝癌）/"    # Output directory for figures
DATA_DIR <- "data（肝癌）/"      # Output directory for data
dir.create(FIG_DIR, showWarnings = FALSE)
dir.create(DATA_DIR, showWarnings = FALSE)

##数据探索
###--------------###--------------###--------------###--------------###--------------
library(DataExplorer)
library(ggplot2)
# Load data
# 重新读取数据，正确处理缺失值
cancer_data <- read.csv("./四川原始数据2.csv", 
                        na.strings = c("", "NA", "NULL", "NaN", "N/A"),
                        stringsAsFactors = FALSE)

# 检查修复后的结果
cat("=== 数据读取诊断 ===\n")
cat("数据维度:", dim(cancer_data), "\n\n")

# 检查BMI列
if("BMI" %in% names(cancer_data)) {
  # 确保BMI是数值型
  cancer_data$BMI <- as.numeric(cancer_data$BMI)
  
  bmi_na_count <- sum(is.na(cancer_data$BMI))
  bmi_na_rate <- mean(is.na(cancer_data$BMI)) * 100
  
  cat("BMI列缺失情况:\n")
  cat("  缺失数量:", bmi_na_count, "/", nrow(cancer_data), "\n")
  cat("  缺失率:", round(bmi_na_rate, 2), "%\n")
  cat("  有效样本:", sum(!is.na(cancer_data$BMI)), "\n")
}
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
  scale_fill_gradientn(colors = c("#377eb8", "#ff7f00", "#e41a1c"),
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
  flextable::save_as_docx(path = "./data（肝癌）/tab1_all.docx")

table_train %>%
  as_flex_table() %>%
  flextable::save_as_docx(path = "./data（肝癌）/tab1_train.docx")

table_test %>%
  as_flex_table() %>%
  flextable::save_as_docx(path = "./data（肝癌）/tab1_test.docx")

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
write.csv(train_data, file = 'train_data（四川新版数据多任务）.csv')
write.csv(test_data, file = 'test_data（四川新版数据）.csv')
train_data = read.csv('train_data（四川新版数据多任务）.csv')
test_data = read.csv('test_data（四川新版数据）2.csv')
train_data$group <- factor(train_data$group, 
                           levels = c("control", "cancer"))
test_data$group <- factor(test_data$group, 
                          levels = c("control", "cancer"))
sum(is.na(train_data))
sum(is.na(test_data))

save(train_data,test_data,file = ".left_data.rdata")
save(train_data,test_data,file = ".original_data")
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
# glmnet (弹性网络)
set.seed(825)
traincontrol <- trainControl(method="repeatedcv", 
                             number=5, 
                             repeats=1, 
                             classProbs = TRUE)

# 使用 glmnet 方法
glmnet_filter <- train(group ~ ., 
                       data = train_data, 
                       method = "glmnet", 
                       trControl = traincontrol,
                       tuneLength = 10,  # 尝试10个 lambda 和 alpha 的组合
                       preProcess = c("center", "scale"))  # glmnet 需要数据标准化

# 查看调参结果
print(glmnet_filter$bestTune)

# 提取变量重要性
glmnet_imp <- varImp(glmnet_filter, scale = TRUE)
plot(glmnet_imp)

# 可选：查看被选中的非零系数特征
# 提取最终模型的系数
coef_matrix <- as.matrix(coef(glmnet_filter$finalModel, 
                              s = glmnet_filter$bestTune$lambda))
# 找出非零系数的特征（排除截距）
selected_features <- rownames(coef_matrix)[which(coef_matrix[, 1] != 0)]
selected_features <- selected_features[selected_features != "(Intercept)"]
cat("\n被选中的特征数量:", length(selected_features), "\n")
print(selected_features)

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
  geom_point(size = 4, shape = 19, color = "#377eb8") +
  geom_line(linewidth = 1.4, color = "#e41a1c") +
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

varimp_data <- data.frame(feature = row.names(varImp(result_rfe1))[1:17],
                          importance = varImp(result_rfe1)[1:17, 1])
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
    aes(label = sprintf("%.1f", importance)), 
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
  scale_fill_gradient(low = "#377eb8", high = "#e41a1c") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  scale_x_discrete(labels = function(x) gsub("_", " ", x))  # 只替换下划线为空格，不换行

print(p3_rotated)
ggsave(file.path(FIG_DIR, "glmnet_fixed_calibration1.TIFF"), 
       p3_rotated, width = 8, height = 8.5, dpi = 300)

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

