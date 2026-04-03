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
set.seed(3456)  # For reproducibility
FIG_DIR <- "figures/"    # Output directory for figures
DATA_DIR <- "data/"      # Output directory for data
dir.create(FIG_DIR, showWarnings = FALSE)
dir.create(DATA_DIR, showWarnings = FALSE)

##数据探索
###--------------###--------------###--------------###--------------###--------------
library(DataExplorer)
# Load data
Gastrointestinal_cancer_data1 <- read.csv("./四川省肿瘤医院合并数据（2）.csv")
Gastrointestinal_cancer_data <- read.csv("./武汉大学中南医院合并数据（3）.csv")
#基础探索
str(Gastrointestinal_cancer_data)
dim(Gastrointestinal_cancer_data)
#直接看
View(Gastrointestinal_cancer_data)
#data explore包
introduce(Gastrointestinal_cancer_data)
plot_intro(Gastrointestinal_cancer_data)
#离群值
plot_missing(Gastrointestinal_cancer_data)
#分类或者文本变量
plot_bar(Gastrointestinal_cancer_data)
#连续变量
plot_histogram(Gastrointestinal_cancer_data)
plot_qq(Gastrointestinal_cancer_data)
log_qq_data <- update_columns(Gastrointestinal_cancer_data1, 5:84, function(x) log(x+1))
plot_qq(log_qq_data[, 5:84], sampled_rows = 1000L)

##多变量
#相关
plot_correlation(na.omit(Gastrointestinal_cancer_data[,-ncol(Gastrointestinal_cancer_data)]))
#主成分
plot_prcomp(na.omit(Gastrointestinal_cancer_data[,-ncol(Gastrointestinal_cancer_data)]))
#箱体图
plot_boxplot(Gastrointestinal_cancer_data, by = "group")
create_report(Gastrointestinal_cancer_data)

pdf("./figures/eda.pdf",10,10)
plot_intro(Gastrointestinal_cancer_data)
#离群值
plot_missing(Gastrointestinal_cancer_data)
plot_bar(Gastrointestinal_cancer_data)
plot_histogram(Gastrointestinal_cancer_data)
plot_qq(Gastrointestinal_cancer_data)
log_qq_data <- update_columns(Gastrointestinal_cancer_data, 5:84, function(x) log(x + 1))
plot_qq(log_qq_data[, 5:84])
plot_correlation(na.omit(Gastrointestinal_cancer_data[,-ncol(Gastrointestinal_cancer_data)]))
plot_prcomp(na.omit(Gastrointestinal_cancer_data[,-ncol(Gastrointestinal_cancer_data)]))
plot_boxplot(Gastrointestinal_cancer_data, by = "group")
dev.off()

##数据准备
###--------------###--------------###--------------###--------------###--------------
# Split into training and test sets (70%/30%)
library(caret) #机器学习
library(tidyverse) #包括ggplot2,等基础R包
library(gtsummary) #表格输出
library(eoffice) #输出office文件
train_data <- Gastrointestinal_cancer_data1
test_data <- Gastrointestinal_cancer_data
Gastrointestinal_cancer_data_test=Gastrointestinal_cancer_data


#输出
table_all <- 
  tbl_summary(
    test_data,
    by = group,
    missing = "no"
  ) %>%
  add_n() %>%
  add_p(
    test = list(
      all_continuous() ~ "t.test",
      all_categorical() ~ "chisq.test"
    ),
    test.args = list(
      all_continuous() ~ list(var.equal = FALSE),  # 使用Welch T检验（不要求方差齐性）
      all_categorical() ~ list(correct = FALSE)
    )
  ) %>%
  modify_header(label = "**Variable**") %>%
  bold_labels()

table_all <- 
  tbl_summary(
    test_data,
    by = group,
    missing = "no"
  ) %>%
  add_n() %>%
  add_p(
    test = list(
      all_continuous() ~ "aov",          # 连续变量使用单因素方差分析
      all_categorical() ~ "chisq.test"   # 分类变量仍使用卡方检验
    ),
    test.args = all_categorical() ~ list(correct = FALSE)
  ) %>%
  modify_header(label = "**Variable**") %>%
  bold_labels()


table_all %>%
  as_flex_table() %>%
  flextable::width(width = 1.2) %>%  # 设置所有列宽度为1.2英寸
  flextable::save_as_docx(path = "./data/tab1_all.docx")

###------离群值不涉及
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
#"center", "scale", 标准化
#"YeoJohnson" 正态化
train_pre<- preProcess(train_data, method = c("center", "scale", "YeoJohnson"))
train_data=predict(train_pre,train_data)
test_pre<- preProcess(test_data, method = c("center", "scale", "YeoJohnson"))
test_data=predict(test_pre,test_data)
write.csv(train_data, file = 'train_data.csv')
write.csv(test_data, file = 'test_data.csv')
train_data = read.csv('train_data（278）.csv')
test_data = read.csv('test_data（四川内部）.csv')
train_data$group <- factor(train_data$group, 
                           levels = c("control", "cancer"))
test_data$group <- factor(test_data$group, 
                          levels = c("control", "cancer"))
sum(is.na(train_data))
sum(is.na(test_data))

save(train_data,test_data,file = ".left_data.rdata")

