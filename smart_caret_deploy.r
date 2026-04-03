# ============================================
# 智能化 Caret 模型自动部署系统（修复版）
# 自动提取模型参数、输入输出，一键生成 Web 应用
# ============================================

# 安装必要的包
required_packages <- c("caret", "shiny", "DT", "plotly", "bslib")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(caret)
library(shiny)
library(DT)
library(plotly)
library(bslib)

# ============================================
# 核心功能：智能模型分析器（使用 list 代替 R6）
# ============================================

create_model_analyzer <- function(caret_model) {
  
  analyzer <- list()
  analyzer$model <- caret_model
  analyzer$model_info <- list()
  
  # 自动提取模型所有信息
  extract_info <- function() {
    m <- caret_model
    
    # 1. 基本信息
    analyzer$model_info$method <<- m$method
    analyzer$model_info$model_type <<- m$modelType
    
    # 2. 自动提取输入特征
    train_data <- m$trainingData
    outcome_col <- which(names(train_data) == ".outcome")
    feature_cols <- setdiff(1:ncol(train_data), outcome_col)
    
    analyzer$model_info$features <<- names(train_data)[feature_cols]
    analyzer$model_info$feature_types <<- sapply(train_data[, feature_cols, drop = FALSE], class)
    
    # 3. 自动提取输出信息
    if (m$modelType == "Classification") {
      analyzer$model_info$classes <<- levels(train_data$.outcome)
      analyzer$model_info$n_classes <<- length(analyzer$model_info$classes)
    }
    
    # 4. 自动提取特征范围
    analyzer$model_info$feature_ranges <<- lapply(analyzer$model_info$features, function(feat) {
      val <- train_data[[feat]]
      if (is.numeric(val)) {
        list(
          type = "numeric",
          min = min(val, na.rm = TRUE),
          max = max(val, na.rm = TRUE),
          mean = mean(val, na.rm = TRUE),
          median = median(val, na.rm = TRUE)
        )
      } else if (is.factor(val)) {
        list(
          type = "factor",
          levels = levels(val)
        )
      } else {
        list(
          type = "character",
          unique_values = unique(as.character(val))
        )
      }
    })
    names(analyzer$model_info$feature_ranges) <<- analyzer$model_info$features
    
    # 5. 模型性能
    analyzer$model_info$results <<- m$results
    analyzer$model_info$best_tune <<- m$bestTune
    
    invisible(analyzer)
  }
  
  # 智能预测
  analyzer$smart_predict <- function(input_data, type = "raw") {
    # 确保输入是数据框
    if (!is.data.frame(input_data)) {
      input_data <- as.data.frame(input_data, stringsAsFactors = FALSE)
    }
    
    # 确保列名正确
    required_features <- analyzer$model_info$features
    if (!all(required_features %in% names(input_data))) {
      missing <- setdiff(required_features, names(input_data))
      stop(paste("Missing required features:", paste(missing, collapse = ", ")))
    }
    
    # 选择需要的列
    input_data <- input_data[, required_features, drop = FALSE]
    
    # 预测
    if (type == "prob" && analyzer$model_info$model_type == "Classification") {
      pred_class <- predict(analyzer$model, input_data, type = "raw")
      pred_prob <- predict(analyzer$model, input_data, type = "prob")
      
      return(list(
        prediction = as.character(pred_class),
        probabilities = pred_prob,
        input = input_data
      ))
    } else {
      pred <- predict(analyzer$model, input_data)
      return(list(
        prediction = as.numeric(pred),
        input = input_data
      ))
    }
  }
  
  # 生成模型报告
  analyzer$generate_report <- function() {
    info <- analyzer$model_info
    
    report <- list(
      title = paste(toupper(info$method), "Model Report"),
      model_type = info$model_type,
      features = data.frame(
        Feature = info$features,
        Type = as.character(info$feature_types),
        stringsAsFactors = FALSE
      ),
      performance = info$results
    )
    
    if (info$model_type == "Classification") {
      report$classes <- info$classes
    }
    
    return(report)
  }
  
  # 初始化
  extract_info()
  
  return(analyzer)
}

# ============================================
# 自动生成 Shiny App
# ============================================

create_smart_app <- function(caret_model, app_title = "Model Deployment App") {
  
  # 创建模型分析器
  analyzer <- create_model_analyzer(caret_model)
  info <- analyzer$model_info
  
  # UI 部分
  ui <- fluidPage(
    theme = bs_theme(version = 5, bootswatch = "flatly"),
    
    titlePanel(app_title),
    
    sidebarLayout(
      sidebarPanel(
        width = 3,
        h4("📊 Model Information"),
        tags$div(
          style = "background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;",
          tags$strong("Method: "), info$method, tags$br(),
          tags$strong("Type: "), info$model_type, tags$br(),
          if (info$model_type == "Classification") {
            tagList(
              tags$strong("Classes: "), paste(info$classes, collapse = ", ")
            )
          }
        ),
        
        hr(),
        
        h4("🎯 Make Prediction"),
        
        # 动态生成输入控件
        lapply(info$features, function(feat) {
          range_info <- info$feature_ranges[[feat]]
          
          if (range_info$type == "numeric") {
            # 计算合适的步长（保留2位小数）
            range_span <- range_info$max - range_info$min
            step_size <- round(range_span / 100, 2)
            if (step_size == 0) step_size <- 0.01
            
            sliderInput(
              inputId = paste0("input_", make.names(feat)),
              label = feat,
              min = round(range_info$min, 2),
              max = round(range_info$max, 2),
              value = round(range_info$median, 2),
              step = step_size
            )
          } else if (range_info$type == "factor") {
            selectInput(
              inputId = paste0("input_", make.names(feat)),
              label = feat,
              choices = range_info$levels
            )
          } else {
            textInput(
              inputId = paste0("input_", make.names(feat)),
              label = feat,
              value = range_info$unique_values[1]
            )
          }
        }),
        
        hr(),
        actionButton("predict_btn", "🚀 Predict", 
                     class = "btn-primary btn-lg btn-block",
                     style = "margin-bottom: 10px;"),
        
        actionButton("batch_btn", "📁 Batch Prediction", 
                     class = "btn-secondary btn-block",
                     onclick = "document.getElementById('main_tabs').getElementsByTagName('a')[2].click();")
      ),
      
      mainPanel(
        width = 9,
        
        tabsetPanel(
          id = "main_tabs",
          
          # Tab 1: Prediction Results
          tabPanel(
            "Prediction Results",
            icon = icon("chart-line"),
            br(),
            h4("📥 Input Data"),
            DTOutput("input_data_table"),
            br(),
            uiOutput("prediction_output"),
            br(),
            conditionalPanel(
              condition = "output.show_prob_plot",
              plotlyOutput("prob_plot", height = "300px")
            )
          ),
          
          # Tab 2: Model Details
          tabPanel(
            "Model Details",
            icon = icon("info-circle"),
            br(),
            h4("📈 Model Performance"),
            DTOutput("performance_table"),
            br(),
            h4("🔍 Feature Information"),
            DTOutput("feature_table")
          ),
          
          # Tab 3: Batch Prediction
          tabPanel(
            "Batch Prediction",
            icon = icon("file-csv"),
            br(),
            h4("Upload CSV File for Batch Prediction"),
            fileInput("batch_file", "Choose CSV File", 
                      accept = c(".csv", ".txt"),
                      buttonLabel = "Browse...",
                      placeholder = "No file selected"),
            hr(),
            h4("Or Enter Data Variables"),
            helpText("Enter variable names separated by commas (e.g., Sepal.Length, Sepal.Width)"),
            textInput("data_vars", "Data Variables:", 
                      value = paste(info$features, collapse = ", "),
                      width = "100%"),
            textAreaInput("data_values", "Data Values (one row per line, comma-separated):", 
                          value = "", 
                          rows = 5,
                          width = "100%",
                          placeholder = "Example:\n5.1, 3.5, 1.4, 0.2\n4.9, 3.0, 1.4, 0.2"),
            hr(),
            actionButton("run_batch", "Run Batch Prediction", 
                         class = "btn-success",
                         icon = icon("play")),
            downloadButton("download_batch", "Download Results", 
                           class = "btn-info",
                           style = "margin-left: 10px;"),
            br(), br(),
            DTOutput("batch_results")
          )
        )
      )
    )
  )
  
  # Server 部分
  server <- function(input, output, session) {
    
    # 收集用户输入
    get_input_data <- reactive({
      input_list <- list()
      for (feat in info$features) {
        input_id <- paste0("input_", make.names(feat))
        val <- input[[input_id]]
        
        # 类型转换并保留2位小数
        if (info$feature_ranges[[feat]]$type == "numeric") {
          input_list[[feat]] <- round(as.numeric(val), 2)
        } else {
          input_list[[feat]] <- val
        }
      }
      
      as.data.frame(input_list, stringsAsFactors = FALSE)
    })
    
    # 预测结果
    prediction_result <- eventReactive(input$predict_btn, {
      tryCatch({
        input_data <- get_input_data()
        analyzer$smart_predict(input_data, type = "prob")
      }, error = function(e) {
        list(error = e$message)
      })
    })
    
    # 显示输入数据表格
    output$input_data_table <- renderDT({
      req(prediction_result())
      result <- prediction_result()
      
      if (is.null(result$error)) {
        input_df <- result$input
        datatable(
          input_df,
          options = list(
            pageLength = 10, 
            scrollX = TRUE, 
            dom = 't',
            ordering = FALSE
          ),
          class = "display cell-border stripe",
          rownames = FALSE
        ) %>%
          formatRound(columns = which(sapply(input_df, is.numeric)), digits = 2)
      }
    })
    
    # 显示预测结果
    output$prediction_output <- renderUI({
      req(prediction_result())
      result <- prediction_result()
      
      if (!is.null(result$error)) {
        return(
          tags$div(
            class = "alert alert-danger",
            icon("exclamation-triangle"),
            " Error: ", result$error
          )
        )
      }
      
      if (info$model_type == "Classification") {
        pred_class <- result$prediction
        probs <- result$probabilities
        max_prob <- max(probs[1,])
        
        tagList(
          tags$div(
            style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 30px; border-radius: 10px; 
                     text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.2);",
            h2(style = "margin: 0; font-weight: bold;", "Prediction"),
            h1(style = "margin: 10px 0; font-size: 3em;", pred_class),
            h4(style = "margin: 0; opacity: 0.9;", 
               sprintf("Confidence: %.2f%%", max_prob * 100))
          ),
          br(),
          tags$div(
            style = "background: white; padding: 20px; border-radius: 10px; 
                     box-shadow: 0 2px 10px rgba(0,0,0,0.1);",
            h4("📊 Class Probabilities"),
            tags$table(
              class = "table table-striped",
              style = "margin-top: 15px;",
              tags$thead(
                tags$tr(
                  tags$th("Class"),
                  tags$th("Probability"),
                  tags$th("Visual")
                )
              ),
              tags$tbody(
                lapply(names(probs), function(cls) {
                  prob <- probs[[cls]][1]
                  tags$tr(
                    tags$td(tags$strong(cls)),
                    tags$td(sprintf("%.2f (%.2f%%)", prob, prob * 100)),
                    tags$td(
                      tags$div(
                        style = sprintf(
                          "background: linear-gradient(90deg, #667eea, #764ba2); 
                           width: %.2f%%; height: 25px; border-radius: 5px;
                           transition: width 0.3s ease;",
                          prob * 100
                        )
                      )
                    )
                  )
                })
              )
            )
          )
        )
      } else {
        # 回归结果
        pred_value <- result$prediction[1]
        tagList(
          tags$div(
            style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 30px; border-radius: 10px; 
                     text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.2);",
            h2(style = "margin: 0;", "Prediction Result"),
            h1(style = "font-size: 3em; margin: 10px 0;", 
               format(round(pred_value, 2), big.mark = ","))
          )
        )
      }
    })
    
    # 是否显示概率图
    output$show_prob_plot <- reactive({
      result <- prediction_result()
      !is.null(result) && is.null(result$error) && info$model_type == "Classification"
    })
    outputOptions(output, "show_prob_plot", suspendWhenHidden = FALSE)
    
    # 概率图
    output$prob_plot <- renderPlotly({
      req(prediction_result())
      result <- prediction_result()
      
      if (info$model_type == "Classification" && is.null(result$error)) {
        probs <- result$probabilities
        
        plot_ly(
          x = names(probs),
          y = as.numeric(probs[1,]),
          type = "bar",
          marker = list(
            color = c('#667eea', '#764ba2', '#f093fb'),
            line = list(color = 'white', width = 2)
          ),
          text = sprintf("%.2f%%", as.numeric(probs[1,]) * 100),
          textposition = "outside"
        ) %>%
          layout(
            title = "Prediction Probabilities",
            xaxis = list(title = "Class"),
            yaxis = list(title = "Probability", range = c(0, 1.1)),
            plot_bgcolor = '#f8f9fa',
            paper_bgcolor = '#f8f9fa',
            showlegend = FALSE
          )
      }
    })
    
    # 性能表格
    output$performance_table <- renderDT({
      datatable(
        info$results,
        options = list(pageLength = 10, scrollX = TRUE, dom = 'Bfrtip'),
        class = "display cell-border stripe",
        rownames = FALSE
      ) %>%
        formatRound(columns = which(sapply(info$results, is.numeric)), digits = 2)
    })
    
    # 特征表格
    output$feature_table <- renderDT({
      report <- analyzer$generate_report()
      
      # 添加范围信息
      feature_df <- report$features
      feature_df$Range <- sapply(info$features, function(feat) {
        rng <- info$feature_ranges[[feat]]
        if (rng$type == "numeric") {
          sprintf("[%.2f, %.2f]", rng$min, rng$max)
        } else {
          paste(rng$levels, collapse = ", ")
        }
      })
      
      datatable(
        feature_df,
        options = list(pageLength = 20, dom = 'Bfrtip'),
        class = "display cell-border stripe",
        rownames = FALSE
      )
    })
    
    # 批量预测结果存储
    batch_results_data <- reactiveVal(NULL)
    
    # 批量预测
    observeEvent(input$run_batch, {
      
      tryCatch({
        # 判断数据来源
        if (!is.null(input$batch_file)) {
          # 从文件读取
          batch_data <- read.csv(input$batch_file$datapath, stringsAsFactors = FALSE)
        } else if (nzchar(trimws(input$data_values))) {
          # 从文本框读取
          vars <- trimws(strsplit(input$data_vars, ",")[[1]])
          
          # 解析数据行
          lines <- strsplit(input$data_values, "\n")[[1]]
          lines <- lines[nzchar(trimws(lines))]
          
          if (length(lines) == 0) {
            showNotification("❌ Please enter data values", type = "error", duration = 3)
            return()
          }
          
          # 构建数据框
          data_list <- lapply(lines, function(line) {
            values <- trimws(strsplit(line, ",")[[1]])
            if (length(values) != length(vars)) {
              stop(sprintf("Row has %d values but expected %d", length(values), length(vars)))
            }
            values
          })
          
          batch_data <- as.data.frame(do.call(rbind, data_list), stringsAsFactors = FALSE)
          names(batch_data) <- vars
          
          # 类型转换并保留2位小数
          for (feat in info$features) {
            if (feat %in% names(batch_data)) {
              if (info$feature_ranges[[feat]]$type == "numeric") {
                batch_data[[feat]] <- round(as.numeric(batch_data[[feat]]), 2)
              }
            }
          }
        } else {
          showNotification("❌ Please upload a file or enter data values", type = "error", duration = 3)
          return()
        }
        
        results <- analyzer$smart_predict(batch_data, type = "prob")
        
        if (info$model_type == "Classification") {
          output_df <- cbind(
            batch_data,
            Prediction = results$prediction,
            results$probabilities
          )
        } else {
          output_df <- cbind(
            batch_data,
            Prediction = results$prediction
          )
        }
        
        batch_results_data(output_df)
        
        output$batch_results <- renderDT({
          datatable(
            output_df,
            options = list(pageLength = 20, scrollX = TRUE, dom = 'Bfrtip'),
            class = "display cell-border stripe",
            rownames = FALSE
          ) %>%
            formatRound(columns = which(sapply(output_df, is.numeric)), digits = 2)
        })
        
        showNotification("✅ Batch prediction completed!", type = "message", duration = 3)
        
      }, error = function(e) {
        showNotification(paste("❌ Error:", e$message), type = "error", duration = 5)
      })
    })
    
    # 下载批量结果
    output$download_batch <- downloadHandler(
      filename = function() {
        paste0("batch_predictions_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv")
      },
      content = function(file) {
        req(batch_results_data())
        write.csv(batch_results_data(), file, row.names = FALSE)
      }
    )
  }
  
  # 返回 Shiny App
  shinyApp(ui = ui, server = server)
}

# ============================================
# 快速启动函数
# ============================================

#' 一键部署 Caret 模型到 Web 应用
#' 
#' @param caret_model 训练好的 caret 模型对象
#' @param title 应用标题（可选，留空则提示用户输入）
#' @param port 端口号（可选，留空则提示用户输入）
#' @param auto_launch 是否自动打开浏览器
#' @export
deploy_model <- function(caret_model, title = NULL, port = NULL, auto_launch = TRUE) {
  
  # 交互式输入标题
  if (is.null(title) || nchar(trimws(title)) == 0) {
    cat("\n========================================\n")
    cat("📝 请输入应用标题\n")
    cat("========================================\n")
    title <- readline(prompt = "应用标题 (默认: Model Deployment): ")
    if (nchar(trimws(title)) == 0) {
      title <- "Model Deployment"
    }
  }
  
  # 交互式输入端口
  if (is.null(port)) {
    cat("\n========================================\n")
    cat("🌐 请输入端口号\n")
    cat("========================================\n")
    port_input <- readline(prompt = "端口号 (默认: 8888): ")
    if (nchar(trimws(port_input)) == 0) {
      port <- 8888
    } else {
      port <- as.integer(port_input)
      if (is.na(port) || port < 1024 || port > 65535) {
        cat("⚠️  端口号无效，使用默认端口 8888\n")
        port <- 8888
      }
    }
  }
  
  # 询问是否自动打开浏览器
  if (interactive()) {
    cat("\n========================================\n")
    cat("🚀 准备部署模型\n")
    cat("========================================\n")
    cat(sprintf("标题: %s\n", title))
    cat(sprintf("端口: %d\n", port))
    cat(sprintf("地址: http://localhost:%d\n", port))
    cat("========================================\n")
    
    launch_input <- readline(prompt = "是否自动打开浏览器? (Y/n): ")
    if (tolower(trimws(launch_input)) == "n") {
      auto_launch <- FALSE
    }
  }
  
  cat("\n🚀 正在部署模型到 Web 应用...\n")
  cat("📊 分析模型结构中...\n")
  
  tryCatch({
    app <- create_smart_app(caret_model, app_title = title)
    
    cat("✅ 模型部署成功!\n")
    cat(sprintf("🌐 访问地址: http://localhost:%d\n", port))
    cat("📌 按 Ctrl+C 或 Esc 停止应用\n\n")
    
    runApp(app, port = port, launch.browser = auto_launch)
    
  }, error = function(e) {
    cat("\n❌ 部署模型时出错:\n")
    cat(e$message, "\n")
    cat("\n请检查:\n")
    cat("  1. 模型是否为有效的 caret train 对象\n")
    cat("  2. 必需的包是否已安装\n")
    cat(sprintf("  3. 端口 %d 是否已被占用\n", port))
    cat("\n💡 提示: 尝试使用不同的端口号\n")
  })
}

# ============================================
# 导出为独立 Shiny App 文件
# ============================================

#' 导出 Shiny App 到文件夹（用于部署到 shinyapps.io）
#' 
#' @param caret_model 训练好的 caret 模型对象
#' @param output_dir 输出文件夹路径
#' @param app_title 应用标题
#' @export
export_app <- function(caret_model, output_dir = "shiny_app", app_title = "Model Deployment") {
  
  cat("\n========================================\n")
  cat("📦 导出 Shiny App 用于在线部署\n")
  cat("========================================\n")
  
  # 创建输出目录
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # 保存模型
  model_file <- file.path(output_dir, "model.rds")
  saveRDS(caret_model, model_file)
  cat(sprintf("✅ 模型已保存: %s\n", model_file))
  
  # 创建 app.R 文件
  app_code <- sprintf('
# ============================================
# Shiny App for Model Deployment
# Auto-generated by smart_caret_deploy
# ============================================

library(shiny)
library(caret)
library(DT)
library(plotly)
library(bslib)

# 加载模型
model <- readRDS("model.rds")

# 创建模型分析器
analyzer <- list()
analyzer$model <- model
analyzer$model_info <- list()

# 提取模型信息
m <- model
analyzer$model_info$method <- m$method
analyzer$model_info$model_type <- m$modelType

train_data <- m$trainingData
outcome_col <- which(names(train_data) == ".outcome")
feature_cols <- setdiff(1:ncol(train_data), outcome_col)

analyzer$model_info$features <- names(train_data)[feature_cols]
analyzer$model_info$feature_types <- sapply(train_data[, feature_cols, drop = FALSE], class)

if (m$modelType == "Classification") {
  analyzer$model_info$classes <- levels(train_data$.outcome)
  analyzer$model_info$n_classes <- length(analyzer$model_info$classes)
}

analyzer$model_info$feature_ranges <- lapply(analyzer$model_info$features, function(feat) {
  val <- train_data[[feat]]
  if (is.numeric(val)) {
    list(type = "numeric", min = min(val, na.rm = TRUE), max = max(val, na.rm = TRUE),
         mean = mean(val, na.rm = TRUE), median = median(val, na.rm = TRUE))
  } else if (is.factor(val)) {
    list(type = "factor", levels = levels(val))
  } else {
    list(type = "character", unique_values = unique(as.character(val)))
  }
})
names(analyzer$model_info$feature_ranges) <- analyzer$model_info$features
analyzer$model_info$results <- m$results
analyzer$model_info$best_tune <- m$bestTune

# 预测函数
analyzer$smart_predict <- function(input_data, type = "raw") {
  if (!is.data.frame(input_data)) {
    input_data <- as.data.frame(input_data, stringsAsFactors = FALSE)
  }
  required_features <- analyzer$model_info$features
  if (!all(required_features %%in%% names(input_data))) {
    missing <- setdiff(required_features, names(input_data))
    stop(paste("Missing required features:", paste(missing, collapse = ", ")))
  }
  input_data <- input_data[, required_features, drop = FALSE]
  
  if (type == "prob" && analyzer$model_info$model_type == "Classification") {
    pred_class <- predict(analyzer$model, input_data, type = "raw")
    pred_prob <- predict(analyzer$model, input_data, type = "prob")
    return(list(prediction = as.character(pred_class), probabilities = pred_prob, input = input_data))
  } else {
    pred <- predict(analyzer$model, input_data)
    return(list(prediction = as.numeric(pred), input = input_data))
  }
}

info <- analyzer$model_info

# UI
ui <- fluidPage(
  theme = bs_theme(version = 5, bootswatch = "flatly"),
  titlePanel("%s"),
  sidebarLayout(
    sidebarPanel(width = 3,
      h4("📊 Model Information"),
      tags$div(style = "background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;",
        tags$strong("Method: "), info$method, tags$br(),
        tags$strong("Type: "), info$model_type, tags$br(),
        if (info$model_type == "Classification") {
          tagList(tags$strong("Classes: "), paste(info$classes, collapse = ", "))
        }
      ),
      hr(),
      h4("🎯 Make Prediction"),
      lapply(info$features, function(feat) {
        range_info <- info$feature_ranges[[feat]]
        if (range_info$type == "numeric") {
          sliderInput(paste0("input_", make.names(feat)), feat,
            min = range_info$min, max = range_info$max, value = range_info$median,
            step = (range_info$max - range_info$min) / 100)
        } else if (range_info$type == "factor") {
          selectInput(paste0("input_", make.names(feat)), feat, choices = range_info$levels)
        } else {
          textInput(paste0("input_", make.names(feat)), feat, value = range_info$unique_values[1])
        }
      }),
      hr(),
      actionButton("predict_btn", "🚀 Predict", class = "btn-primary btn-lg btn-block")
    ),
    mainPanel(width = 9,
      uiOutput("prediction_output"),
      br(),
      conditionalPanel(condition = "output.show_prob_plot",
        plotlyOutput("prob_plot", height = "300px"))
    )
  )
)

# Server
server <- function(input, output, session) {
  get_input_data <- reactive({
    input_list <- list()
    for (feat in info$features) {
      input_id <- paste0("input_", make.names(feat))
      val <- input[[input_id]]
      if (info$feature_ranges[[feat]]$type == "numeric") {
        input_list[[feat]] <- as.numeric(val)
      } else {
        input_list[[feat]] <- val
      }
    }
    as.data.frame(input_list, stringsAsFactors = FALSE)
  })
  
  prediction_result <- eventReactive(input$predict_btn, {
    tryCatch({
      input_data <- get_input_data()
      analyzer$smart_predict(input_data, type = "prob")
    }, error = function(e) {
      list(error = e$message)
    })
  })
  
  output$prediction_output <- renderUI({
    req(prediction_result())
    result <- prediction_result()
    if (!is.null(result$error)) {
      return(tags$div(class = "alert alert-danger", icon("exclamation-triangle"), " Error: ", result$error))
    }
    if (info$model_type == "Classification") {
      pred_class <- result$prediction
      probs <- result$probabilities
      max_prob <- max(probs[1,])
      tagList(
        tags$div(style = "background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%); 
                 color: white; padding: 30px; border-radius: 10px; text-align: center;",
          h2(style = "margin: 0; font-weight: bold;", "Prediction"),
          h1(style = "margin: 10px 0; font-size: 3em;", pred_class),
          h4(style = "margin: 0; opacity: 0.9;", sprintf("Confidence: %%.2f%%%%", max_prob * 100))
        )
      )
    } else {
      pred_value <- result$prediction[1]
      tags$div(style = "background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%); 
               color: white; padding: 30px; border-radius: 10px; text-align: center;",
        h2(style = "margin: 0;", "Prediction Result"),
        h1(style = "font-size: 3em; margin: 10px 0;", format(round(pred_value, 2), big.mark = ","))
      )
    }
  })
  
  output$show_prob_plot <- reactive({
    result <- prediction_result()
    !is.null(result) && is.null(result$error) && info$model_type == "Classification"
  })
  outputOptions(output, "show_prob_plot", suspendWhenHidden = FALSE)
  
  output$prob_plot <- renderPlotly({
    req(prediction_result())
    result <- prediction_result()
    if (info$model_type == "Classification" && is.null(result$error)) {
      probs <- result$probabilities
      plot_ly(x = names(probs), y = as.numeric(probs[1,]), type = "bar",
        marker = list(color = c("#667eea", "#764ba2", "#f093fb")),
        text = sprintf("%%.2f%%%%", as.numeric(probs[1,]) * 100), textposition = "outside") %%>%%
        layout(title = "Prediction Probabilities", xaxis = list(title = "Class"),
               yaxis = list(title = "Probability", range = c(0, 1.1)))
    }
  })
}

shinyApp(ui = ui, server = server)
', app_title)
  
  app_file <- file.path(output_dir, "app.R")
  writeLines(app_code, app_file)
  cat(sprintf("✅ App 文件已创建: %s\n", app_file))
  
  # 创建 README
  readme <- sprintf('
# %s

这是一个自动生成的 Shiny 应用，用于部署 Caret 模型。

## 本地运行

```r
library(shiny)
runApp()
```

## 部署到 shinyapps.io

1. 安装 rsconnect 包:
```r
install.packages("rsconnect")
```

2. 配置账户 (首次使用):
   - 访问 https://www.shinyapps.io/
   - 注册/登录账户
   - 点击 Account -> Tokens -> Show
   - 复制 rsconnect 命令并在 R 中运行

3. 部署应用:
```r
library(rsconnect)
deployApp(appDir = ".")
```

## 文件说明

- `app.R` - Shiny 应用主文件
- `model.rds` - 训练好的模型文件
- `README.md` - 说明文档

## 依赖包

- shiny
- caret
- DT
- plotly
- bslib
', app_title)
  
  readme_file <- file.path(output_dir, "README.md")
  writeLines(readme, readme_file)
  cat(sprintf("✅ README 已创建: %s\n", readme_file))
  
  cat("\n========================================\n")
  cat("✅ 导出完成!\n")
  cat("========================================\n")
  cat(sprintf("📁 输出目录: %s\n", normalizePath(output_dir)))
  cat("\n📋 下一步操作:\n")
  cat("1. 本地测试:\n")
  cat(sprintf("   setwd(\"%s\")\n", output_dir))
  cat("   library(shiny)\n")
  cat("   runApp()\n\n")
  cat("2. 部署到 shinyapps.io:\n")
  cat("   install.packages(\"rsconnect\")\n")
  cat("   library(rsconnect)\n")
  cat(sprintf("   deployApp(appDir = \"%s\")\n\n", output_dir))
  cat("========================================\n")
  
  invisible(output_dir)
}

# ============================================
# 使用示例
# ============================================

cat("\n========================================\n")
cat("✨ 智能化 Caret 模型部署系统已加载！\n")
cat("========================================\n\n")

cat("📖 快速开始指南:\n\n")

cat("【方式1】本地运行 - 交互式部署（推荐新手）:\n")
cat("   data(iris)\n")
cat("   model <- train(Species ~ ., iris, method = 'rf')\n")
cat("   deploy_model(model)  # 系统会提示输入标题和端口\n\n")

cat("【方式2】本地运行 - 直接指定参数:\n")
cat("   deploy_model(model, title = 'Iris分类器', port = 8888)\n\n")

cat("【方式3】在线部署 - 导出并上传到 shinyapps.io:\n")
cat("   # 步骤1: 导出应用文件\n")
cat("   export_app(model, output_dir = 'my_app', app_title = 'Iris分类器')\n\n")
cat("   # 步骤2: 安装 rsconnect 并配置账户\n")
cat("   install.packages('rsconnect')\n")
cat("   library(rsconnect)\n")
cat("   # 访问 https://www.shinyapps.io/ 注册账户\n")
cat("   # 在 Account -> Tokens 获取配置命令\n\n")
cat("   # 步骤3: 部署到云端\n")
cat("   deployApp(appDir = 'my_app')\n\n")

cat("========================================\n")
cat("💡 关键功能:\n")
cat("   • deploy_model()  - 本地运行应用\n")
cat("   • export_app()    - 导出文件用于在线部署\n")
cat("========================================\n")
cat("🌐 在线部署平台:\n")
cat("   • shinyapps.io (免费 5 个应用)\n")
cat("   • Shiny Server (自建服务器)\n")
cat("   • RStudio Connect (企业级)\n")
cat("========================================\n\n")