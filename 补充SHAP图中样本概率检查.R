# 检查这个样本的详细信息
check_sample <- function(sample_idx) {
  cat("样本索引:", sample_idx, "\n")
  cat("真实标签:", actual_labels[sample_idx], "\n")
  cat("预测概率:", pred_probs[sample_idx], "\n")
  
  # 检查最重要的特征值
  shap_row <- shap_values[sample_idx, ]
  top_idx <- order(abs(shap_row), decreasing = TRUE)[1:10]
  
  cat("\n最重要的10个特征:\n")
  for(i in 1:10) {
    idx <- top_idx[i]
    cat(sprintf("  %s: SHAP=%+.3f, Value=%.2f\n",
                colnames(shap_values)[idx],
                shap_row[idx],
                explain_data[sample_idx, idx]))
  }
}

# 调用检查函数
check_sample(explain_idx)
