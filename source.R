library(dplyr)
library(ggraph)
library(igraph)

tree_func <- function(final_model, 
                      tree_num) {
  
  # get tree by index
  tree <- randomForest::getTree(final_model, 
                                k = tree_num, 
                                labelVar = TRUE) %>%
    tibble::rownames_to_column() %>%
    # make leaf split points to NA, so the 0s won't get plotted
    mutate(`split point` = ifelse(is.na(prediction), `split point`, NA))
  
  # prepare data frame for graph
  graph_frame <- data.frame(from = rep(tree$rowname, 2),
                            to = c(tree$`left daughter`, tree$`right daughter`))
  
  # convert to graph and delete the last node that we don't want to plot
  graph <- graph_from_data_frame(graph_frame) %>%
    delete_vertices("0")
  
  # set node labels
  V(graph)$node_label <- gsub("_", " ", as.character(tree$`split var`))
  V(graph)$leaf_label <- as.character(tree$prediction)
  V(graph)$split <- as.character(round(tree$`split point`, digits = 2))
  
  # plot
  plot <- ggraph(graph, 'dendrogram') + 
    theme_bw() +
    geom_edge_link() +
    geom_node_point() +
    geom_node_text(aes(label = node_label), na.rm = TRUE, repel = TRUE) +
    geom_node_label(aes(label = split), vjust = 2.5, na.rm = TRUE, fill = "white") +
    geom_node_label(aes(label = leaf_label, fill = leaf_label), na.rm = TRUE, 
                    repel = TRUE, colour = "white", fontface = "bold", show.legend = FALSE) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_blank(),
          panel.background = element_blank(),
          plot.background = element_rect(fill = "white"),
          panel.border = element_blank(),
          axis.line = element_blank(),
          axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          plot.title = element_text(size = 18))
  
  print(plot)
}

#' The last column is used as grouping, and the other columns are subjected to Wilcoxon test
#'
#' @param mat a data frame,the last column is 'group', and the content is binary classification text
#' @param p_value the cut off value of p value
#' @return Wilcoxon test results
#' @import dplyr
#' @export
#'
#' @examples
#' library(Clabomic)
#' colnames(iris)[ncol(iris)] <- "group"
#' iris_batch_wilcoxon_res <- do_batch_Wilcoxon(iris)
do_batch_Wilcoxon <- function(mat,p_value=0.05) {
  # wilcon first step to select factor
  test.fun <- function(dat, col) {
    # dat=mat;col=1
    index <- unique(dat$group)
    sigs <- wilcox.test(
      dat[dat$group == index[1], col],
      dat[dat$group == index[2], col]
    )
    
    tests <- data.frame(
      W = sigs$statistic,
      p = sigs$p.value,
      mean_x = mean(dat[dat$group == index[1], col]),
      mean_y = mean(dat[dat$group == index[2], col]),
      median_x = median(dat[dat$group == index[1], col]),
      median_y = median(dat[dat$group == index[2], col])
    )
    
    return(tests)
  }
  
  # mat=do_process_res[[1]]
  tests <- do.call(rbind, lapply(colnames(mat)[-ncol(mat)], function(x) test.fun(mat, x)))
  rownames(tests) <- colnames(mat)[-ncol(mat)]
  
  test_sig <- tests[tests$p <p_value, ]
  str(test_sig)
  test_sig$p.adjust <- p.adjust(test_sig$p, method = "bonferroni")
  test_sig <- test_sig[order(test_sig$p), ]
  
  ## calculate the sd and logFC
  # mat=do_process_res[[1]]
  sd_file <- mat %>%
    group_by(group) %>%
    summarise_all(sd) %>%
    t(.)
  colnames(sd_file) <- sd_file[1, ]
  sd_file <- as.data.frame(sd_file[-1, ])
  sd_file$id <- rownames(sd_file)
  colnames(sd_file)[1:2] <- paste("sd", colnames(sd_file)[1:2], sep = "_")
  mean_file <- mat %>%
    group_by(group) %>%
    summarise_all(mean) %>%
    select(-group) %>%
    log2(.)
  logFC <- mean_file[2, ] - mean_file[1, ]
  logFC <- as.data.frame(t(logFC))
  logFC$id <- rownames(logFC)
  colnames(logFC)[1] <- "logFC"
  
  rownames(test_sig)
  test_sig$id <- rownames(test_sig)
  last_test_sig <- merge(test_sig, sd_file, by = "id")
  last_test_sig <- merge(last_test_sig, logFC, by = "id")
  last_test_sig <- last_test_sig[order(last_test_sig$p), ]
  return(last_test_sig)
}


#' The last column is used as grouping, and the other columns are subjected to roc test
#'
#' @param dat a data frame,the last column is 'group', and the content is binary classification text
#' @import pROC
#' @return roc test results
#' @export
#'
#' @examples
#' library(Clabomic)
#' colnames(iris)[ncol(iris)] <- "group"
#' res <- do_batch_roc(iris)
do_batch_roc <- function(dat) {
  index <- colnames(dat)
  res <- c()
  for (i in 1:(ncol(dat) - 1)) {
    # i=1
    tem_dat <- dat[, c(i, ncol(dat))]
    colnames(tem_dat)[1] <- "feature"
    tem_dat$feature <- as.numeric(tem_dat$feature)
    tem_roc <- roc(group ~ feature, data = tem_dat, ci = T)
    res[index[i]] <- round(tem_roc$auc, 2)
  }
  return(as.data.frame(res))
}

