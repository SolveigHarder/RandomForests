source("R/GierigesVerf_Regression.R")
source("R/Pruning.R")

# Originale Grafikparameter speichern, damit wir sie später zurücksetzen können
.pardefault <- par(no.readonly=T)

# Siehe Abbildung 6.2 (Seite 166) im Buch
#' plot regression fit
#' @export
plot_regression_fit <- function(tree, X, y, title,
                                original_function, show_original_func) {
  stopifnot("Eingabe muss 1D sein" = ncol(X) == 1)
  x_label <- names(X)[1]
  x_vals <- X[[1]]

  # Graue Punkte für Daten
  plot(x_vals, y,
       main = title,
       xlab = x_label, ylab = "y",
       pch = 19, col = "gray",
       cex = 0.8)

  x_grid <- seq(min(x_vals), max(x_vals), length.out = 5000)
  X_grid <- data.frame(x = x_grid)

  # Originale Funktion (ohne Rauschen) als Referenz plotten
  if (show_original_func) {
    y_orig <- original_function(x_grid)
    lines(x_grid, y_orig, col = "gray", lwd = 2)
  }

  # Vorhersagefunktion stufenartig plotten
  y_pred <- predict(tree, X_grid)
  r <- rle(y_pred)
  end_idx   <- cumsum(r$lengths)
  start_idx <- c(1, head(end_idx, -1) + 1)

  segments(x0 = x_grid[start_idx],
           y0 = r$values,
           x1 = x_grid[end_idx],
           y1 = r$values,
           col = "red", lwd = 3)

  # Sammle die Splits und plotte
  splits <- c()
  collect_splits <- function(node_id) {
    nd <- tree$nodes[[node_id]]
    if (!nd$is_leaf) {
      splits <<- c(splits, nd$s)
      collect_splits(nd$left)
      collect_splits(nd$right)
    }
  }
  collect_splits(1) # starte von der Wurzel
  abline(v = splits, col = "blue", lty = 2)
}

plot_using_scatter_function <- function(title, min_x = -1, max_x = 1,
                                        input_func, show_original_func = TRUE) {
  n <- 60
  x <- sort(runif(n, min_x, max_x))

  # Punkte mit Rauschen erzeugen
  y <- input_func(x) + rnorm(n, sd = 0.2)
  X <- data.frame(x = x)

  # Fit
  fit <- fit_greedy_cart_regression(X, y, max_splits = .Machine$integer.max,
                                    min_improve = 0, min_leaf_size = 1, print_splits = FALSE)

  # Pruning
  pruning_seq <- cost_complexity_sequence(fit$nodes, y, "regression")

  cv_result <- cv_optimal_lambda(X, y, fit, pruning_seq, mode = "regression", M = 5, max_splits = 10^12)
  best_lambda <- cv_result$best_lambda
  best_tree_nodes <- cv_result$best_tree

  cat("Optimales Lambda:", best_lambda, "\n")

  # Den besten Baum direkt extrahieren:
  fit_pruned <- structure(list(nodes = best_tree_nodes), class = "greedy_cart_reg")

  # Grafik: 2 Spalten (mfrow) und Bottom-Margin (oma) für die Legende
  par(mfrow = c(1, 2), oma = c(2, 0, 0, 0))

  plot_regression_fit(fit, X, y, paste(title, "\n(Ungestutzt, voll ausgewachsen)"), input_func, show_original_func)
  plot_regression_fit(fit_pruned, X, y, sprintf("%s\n(Gestutzt, Lambda = %.3f)", title, best_lambda), input_func, show_original_func)

  # Legende
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
  plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")

  leg_txt <- c("Daten", "Originale Funktion", "Vorhersage", "Splits")
  leg_col <- c("gray", "gray", "red", "blue")
  leg_lty <- c(NA, 1, 1, 2)
  leg_pch <- c(19, NA, NA, NA)
  leg_lwd <- c(NA, 2, 2, 1)
  keep <- c(TRUE, show_original_func, TRUE, TRUE)

  legend("bottom", legend = leg_txt[keep],
         col = leg_col[keep], lty = leg_lty[keep], pch = leg_pch[keep],
         lwd = leg_lwd[keep], bg = "white", xpd = TRUE, bty = "n")

  # Grafikparameter zurücksetzen
  par(.pardefault)
}

set.seed(1)

#plot_using_scatter_function(
#  "Regression",
#  input_func = function(x) {
#    ifelse(x < -0.2, 1,
#           ifelse(x < 0.4, 3, 0))
#  },
#  show_original_func = FALSE
#)

#plot_using_scatter_function(
#  "Regression - Cosinus",
#  min_x = -pi,
#  max_x = pi,
#  input_func = function(x) { cos(x) }
#)

