source("R/GierigesVerf_Regression.R")

# Siehe Abbildung 6.2 (Seite 166) im Buch
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

  # Legende
  leg_txt <- c("Daten", "Originale Funktion", "Vorhersage", "Splits")
  leg_col <- c("gray", "gray", "red", "blue")
  leg_lty <- c(NA, 1, 1, 2)
  leg_pch <- c(19, NA, NA, NA)
  leg_lwd <- c(NA, 2, 2, 1)
  keep <- c(TRUE, show_original_func, TRUE, TRUE)

  legend("topright", legend = leg_txt[keep],
         col = leg_col[keep], lty = leg_lty[keep], pch = leg_pch[keep],
         lwd = leg_lwd[keep], bg = "white")
}

plot_using_scatter_function <- function(title, min_x = -1, max_x = 1,
                                        input_func, show_original_func = TRUE) {
  n <- 60
  x <- sort(runif(n, min_x, max_x))

  # Punkte mit Rauschen erzeugen
  y <- input_func(x) + rnorm(n, sd = 0.2)
  X <- data.frame(x = x)

  # Fit
  fit <- fit_greedy_cart_regression(X, y, max_splits = 5, min_leaf_size = 5)

  plot_regression_fit(fit, X, y, title, input_func, show_original_func)
}

set.seed(1)

plot_using_scatter_function(
  "Gieriges Verfahren (Regression)",
  input_func = function(x) {
    ifelse(x < -0.2, 1,
           ifelse(x < 0.4, 3, 0))
  },
  show_original_func = FALSE
)

plot_using_scatter_function(
  "Gieriges Verfahren - Cosinus (Regression)",
  min_x = -pi,
  max_x = pi,
  input_func = function(x) { cos(x) }
)

