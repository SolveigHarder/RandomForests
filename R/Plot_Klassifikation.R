source("R/GierigesVerf_Klassifikation.R")
source("R/Pruning.R")

# Originale Grafikparameter speichern
.pardefault <- par(no.readonly = TRUE)

#' Funktion zum Plotten eines einzelnen Klassifikationsbaums (2D)
#' plot classification fit
#' @export
plot_classification_fit <- function(tree, X, y, title, input_func = NULL, res = 200) {
  stopifnot("Eingabe für Klassifikationsplot muss 2D sein" = ncol(X) == 2)

  y <- as.factor(y)
  n_classes <- length(levels(y))

  x1_name <- names(X)[1]
  x2_name <- names(X)[2]
  x1_vals <- X[[1]]
  x2_vals <- X[[2]]

  # Achsenlimits bestimmen
  x1_lim <- c(min(x1_vals), max(x1_vals))
  x2_lim <- c(min(x2_vals), max(x2_vals))

  # Dichtes Raster für die farbigen Hinterlegungen (Entscheidungsregel) erstellen
  x1_grid <- seq(x1_lim[1], x1_lim[2], length.out = res)
  x2_grid <- seq(x2_lim[1], x2_lim[2], length.out = res)

  grid_df <- expand.grid(x1_grid, x2_grid)
  names(grid_df) <- c(x1_name, x2_name)

  # Vorhersage für das Raster
  preds <- predict(tree, grid_df)
  z <- matrix(as.numeric(preds), nrow = res, ncol = res)

  bg_colors <- c("#FFAAAA", "#AAAAFF", "#AAFFAA")[1:n_classes]
  pt_colors <- c("red", "blue", "darkgreen")[1:n_classes]

  plot(x1_vals, x2_vals, type = "n",
       xlab = "X1", ylab = "X2", main = title,
       xlim = x1_lim, ylim = x2_lim)

  # Farbige Klassenregionen
  image(x1_grid, x2_grid, z, col = bg_colors, add = TRUE,
        breaks = seq(0.5, n_classes + 0.5, by = 1))

  # Originale Funktion zum Vergleich
  if (!is.null(input_func)) {
    lines(x1_grid, input_func(x1_grid), col = "black", lwd = 2, lty = 2)
  }

  # Datenpunkte
  points(x1_vals, x2_vals, col = pt_colors[as.numeric(y)], pch = 1, lwd = 1, cex = 1)
}


plot_classification_comparison <- function(fit, pruning_seq, X, y, title, input_func) {
  cv_result <- cv_optimal_lambda(X, y, fit, pruning_seq, mode = "classification", M = 5, max_splits = 10^12)
  best_lambda <- cv_result$best_lambda
  best_tree_nodes <- cv_result$best_tree

  cat("Optimales Lambda:", best_lambda, "\n")

  fit_pruned <- structure(list(nodes = best_tree_nodes, levels = fit$levels),
                          class = "greedy_cart_clas")
  par(mfrow = c(1, 2), oma = c(3, 0, 0, 0))

  # Plots generieren
  plot_classification_fit(fit, X, y, paste(title, "\n(Voll ausgewachsen)"), input_func)
  plot_classification_fit(fit_pruned, X, y, sprintf("%s\n(Gestutzt, Lambda = %.3f)", title, best_lambda), input_func)

  # Gemeinsame Legende unten
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
  plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")

  n_classes <- length(levels(as.factor(y)))
  pt_colors <- c("red", "blue", "darkgreen")[1:n_classes]

  legend("bottom",
         legend = c(paste("Klasse", levels(as.factor(y))), "Optimaler Entscheidungsrand"),
         col = c(pt_colors, "black"),
         pch = c(rep(1, n_classes), NA),
         lty = c(rep(NA, n_classes), 2),
         lwd = c(rep(1, n_classes), 2),
         horiz = TRUE, bty = "n", cex = 1.1)

  par(.pardefault)
}

# Generiere Daten mit Input-Funktion und plotte
plot_classification_using_scatter_function <- function(title, min_x, max_x, input_func) {
  set.seed(42)
  n <- 200

  # Wertebereich der input_func
  grid_vals <- input_func(seq(min_x, max_x, length.out = 1000))
  min_y <- min(grid_vals) - 0.5
  max_y <- max(grid_vals) + 0.5

  # 2D-Raum aufbauen
  X1 <- runif(n, min_x, max_x)
  X2 <- runif(n, min_y, max_y)
  X <- data.frame(X1 = X1, X2 = X2)

  # Klasse 1: Unterhalb der Funktion, Klasse 2: Oberhalb der Funktion
  y_true <- ifelse(X2 < input_func(X1), 1, 2)

  # Ungenauigkeit hinzufügen
  flip <- runif(n) < 0.07
  y_true[flip] <- ifelse(y_true[flip] == 1, 2, 1)

  y_factor <- as.factor(y_true)

  print("Fit Root")

  # Trainieren des vollen Baums
  fit <- fit_greedy_cart_classification(X, y_factor, max_splits = .Machine$integer.max,
                                        min_improve = 0, min_leaf_size = 1, print_splits = FALSE)

  print("CCS Root")
  pruning_seq <- cost_complexity_sequence(fit$nodes, y_factor, mode = "classification")

  plot_classification_comparison(fit, pruning_seq, X, y_factor, title, input_func)

  print("Done")
}

#plot_classification_using_scatter_function(
#  "Klassifikation",
#  input_func = function(x) {
#    ifelse(x < -0.2, 1,
#           ifelse(x < 0.4, 3, 0))
#  },
#  min_x = -1,
#  max_x = 1
#)

#plot_classification_using_scatter_function(
#  "Klassifikation - Sinus",
#  min_x = -pi,
#  max_x = pi,
#  input_func = function(x) { sin(x) }
#)
