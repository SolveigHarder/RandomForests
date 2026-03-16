# Bagging für Klassifikationsbäume per Majority Vote (Definition 6.26)

source("R/GierigesVerf_Klassifikation.R")
#' Bagging für Klassifikation
#'
#'@param Parameter:
#'   X: Matrix oder data.frame mit Features (n x d)
#'   y: Vektor mit Klassenlabels (Länge n)
#'   B: Anzahl Bootstrap-Samples (Standard: 100)
#'   ...: Weitere Parameter für fit_greedy_cart_classification (max_splits, min_leaf_size,...)
#'@return Rückgabe:
#'   Objekt der Klasse "bagging_classification" mit:
#'     - trees: Liste der B trainierten Bäume
#'     - B: Anzahl der Bäume
#'     - levels: Klassenlabels
#'@export
bagging_classification <- function(X, y, B = 100, ...) {

  X <- as.data.frame(X)
  n <- nrow(X)
  y <- factor(y)

  if (length(y) != n) {
    stop("Länge von y muss gleich Anzahl Zeilen in X sein")
  }

  if (B < 1) {
    stop("B muss mindestens 1 sein")
  }

  trees <- list()

  for (b in 1:B) {

    # Bootstrap-Sample ziehen (mit Zurücklegen)
    boot_idx <- sample(1:n, n, replace = TRUE)

    X_boot <- X[boot_idx, , drop = FALSE]
    y_boot <- y[boot_idx]

    # Baue Klassifikationsbaum auf Bootstrap-Sample
    trees[[b]] <- fit_greedy_cart_classification(X_boot, y_boot, ...)
  }

  structure(
    list(
      trees = trees,
      B = B,
      levels = levels(y)
    ),
    class = "bagging_classification"
  )
}


#' Vorhersage-Funktion für Bagging-Klassifikation
#'
#' Implementiert Majority Vote aus Def 6.26:
#' f̂ₙᵇᵃᵍᵍ(x) = argmax_k #{b ∈ {1,...,B} : f̂ₙ*ᵇ(x) = k}
#'
#' @export
predict.bagging_classification <- function(object, newdata, ...) {

  B <- object$B
  trees <- object$trees
  levs <- object$levels

  newdata <- as.data.frame(newdata)
  n_new <- nrow(newdata)

  # Matrix für alle Vorhersagen (character statt numeric)
  all_predictions <- matrix(NA_character_, nrow = n_new, ncol = B)

  for (b in 1:B) {
    all_predictions[, b] <- as.character(predict(trees[[b]], newdata))
  }

  # Aggregation: Majority Vote pro Datenpunkt (Definition 6.26)
  predictions <- apply(all_predictions, 1, function(row) {
    tab <- table(factor(row, levels = levs))
    names(tab)[which.max(tab)]
  })

  factor(predictions, levels = levs)
}
