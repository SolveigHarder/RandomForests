# Bagging für Klassifikationsbäume per Majority Vote (Definition 6.26)

source("R/GierigesVerf_Klassifikation.R")

# Hauptfunktion: Bagging für Klassifikation
#
# Parameter:
#   X: Matrix oder data.frame mit Features (n x d)
#   y: Vektor mit Klassenlabels (Länge n)
#   B: Anzahl Bootstrap-Samples (Standard: 100)
#   ...: Weitere Parameter für fit_greedy_cart_classification (max_splits, min_leaf_size,...)
#
# Rückgabe:
#   Objekt der Klasse "bagging_classification" mit:
#     - trees: Liste der B trainierten Bäume
#     - B: Anzahl der Bäume
#     - levels: Klassenlabels
#
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


# Vorhersage-Funktion für Bagging-Klassifikation
#
# Implementiert Majority Vote aus Def 6.26:
# f̂ₙᵇᵃᵍᵍ(x) = argmax_k #{b ∈ {1,...,B} : f̂ₙ*ᵇ(x) = k}
#
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


# Test

set.seed(1)
n <- 150
x <- sort(runif(n, -1, 1))
y <- ifelse(x < -0.2, 1, ifelse(x < 0.4, 2, 3))
flip <- runif(n) < 0.10
y[flip] <- sample(1:3, sum(flip), replace = TRUE)

X <- data.frame(x = x)

set.seed(2)
id <- sample.int(n)
train <- id[1:round(0.7*n)]
test  <- id[(round(0.7*n)+1):n]

# Einzelner Baum
fit_single <- fit_greedy_cart_classification(X[train, , drop=FALSE], y[train],
                                             max_splits = 10, min_leaf_size = 10)
pred_single <- predict(fit_single, X[test, , drop=FALSE])
cat("Einzelner Baum - Fehlerrate:", mean(pred_single != y[test]), "\n")

# Bagging mit verschiedenen B
for (B in c(5, 20, 100)) {
  fit_bag <- bagging_classification(X[train, , drop=FALSE], y[train],
                                    B = B, max_splits = 10, min_leaf_size = 10)
  pred_bag <- predict(fit_bag, X[test, , drop=FALSE])
  cat(sprintf("B=%3d - Fehlerrate: %.4f\n", B, mean(pred_bag != y[test])))
}
