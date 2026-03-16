# Bagging für Klassifikationsbäume per Klassen-Wahrscheinlichkeiten (Definition 6.28)
# Benötigt: Def. 6.27 (predict_proba)

#' Definition 6.27: Klassenwahrscheinlichkeiten eines Baums
#'
#' Traversiert den Baum wie predict, gibt aber pro Punkt
#' den Vektor p̂_k(A) = #{i : X_i ∈ A, Y_i = k} / #A zurück.
#'
#'@param Parameter:
#'   object:  Baum-Objekt (greedy_cart_clas)
#'   newdata: Matrix/data.frame mit neuen Datenpunkten
#'   y_train: Trainings-Labels (factor), die zum Fitten benutzt wurden
#'
#'@return Rückgabe:
#' Matrix n_new x K mit Klassenwahrscheinlichkeiten
#'
#'@export
predict_proba <- function(object, newdata, y_train) {

  nodes <- object$nodes
  levs  <- object$levels
  K     <- length(levs)
  X     <- as.matrix(newdata)
  n     <- nrow(X)

  probs <- matrix(0, nrow = n, ncol = K)
  colnames(probs) <- levs

  for (i in seq_len(n)) {
    node_id <- 1L

    repeat {
      nd <- nodes[[node_id]]

      if (nd$is_leaf) {
        # Def. 6.27: p̂_k(A) = #{i : Y_i = k, X_i ∈ A} / #A
        tab <- table(factor(y_train[nd$idx], levels = levs))
        probs[i, ] <- as.numeric(tab) / length(nd$idx)
        break
      }

      if (X[i, nd$j] < nd$s) node_id <- nd$left else node_id <- nd$right
    }
  }

  probs
}


# Definition 6.28: Bagging per Klassen-Wahrscheinlichkeiten
#
# p̂ᵇᵃᵍᵍ_k(x) = (1/B) * Σᵇ₌₁ᴮ p̂*ᵇ_k(x)
# f̂ᵇᵃᵍᵍ(x)   = argmax_k p̂ᵇᵃᵍᵍ_k(x)
#
#' bagging classification probability
#' @export
bagging_classification_proba <- function(X, y, B = 100, ...) {

  X <- as.data.frame(X)
  n <- nrow(X)
  y <- factor(y)

  if (length(y) != n) stop("Länge von y muss gleich Anzahl Zeilen in X sein")
  if (B < 1) stop("B muss mindestens 1 sein")

  trees   <- list()
  y_boots <- list()  # Trainings-y pro Baum speichern (für predict_proba)

  for (b in 1:B) {
    boot_idx <- sample(1:n, n, replace = TRUE)

    X_boot <- X[boot_idx, , drop = FALSE]
    y_boot <- y[boot_idx]

    trees[[b]]   <- fit_greedy_cart_classification(X_boot, y_boot, ...)
    y_boots[[b]] <- y_boot
  }

  structure(
    list(
      trees   = trees,
      y_boots = y_boots,
      B       = B,
      levels  = levels(y)
    ),
    class = "bagging_classification_proba"
  )
}


# Vorhersage: Mittelt Klassenwahrscheinlichkeiten, dann argmax
#

#' @export
predict.bagging_classification_proba <- function(object, newdata, ...) {

  B     <- object$B
  trees <- object$trees
  levs  <- object$levels
  K     <- length(levs)

  newdata <- as.data.frame(newdata)
  n_new   <- nrow(newdata)

  # Summiere Wahrscheinlichkeiten über alle B Bäume
  probs_sum <- matrix(0, nrow = n_new, ncol = K)
  colnames(probs_sum) <- levs

  for (b in 1:B) {
    probs_sum <- probs_sum + predict_proba(trees[[b]], newdata, object$y_boots[[b]])
  }

  # Mitteln: p̂ᵇᵃᵍᵍ_k(x) = (1/B) * Σ p̂*ᵇ_k(x)
  probs_avg <- probs_sum / B

  # argmax: f̂ᵇᵃᵍᵍ(x) = argmax_k p̂ᵇᵃᵍᵍ_k(x)
  predictions <- levs[apply(probs_avg, 1, which.max)]

  factor(predictions, levels = levs)
}
