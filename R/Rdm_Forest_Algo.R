# Random Forests (Definition 6.52)

#' Hauptfunktion: Random Forests für Regression
#'
#' - Ziehe B Bootstrap-Samples (mit Zurücklegen)
#' - Baue auf jedem Sample einen Regressions-Baum
#' - Finale Vorhersage = Mittelwert aller Baum-Vorhersagen
#'
#' @param Parameter:
#'   X: Matrix oder data.frame mit Features (n x d)
#'   y: Vektor mit Zielwerten (Länge n)
#'   B: Anzahl Bootstrap-Samples (Standard: 100)
#'   ...: Weitere Parameter für fit_greedy_cart_regression (max_splits, min_leaf_size,...)
#'
#' @return Rückgabe:
#'   Objekt der Klasse "random_forest_regression" mit:
#'     - trees: Liste der B trainierten Bäume
#'     - B: Anzahl der Bäume
#'
#' random forest for regression using greedy CART
#' @export
random_forest_regression <- function(X, y, B = 100, mtry = NULL, A_n = NULL, ...) {

  # Eingabe-Validierung
  X <- as.data.frame(X)
  n <- nrow(X)
  #Änderung für Random Forest
  d <- ncol(X)
  if (is.null(A_n)) A_n <- n
  if (A_n < 1 || A_n > n) stop("A_n muss zwischen 1 ind n liegen")
  if (!is.null(mtry) && (mtry < 1 || mtry > d)) stop("mtry muss zwischen 1 und d liegen")


  if (length(y) != n) {
    stop("Länge von y muss gleich Anzahl Zeilen in X sein")
  }

  if (B < 1) {
    stop("B muss mindestens 1 sein")
  }

  # Liste für alle B Bäume
  trees <- list()

  # Hauptschleife: Für b = 1, ..., B
  for (b in 1:B) {

    # Bootstrap-Sample ziehen (mit Zurücklegen)
    # Ziehe zufällig gleichverteilt n Indizes aus {1, ..., n}
    replace_flag <- (A_n == n)
    boot_idx <- sample.int(n, size = A_n, replace = replace_flag)

    # Erstelle Bootstrap-Daten
    X_boot <- X[boot_idx, , drop = FALSE]
    y_boot <- y[boot_idx]

    # Baue Baum auf diesem Bootstrap-Sample
    # f̂ₙ*ᵇ(x) = f̂ₙ(x, X₁*ᵇ, Y₁*ᵇ, ..., Xₙ*ᵇ, Yₙ*ᵇ)
    trees[[b]] <- fit_greedy_cart_regression(X_boot, y_boot, ..., mtry = mtry)
  }


  # Gib Bagging-Modell zurück
  structure(
    list(
      trees = trees,
      B = B
    ),
    class = "random_forest_regression"
  )
}


#' Vorhersage-Funktion für Random Forests
#'
#' Implementiert die Aggregation aus Def 6.25:
#' f̂ₙᵇᵃᵍᵍ(x) = (1/B) * Σᵇ₌₁ᴮ f̂ₙ*ᵇ(x)
#'
#' @param Parameter:
#'   object: Bagging-Modell (von bagging_regression)
#'   newdata: Matrix/data.frame mit neuen Datenpunkten für Vorhersage
#'   ...: Weitere Parameter
#'
#' @return Rückgabe:
#'   Vektor mit Vorhersagen (Länge = Anzahl Zeilen in newdata)
#'
#' @export
predict.random_forest_regression <- function(object, newdata, ...) {

  # Extrahiere Komponenten
  B <- object$B
  trees <- object$trees

  # Eingabe vorbereiten
  newdata <- as.data.frame(newdata)
  n_new <- nrow(newdata)

  # Matrix für alle Vorhersagen
  # Zeilen = Datenpunkte, Spalten = Bäume
  all_predictions <- matrix(0, nrow = n_new, ncol = B)

  # Hole Vorhersage von jedem der B Bäume
  for (b in 1:B) {
    all_predictions[, b] <- predict(trees[[b]], newdata)
  }

  # Aggregation: Mittelwert über alle Bäume (Definition 6.25)
  # f̂ₙᵇᵃᵍᵍ(x) = (1/B) * Σ f̂ₙ*ᵇ(x)
  predictions <- rowMeans(all_predictions)

  return(predictions)
}

#' Hauptfunktion: Random Forests für Klassifikation
#'
#' @param Parameter:
#'   X: Matrix oder data.frame mit Features (n x d)
#'   y: Vektor mit Klassenlabels (Länge n)
#'   B: Anzahl Bootstrap-Samples (Standard: 100)
#'   mtry: Anzahl der zufällig ausgewählten Features pro Split
#'   A_n: Größe des Bootstrap-Samples (Standard: n)
#'   ...: Weitere Parameter für fit_greedy_cart_classification (max_splits, min_leaf_size,...)
#'
#' @return Rückgabe:
#'   Objekt der Klasse "random_forest_classification" mit:
#'     - trees: Liste der B trainierten Bäume
#'     - B: Anzahl der Bäume
#'     - levels: Klassenlabels
#'
#' random forest for classification using greedy CART
#' @export
random_forest_classification <- function(X, y, B = 100, mtry = NULL, A_n = NULL, ...) {

  # Eingabe-Validierung
  X <- as.data.frame(X)
  n <- nrow(X)
  d <- ncol(X)
  y <- factor(y)

  #Änderung für Random Forest
  if (is.null(A_n)) A_n <- n
  if (A_n < 1 || A_n > n) stop("A_n muss zwischen 1 ind n liegen")
  if (!is.null(mtry) && (mtry < 1 || mtry > d)) stop("mtry muss zwischen 1 und d liegen")


  if (length(y) != n) {
    stop("Länge von y muss gleich Anzahl Zeilen in X sein")
  }

  if (B < 1) {
    stop("B muss mindestens 1 sein")
  }

  # Liste für alle B Bäume
  trees <- list()

  # Hauptschleife: Für b = 1, ..., B
  for (b in 1:B) {

    # Bootstrap-Sample ziehen (mit Zurücklegen)
    # Ziehe zufällig gleichverteilt n Indizes aus {1, ..., n}
    replace_flag <- (A_n == n)
    boot_idx <- sample.int(n, size = A_n, replace = replace_flag)

    # Erstelle Bootstrap-Daten
    X_boot <- X[boot_idx, , drop = FALSE]
    y_boot <- y[boot_idx]

    # Baue Klassifikationsbaum auf diesem Bootstrap-Sample
    trees[[b]] <- fit_greedy_cart_classification(X_boot, y_boot, mtry = mtry, ...)
  }

  structure(
    list(
      trees = trees,
      B = B,
      levels = levels(y)
    ),
    class = "random_forest_classification"
  )
}

#' Vorhersage-Funktion für Random Forests-Klassifikation
#' @export
predict.random_forest_classification <- function(object, newdata, ...) {
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
