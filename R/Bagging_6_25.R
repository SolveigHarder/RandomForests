
# Bagging für Regressionsbäume (Definition 6.25)

#
# Implementierung des Bagging-Algorithmus

# Lade die Baum-Funktion vom gierigen Verfahren
source("R/GierigesVerf_Regression.R")


# Hauptfunktion: Bagging für Regression


# - Ziehe B Bootstrap-Samples (mit Zurücklegen)
# - Baue auf jedem Sample einen Regressions-Baum
# - Finale Vorhersage = Mittelwert aller Baum-Vorhersagen
#
# Parameter:
#   X: Matrix oder data.frame mit Features (n x d)
#   y: Vektor mit Zielwerten (Länge n)
#   B: Anzahl Bootstrap-Samples (Standard: 100)
#   ...: Weitere Parameter für fit_greedy_cart_regression (max_splits, min_leaf_size,...)
#
# Rückgabe:
#   Objekt der Klasse "bagging_regression" mit:
#     - trees: Liste der B trainierten Bäume
#     - B: Anzahl der Bäume
#
bagging_regression <- function(X, y, B = 100, ...) {

  # Eingabe-Validierung
  X <- as.data.frame(X)
  n <- nrow(X)

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
    boot_idx <- sample(1:n, n, replace = TRUE)

    # Erstelle Bootstrap-Daten
    X_boot <- X[boot_idx, , drop = FALSE]
    y_boot <- y[boot_idx]

    # Baue Baum auf diesem Bootstrap-Sample
    # f̂ₙ*ᵇ(x) = f̂ₙ(x, X₁*ᵇ, Y₁*ᵇ, ..., Xₙ*ᵇ, Yₙ*ᵇ)
    trees[[b]] <- fit_greedy_cart_regression(X_boot, y_boot, ...)
  }


  # Gib Bagging-Modell zurück
  structure(
    list(
      trees = trees,
      B = B
    ),
    class = "bagging_regression"
  )
}


# Vorhersage-Funktion für Bagging

#
# Implementiert die Aggregation aus Def 6.25:
# f̂ₙᵇᵃᵍᵍ(x) = (1/B) * Σᵇ₌₁ᴮ f̂ₙ*ᵇ(x)
#
# Parameter:
#   object: Bagging-Modell (von bagging_regression)
#   newdata: Matrix/data.frame mit neuen Datenpunkten für Vorhersage
#   ...: Weitere Parameter
#
# Rückgabe:
#   Vektor mit Vorhersagen (Länge = Anzahl Zeilen in newdata)
#
predict.bagging_regression <- function(object, newdata, ...) {

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
