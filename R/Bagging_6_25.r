
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

  cat("=== Bagging für Regression ===\n")
  cat("Anzahl Bootstrap-Samples (B):", B, "\n")
  cat("Anzahl Trainingspunkte (n):", n, "\n\n")

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

    # Fortschrittsanzeige
    if (b %% 10 == 0 || b == B) {
      cat(sprintf("  Bootstrap-Sample %d/%d abgeschlossen\n", b, B))
    }
  }

  cat("\n=== Bagging Training abgeschlossen ===\n\n")

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


# Print-Methode für Bagging-Modell

print.bagging_regression <- function(x, ...) {
  cat("Bagging Regression Modell \n")
  cat("============================================\n")
  cat("Anzahl Bäume (B):", x$B, "\n")
  cat("\nNutze predict() für Vorhersagen auf neuen Daten\n")
}


#
# BEISPIEL / TEST
#
# Testdaten
set.seed(123)
n <- 60
x <- sort(runif(n, -1, 1))
y <- ifelse(x < -0.2, 1, ifelse(x < 0.4, 3, 0)) + rnorm(n, sd = 0.2)
X <- data.frame(x = x)

# Train/Test Split
train_idx <- 1:40
test_idx <- 41:60

X_train <- X[train_idx, , drop = FALSE]
y_train <- y[train_idx]
X_test <- X[test_idx, , drop = FALSE]
y_test <- y[test_idx]

cat("\n=== BEISPIEL: Vergleich Einzelner Baum vs. Bagging ===\n\n")


# EINZELNER BAUM

cat("1. Trainiere einzelnen Baum...\n")
single_tree <- fit_greedy_cart_regression(X_train, y_train,
                                          max_splits = 5,
                                          min_leaf_size = 5)

pred_single_test <- predict(single_tree, X_test)
mse_single <- mean((y_test - pred_single_test)^2)

cat("   MSE (Test) einzelner Baum:", round(mse_single, 4), "\n\n")


# BAGGING

cat("2. Trainiere Bagging mit B=50...\n")
bagging_model <- bagging_regression(X_train, y_train,
                                    B = 50,
                                    max_splits = 5,
                                    min_leaf_size = 5)

pred_bagging_test <- predict(bagging_model, X_test)
mse_bagging <- mean((y_test - pred_bagging_test)^2)

cat("   MSE (Test) Bagging:", round(mse_bagging, 4), "\n\n")


# VERGLEICH

cat("=== ERGEBNISSE ===\n")
cat("MSE einzelner Baum:", round(mse_single, 4), "\n")
cat("MSE Bagging:       ", round(mse_bagging, 4), "\n")

improvement <- (mse_single - mse_bagging) / mse_single * 100
cat("\nVerbesserung durch Bagging:", round(improvement, 1), "%\n")

if (improvement > 0) {
  cat("→ Bagging reduziert den Fehler\n")
} else {
  cat("→ In diesem Fall war der einzelne Baum besser.\n")
}
