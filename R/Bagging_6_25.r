
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

cat("---------------\n")



# TEST VON SOLVEIG

test_single_vs_bagging <- function(B, X, y, train, test) {


  # single
  fit <- fit_greedy_cart_regression(X[train, , drop=FALSE], y[train],
                                    max_splits = 5, min_leaf_size = 5, print_splits = FALSE)
  yhat <- predict(fit, X[test, , drop=FALSE])
  mse1 <- mean((y[test] - yhat)^2)

  # bagging
  fitB <- bagging_regression(X[train, , drop=FALSE], y[train],
                             B,
                             max_splits = 5,
                             min_leaf_size = 5, print_splits = FALSE)
  yhat <- predict(fitB, X[test, , drop=FALSE])
  mseB <- mean((y[test]-yhat)^2)

  #cat(name, "\n")
  cat("MSE single", mse1, "\n")
  cat("MSE bagging", mseB, "\n")
  cat("MSE diff bag-single", mse1-mseB, "\n")
}

f_step <- function(x) ifelse(x < -0.2, 1,
                             ifelse(x < 0.4, 3, 0))

f_sin <- function(x) sin(pi*x)

n <- 200
xmin<- -1
xmax<- 1
test_train_split <- .7
set.seed(100)
x <- sort(runif(n, xmin, xmax))
y <- f_sin(x) + rnorm(n, sd = 0.2)
X <- data.frame(x = x)

id <- sample.int(n)
train <- id[1:round(test_train_split*n)]
test  <- id[(round(test_train_split*n)+1):n]

for(B in c(1, 5, 20, 100)) {
  cat("B=", B, "\n")
  test_single_vs_bagging(B, X, y, train, test)
}
