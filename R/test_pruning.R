source("R/GierigesVerf_Regression.R")

set.seed(1)
n <- 150
X <- matrix(runif(n * 2, -5, 5), ncol = 2)
y <- sin(X[, 1]) + cos(X[, 2]) + rnorm(n, 0, 0.5)

tree_model <- fit_greedy_cart_regression(X, y, max_splits = 20)

cat("Baum fertig. Anzahl Knoten:", length(tree_model$nodes), "\n")

pruning_result <- cost_complexity_sequence(tree_model$nodes, y)

cat("Anzahl der Teilbäume in der Sequenz:", length(pruning_result$trees), "\n")
cat("Werte für Lambda:", round(pruning_result$lambdas, 4), "\n")

