#source("R/Bagging_6_28.R")
library(RandomForests)

# Test
gen_data <- function(f, n, noise, xmin, xmax, test_train_split) {
  x <- sort(runif(n, xmin, xmax))
  y <- f(x)
  flip <- runif(n) < noise
  y[flip] <- sample(1:3, sum(flip), replace = TRUE)

  X <- data.frame(x = x)
  id <- sample.int(n)
  train <- id[1:round(test_train_split*n)]
  test  <- id[(round(test_train_split*n)+1):n]

  list(X=X, y=y, train=train, test=test)
}

set.seed(1)
n <- 500
noise <- .1
xmin <- 0
xmax <- 2*pi
test_train_split <- .7

f_step <- function(x) ifelse(x < -0.2, 1, ifelse(x < 0.4, 2, 3))
f_sin3 <- function(x) ifelse(sin(2*x) > 0.3, 1, ifelse(sin(2*x) > -0.3, 2, 3))

data <- gen_data(f_sin3, n, noise, xmin, xmax, test_train_split)
X <- data$X
y <- data$y
test <- data$test
train <- data$train

# Einzelner Baum
fit_single <- fit_greedy_cart_classification(X[train, , drop=FALSE], y[train],
                                             max_splits = 50, min_leaf_size = 3, print_splits = FALSE)
pred_single <- predict(fit_single, X[test, , drop=FALSE])
cat("Einzelner Baum - Fehlerrate:", mean(pred_single != y[test]), "\n")

# Bagging mit verschiedenen B
for (B in c(5, 20, 100)) {
  fit_bag <- bagging_classification_proba(X[train, , drop=FALSE], y[train],
                                          B = B, max_splits = 50, min_leaf_size = 3, print_splits = FALSE)
  pred_bag <- predict(fit_bag, X[test, , drop=FALSE])
  cat(sprintf("B=%3d - Fehlerrate: %.4f\n", B, mean(pred_bag != y[test])))
  # check
  #single_errors <- sapply(fit_bag$trees, function(tree) {
  #  p <- predict(tree, X[test,,drop=FALSE])
  #  mean(p != y[test])
  #})
  #cat("Individual tree errors:", round(single_errors, 3), "\n")
  #cat("Mean:", mean(single_errors), "\n")
  #p1 <- as.character(predict(fit_bag$trees[[1]], X[test,,drop=FALSE]))
  #p2 <- as.character(predict(fit_single, X[test,,drop=FALSE]))
  #head(cbind(p1, p2, y_true=y[test]), 20)
}
