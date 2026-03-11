source("R/Bagging_6_25.R")

test_reg <- function(data, title, fit, ...) {
  X <- data$X
  x <- X[[1]]
  y <- data$y
  test <- data$test
  train <- data$train

  fitted <- fit(X[train, , drop=FALSE], y[train], ...)
  yhat <- predict(fitted, X[test, , drop=FALSE])
  mse <- mean((y[test]-yhat)^2)

  plot(x[train], y[train], main=title, col="gray")
  points(x[test], yhat, col="red")

  mse
}

gen_data <- function(f, n, sd, xmin, xmax, test_train_split) {
  x <- sort(runif(n, xmin, xmax))
  y <- f(x) + rnorm(n, sd=sd)
  X <- data.frame(x = x)

  id <- sample.int(n)
  train <- id[1:round(test_train_split*n)]
  test  <- id[(round(test_train_split*n)+1):n]

  list(
    X=X,
    y=y,
    test=test,
    train=train
  )
}


set.seed(123)

f_step <- function(x) ifelse(x < -.66, -1,
                             ifelse(x < 0.33, 0, 1))
f_sin <- function(x) sin(2*pi*x)
f_wiggly <- function(x) sin(4*pi*x) + 0.5*cos(7*pi*x)
n <- 100
sd <- .1
xmin <- 0
xmax <- 1
test_train_split <- .7

x_grid <- seq(xmin, xmax, length.out = 5000)
plot(x_grid, f_wiggly(x_grid), xlab="x", ylab="y", main="f_wiggly")

data <- gen_data(f_wiggly, n, sd, xmin, xmax, test_train_split)

# single
mse <- test_reg(data, "single", fit_greedy_cart_regression, max_splits=20, min_leaf_size=1, print_splits=FALSE)
cat("MSE single", mse, "\n")

for(B in c(1, 5, 20, 100)) {
  mse <- test_reg(data, sprintf("bagging B=%d", B), bagging_regression, B, max_splits=20, min_leaf_size=1, print_splits=FALSE)
  cat("B=", B, "MSE=", mse, "\n")
}
