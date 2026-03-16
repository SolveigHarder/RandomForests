#source("R/GierigesVerf_Regression.R")
library(RandomForests)

set.seed(1)
n <- 60
x <- sort(runif(n, -1, 1))
y <- ifelse(x < -0.2, 1,
            ifelse(x < 0.4, 3, 0)) + rnorm(n, sd = 0.2)

X <- data.frame(x = x)
X

#Test Workflow
fit <- fit_greedy_cart_regression(X, y, max_splits = 5, min_leaf_size = 5)

yhat <- predict(fit, X)

# 1) stimmt die Länge?
stopifnot(length(yhat) == length(y))

# 2) sind die Werte endlich (kein NA/NaN/Inf)?
stopifnot(all(is.finite(yhat)))

# 3) schneller Plausibilitätscheck: Trainings-MSE
mean((y - yhat)^2)

#sanity check
length(unique(yhat))
table(yhat)



#TestDaten 2

set.seed(123)
n <- nrow(X)
id <- sample.int(n)
train <- id[1:round(0.7*n)]
test  <- id[(round(0.7*n)+1):n]

fit2 <- fit_greedy_cart_regression(X[train, , drop=FALSE], y[train],
                                   max_splits = 5, min_leaf_size = 5)

pred_test <- predict(fit2, X[test, , drop=FALSE])
mean((y[test] - pred_test)^2)

