source("R/GierigesVerf_Klassifikation.R")


#TESTEN

set.seed(1)
n <- 150
x <- sort(runif(n, -1, 1))

# echte Klasse (1/2/3)
y <- ifelse(x < -0.2, 1,
            ifelse(x < 0.4, 2, 3))

# bisschen Label-Noise (10%) also Messfehler
flip <- runif(n) < 0.10
y[flip] <- sample(1:3, sum(flip), replace = TRUE)

X <- data.frame(x = x)

# Train/Test split
set.seed(2)
id <- sample.int(n)
train <- id[1:round(0.7*n)]
test  <- id[(round(0.7*n)+1):n]

fit <- fit_greedy_cart_classification(X[train, , drop=FALSE], y[train],
                                      max_splits = 10, min_leaf_size = 10)

pred_train <- predict(fit, X[train, , drop=FALSE])
pred_test  <- predict(fit, X[test, , drop=FALSE])

# Checks
stopifnot(length(pred_test) == length(test))
stopifnot(!any(is.na(pred_test)))

# Fehlerraten
train_err <- mean(pred_train != y[train])
#Fehlerquote auf den Daten, mit denen wir den Baum trainiert haben
test_err  <- mean(pred_test  != y[test])
#Fehlerquote auf neuen/ungesehenen Daten
train_err
test_err

# Confusion Matrix
table(truth = y[test], pred = pred_test)

