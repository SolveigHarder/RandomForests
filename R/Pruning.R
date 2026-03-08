# Finde alle Indizes der Blätter unterhalb eines bestimmten Knoten
get_leaves_of_subtree <- function(nodes, node_idx) {
  leaves <- c() # Indizes der Blätter
  stack <- c(node_idx) # Starte mit node_idx

  while (length(stack) > 0) {
    current_idx <- stack[length(stack)]
    stack <- stack[-length(stack)]

    node <- nodes[[current_idx]]

    if (node$is_leaf) {
      leaves <- c(leaves, current_idx)
    } else {
      # Innerer Knoten -> lege Kinder auf den Stack
      stack <- c(stack, node$right, node$left)
    }
  }

  return(leaves)
}

# Berechnet den Trainingsfehler eines Knotens abhängig vom Modus
get_node_error <- function(node, y, mode = "regression") {
  idx <- node$idx


  if (mode == "regression") {
    if (length(idx) <= 1) return(0)
    mu <- mean(y[idx])
    return(sum((y[idx] - mu)^2) / length(y))

  } else if (mode == "classification") {
    if (length(idx) == 0) return(0)
    # Mehrheitsklasse bestimmen
    tab <- table(y[idx])
    pred_class <- names(tab)[which.max(tab)]
    return(sum(y[idx] != pred_class) / length(y))

  } else {
    stop("mode value is invalid")
  }
}

# Berechnet den Trainingsfehler eines Teilbaums R(T_t)
get_subtree_error <- function(nodes, node_idx, y, mode) {
  leaf_indices <- get_leaves_of_subtree(nodes, node_idx)
  total <- 0
  for (l_idx in leaf_indices) {
    total <- total + get_node_error(nodes[[l_idx]], y, mode)
  }
  return(total)
}

# Findet und eliminiert den Weakest Link
prune_weakest_link <- function(nodes, y, mode) {
  values <- rep(Inf, length(nodes))

  for (i in seq_along(nodes)) {
    node <- nodes[[i]]

    # Berechne nur für existierende innere Knoten
    if (!is.null(node) &&!node$is_leaf) {
      leaves_in_subtree <- get_leaves_of_subtree(nodes, i)
      # R_t = R_n(f_T)
      R_t <- get_node_error(node, y, mode)
      # R_T_t = R_n(f_T^(p-1))
      R_T_t <- get_subtree_error(nodes, i, y, mode)
      # N_T_t = #T^(p-1) - 1
      # N_T_t = Anzahl Blätter im Subtree, der abgeschnitten wird
      N_T_t <- length(leaves_in_subtree)

      stopifnot("Ein innerer Knoten muss mindestens zwei Blätter haben" = N_T_t > 1)
      values[i] <- (R_t - R_T_t) / (N_T_t - 1)
    }
  }

  # arg_min { ... }
  weakest_link_idx <- which.min(values)

  # Abbrechen wenn keine prunbaren Knoten mehr existieren
  if (length(weakest_link_idx) == 0 || is.infinite(values[weakest_link_idx])) {
    return(list(nodes = nodes, lambda = Inf, pruned = FALSE))
  }

  min_lambda <- values[weakest_link_idx]

  # Abschneiden: Mache den inneren Knoten zu einem Blatt
  nodes[[weakest_link_idx]]$is_leaf <- TRUE
  nodes[[weakest_link_idx]]$left <- NULL
  nodes[[weakest_link_idx]]$right <- NULL
  # Die Kindsknoten werden abgeschnitten, befinden sich (und ihre Kinder) aber
  # weiterhin im nodes-Vektor. Das hat keine Auswirkung auf den Hauptbaum.

  # Setze den Vorhersagewert (pred) für das neue Blatt
  idx <- nodes[[weakest_link_idx]]$idx
  if (length(idx) > 0) {
    if (mode == "regression") {
      nodes[[weakest_link_idx]]$pred <- mean(y[idx])
    } else {
      tab <- table(y[idx])
      nodes[[weakest_link_idx]]$pred <- names(tab)[which.max(tab)]
    }
  }

  return(list(nodes = nodes, lambda = min_lambda, pruned = TRUE))
}

# Generiert die gesamte Sequenz der gestutzten Bäume
cost_complexity_sequence <- function(initial_nodes, y, mode) {
  stopifnot("Parameter 'mode' muss 'regression' oder 'classification' enthalten" = mode %in% c("regression", "classification"))

  tree_sequence <- list()
  lambda_sequence <- numeric()

  current_nodes <- initial_nodes

  # Startzustand
  tree_sequence[[1]] <- current_nodes
  lambda_sequence[1] <- 0

  step <- 2
  repeat {
    # Abbrechen, wenn der Baum bis auf die Wurzel geprunt wurde
    if (current_nodes[[1]]$is_leaf) {
      break
    }

    result <- prune_weakest_link(current_nodes, y, mode)
    if (!result$pruned) {
      break
    }

    current_nodes <- result$nodes

    tree_sequence[[step]] <- current_nodes
    lambda_sequence[step] <- result$lambda
    step <- step + 1
  }

  unique_lambdas <- unique(lambda_sequence)
  final_trees <- list()
  final_lambdas <- numeric(length(unique_lambdas))

  for (i in seq_along(unique_lambdas)) {
    lam <- unique_lambdas[i]
    # Finde den letzten Index in der unbereinigten Liste, der dieses Lambda hat
    last_idx <- max(which(lambda_sequence == lam))

    final_lambdas[i] <- lam
    final_trees[[i]] <- tree_sequence[[last_idx]]
  }

  return(list(
    trees = final_trees,
    lambdas = final_lambdas
  ))
}

# Automatische Lambdabestimmung (mit Cross-Validation) nach Bem. 6.21

# Aus Lemma 6.20: Findet den Index p in der Sequenz, der R_n(f_T^(p)) + lambda * #T^(p) minimiert
get_optimal_tree_for_lambda <- function(pruning_seq, lambda, y_train, mode) {
  P <- length(pruning_seq$trees)
  scores <- numeric(P)

  for (p in 1:P) {
    nodes_p <- pruning_seq$trees[[p]]

    # R_n(T^(p)) - Empirisches Risiko auf den Trainingsdaten
    R_n <- get_subtree_error(nodes_p, 1, y_train, mode = mode)
    # #T^(p) - Anzahl der Blätter des Teilbaums
    num_leaves <- length(get_leaves_of_subtree(nodes_p, 1))
    # Gütekriterium berechnen
    scores[p] <- R_n + lambda * num_leaves
  }

  min_score <- min(scores)

  # workaround: Toleranz für Float-Ungenauigk.
  best_indices <- which(scores <= min_score + 1e-9)

  # Wähle den am stärksten geprunten Baum (höchster Index in der Sequenz)
  return(max(best_indices))
}

# Aus Bemerkung 6.21: M-Fold Cross-Validation
cv_optimal_lambda <- function(X, y, fit, seq_full, mode, M = 5, max_splits = .Machine$integer.max) {
  n <- nrow(X)

  lambdas <- seq_full$lambdas
  if (length(lambdas) == 0) lambdas <- c(0)

  y <- if (mode == "classification") as.factor(y) else as.numeric(y)
  folds <- sample(rep(1:M, length.out = n))
  cv_errors <- rep(0, length(lambdas))

  for (m in 1:M) {
    cat("Fold", m, "\n")
    test_idx <- which(folds == m)
    train_idx <- setdiff(1:n, test_idx)

    X_train <- X[train_idx, , drop = FALSE]
    y_train <- y[train_idx]
    X_test <- X[test_idx, , drop = FALSE]
    y_test <- y[test_idx]

    # Ermittle T_n(m) auf {1..n} \ I_m und die Pruning-Sequenz
    if (mode == "regression") {
      fit_m <- fit_greedy_cart_regression(X_train, y_train, max_splits = max_splits, print_splits = FALSE)
    } else {
      fit_m <- fit_greedy_cart_classification(X_train, y_train, max_splits = max_splits, print_splits = FALSE)
    }
    seq_m <- cost_complexity_sequence(fit_m$nodes, y_train, mode = mode)

    for (k in seq_along(lambdas)) {
      lam <- lambdas[k]

      # Lemma 6.20 anwenden um p_hat(lambda, m) zu finden
      best_p <- get_optimal_tree_for_lambda(seq_m, lam, y_train, mode = mode)
      best_nodes <- seq_m$trees[[best_p]]

      temp_fit <- list(nodes = best_nodes)

      # Fehler auf dem Testset berechnen
      if (mode == "regression") {
        class(temp_fit) <- "greedy_cart_reg"
        preds <- predict(temp_fit, X_test)
        err <- sum((y_test - preds)^2)
      } else {
        temp_fit$levels <- fit_m$levels
        class(temp_fit) <- "greedy_cart_clas"
        preds <- predict(temp_fit, X_test)
        err <- sum(y_test != preds)
      }

      cv_errors[k] <- cv_errors[k] + err
    }
  }

  # Durchschn. Fehler
  cv_errors <- cv_errors / n

  min_cv_error <- min(cv_errors)

  # Es kann mehrere Bäume geben, die den minimalen CV-Fehler erreichen
  best_indices <- which(cv_errors == min_cv_error & lambdas > 0)
  if (length(best_indices) == 0) {
    best_indices <- which(cv_errors == min_cv_error)
  }

  # TODO: wie wählt man hier am besten aus?
  best_tree_idx <- min(best_indices)
  best_lambda <- lambdas[best_tree_idx]

  print("cv_errors")
  print(cv_errors)
  print("lambdas")
  print(lambdas)
  cat("best_tree_idx:", best_tree_idx, "\n")

  structure(
    list(best_lambda = best_lambda, best_tree = seq_full$trees[[best_tree_idx]]),
    class = "cross_validation_result"
  )
}
