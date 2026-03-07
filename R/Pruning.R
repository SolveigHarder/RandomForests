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
    mu <- node$pred
    return(sum((y[idx] - mu)^2) / length(y))

  } else if (mode == "classification") {
    if (length(idx) == 0) return(0)
    pred_class <- node$pred
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

  return(list(nodes = nodes, lambda = min_lambda, pruned = TRUE))
}

# Generiert die gesamte Sequenz der gestutzten Bäume
cost_complexity_sequence <- function(initial_nodes, y, mode) {
  stopifnot("Paraneter 'mode' muss 'regression' oder 'classification' enthalten" = mode %in% c("regression", "classification"))

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

  return(list(
    trees = tree_sequence,
    lambdas = lambda_sequence
  ))
}
