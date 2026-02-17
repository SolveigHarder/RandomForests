#Hilfsfunktion: SSE einer Menge von Indizes
# Summe der der quadrierten Abweichungen von der besten Konstante im Blatt

sse_of_indices <- function(y, idx) {
  # idx: Integer-Vektor mit Zeilenindizes (Trainingspunkte im Blatt)
  # SSE = sum (y_i - mean(y))^2
  if (length(idx) == 0) return(Inf)       # leerer Knoten -> ungültig
  if (length(idx) == 1) return(0)
  mu <- mean(y[idx])
  sum((y[idx] - mu)^2)
}

# Kandidaten-Schwellen s für ein Feature j
candidate_splits <- function(xj) {
  ux <- sort(unique(xj))
  if (length(ux) <= 1) return(numeric(0))
  (ux[-1] + ux[-length(ux)]) / 2
}

# Best split für EIN Blatt (Indices idx) finden
# X: n x d (matrix oder df), y: Länge n
# idx: welche Punkte sind im aktuellen Blatt A(v)

best_split_for_leaf <- function(X, y, idx, min_leaf_size = 1) { #nach Buch



  X <- as.matrix(X)  # sicherheitshalber
  if (is.null(nrow(X))) X <- matrix(X, nrow = 1)
  d <- ncol(X)

  sse_parent <- sse_of_indices(y, idx) #FEHLT

  best <- list(
    score = Inf,          # = SSE_left + SSE_right  (das ist (6.10))
    j = NA_integer_,
    s = NA_real_,
    left_idx = NULL,
    right_idx = NULL,
    c1 = NA_real_,        # ĉ1(j,s)
    c2 = NA_real_         # ĉ2(j,s)
  )

  for (j in seq_len(d)) {

    xj_all <- X[idx, j]                 # Feature j im Blatt
    s_candidates <- candidate_splits(xj_all)
    if (length(s_candidates) == 0) next

    for (s in s_candidates) {

      left_local  <- which(xj_all <  s)
      right_local <- which(xj_all >= s)

      left_idx  <- idx[left_local]      # A1(j,s) auf Datenebene
      right_idx <- idx[right_local]     # A2(j,s) auf Datenebene

      if (length(left_idx) < min_leaf_size || length(right_idx) < min_leaf_size) next

      # ĉ1(j,s), ĉ2(j,s) = Mittelwerte in den Kindern
      c1 <- mean(y[left_idx])
      c2 <- mean(y[right_idx])

      # Zielwert aus (6.10): SSE links + SSE rechts
      # (robuster: direkt über SSE-Funktion)
      sse_left  <- sse_of_indices(y, left_idx)
      sse_right <- sse_of_indices(y, right_idx)
      score <- sse_left + sse_right

      if (score < best$score) {
        best$score <- score
        best$j <- j
        best$s <- s
        best$left_idx <- left_idx
        best$right_idx <- right_idx
        best$c1 <- c1
        best$c2 <- c2
      }
    }
  }

  # optional: falls kein Split gefunden wurde, bleibt best$j = NA
  best
}

# Datenstruktur für den Baum
#
# Wir speichern den Baum als Liste von Knoten (Nodes).
# Jeder Knoten hat:
# - idx: welche Trainingspunkte liegen in diesem Knoten?
# - is_leaf: TRUE/FALSE
# - pred: Blattwert (Mittelwert)
# - j, s: Split-Parameter (wenn innerer Knoten)
# - left, right: Kinder-Node-IDs (Index in nodes-Liste)

new_node <- function(idx, y) {
  list(
    idx = idx,
    is_leaf = TRUE,
    pred = mean(y[idx]),
    j = NA_integer_,
    s = NA_real_,
    left = NA_integer_,
    right = NA_integer_
  )
}

# Greedy Build: iterativ Blätter splitten
# Variante A: "global greedy" (typisch)
#   -> in jedem Schritt splitte das Blatt, das die größte Verbesserung bringt
#
#  passt zu "wir verringern das empirische Risiko am stärksten".

fit_greedy_cart_regression <- function(X, y,
                                       max_splits = 10^9, #Buch Abbruch: nur wenn kein Blatt mehr splitbar ist
                                       min_leaf_size = 1,
                                       min_improve = 1e-12) { #nach Buch sonst -infinity?

  X <- as.matrix(X)
  n <- nrow(X)

  nodes <- list()
  nodes[[1]] <- new_node(idx = seq_len(n), y = y)

  leaf_ids <- function(nodes) {
    which(vapply(nodes, function(nd) nd$is_leaf, logical(1)))
  }

  for (k in seq_len(max_splits)) {

    leaves <- leaf_ids(nodes)

    # Wir suchen jetzt global den Split mit GRÖSSTER Verbesserung
    # Verbesserung = SSE(parent) - (SSE_left + SSE_right)
    best_global <- list(
      improvement = -Inf,
      leaf_id = NA_integer_,
      split = NULL,
      sse_parent = NA_real_
    )

    for (lid in leaves) {
      idx <- nodes[[lid]]$idx

      # Buch: wenn Blatt nur 1 Punkt hat, kann man nicht mehr splitten
      if (length(idx) <= 1) next

      # Praktisch: min_leaf_size beachten
      if (length(idx) < 2 * min_leaf_size) next

      sse_parent <- sse_of_indices(y, idx)
      split <- best_split_for_leaf(X, y, idx, min_leaf_size = min_leaf_size)

      # Falls kein gültiger Split existiert:
      if (is.na(split$j)) next

      improvement <- sse_parent - split$score

      if (improvement > best_global$improvement) {
        best_global$improvement <- improvement
        best_global$leaf_id <- lid
        best_global$split <- split
        best_global$sse_parent <- sse_parent
      }
    }

    # Abbruch: kein Split gefunden
    if (is.na(best_global$leaf_id)) break

    # Optionaler Abbruch über "Verbesserung":
    # Verbesserung = SSE(parent) - score. Das ist buch-fremd, aber praktisch.
    if (best_global$improvement < min_improve) break

    # Split ausführen
    lid <- best_global$leaf_id
    sp  <- best_global$split

    left_id  <- length(nodes) + 1
    right_id <- length(nodes) + 2
    nodes[[left_id]]  <- new_node(sp$left_idx,  y)
    nodes[[right_id]] <- new_node(sp$right_idx, y)

    nodes[[lid]]$is_leaf <- FALSE
    nodes[[lid]]$j <- sp$j
    nodes[[lid]]$s <- sp$s
    nodes[[lid]]$left <- left_id
    nodes[[lid]]$right <- right_id

    cat(sprintf("Split %d: leaf %d by j=%d at s=%.4f | score=%.4f | improvement=%.4f\n",
                k, lid, sp$j, sp$s, sp$score, best_global$improvement))
  }

  structure(list(nodes = nodes), class = "greedy_cart_reg")
}


# Prediction für greedy_cart_reg
# Traversiere den Baum: bei innerem Knoten splitte nach (j,s),
# bis ein Blatt erreicht ist, dann gib pred zurück.

predict.greedy_cart_reg <- function(object, newdata, ...) {
  nodes <- object$nodes
  X <- as.matrix(newdata)
  n <- nrow(X)

  preds <- numeric(n)

  for (i in seq_len(n)) {#Für jeden neuen Punkt müssen wir im Baum einen Pfad gehen.
    node_id <- 1L  # root

    repeat {
      nd <- nodes[[node_id]] #aktueller Knoten

      if (nd$is_leaf) { #Fall 1
        preds[i] <- nd$pred #wir sind in einem Blatt -> prediction ist gespeicherter Blattwert
        break
      }
      #Fall 2: innerer Knoten
      j <- nd$j
      s <- nd$s

      # Split-Regel wie beim Training: links < s, rechts >= s
      if (X[i, j] < s) {
        node_id <- nd$left
      } else {
        node_id <- nd$right
      }
    }
  }

  preds
}



 # wichtigste Frage: sagt der Algorithmus wie ich denn verschiedene Blätter dann durchgehen soll?


#Testsets
#1D
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






