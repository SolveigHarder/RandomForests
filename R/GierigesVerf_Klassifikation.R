#Zu ändern für Klassifikation:
#Für Klassifikation ist Schritt (3) nicht mehr SSE, sondern Fehlklassifikationsanzahl nach Majority-Vote
#in den beiden Kindknoten

#d. h. in jedem Kindknoten nimmt man als die häufigste Klasse und zählt die falsch klassifizierten Punkte.

#1. in best split for leaf

#Hilfsfunktion: SSE einer Menge von Indizes
# Summe der der quadrierten Abweichungen von der besten Konstante im Blatt

#hier: Statt SSE
# Mehrheitsklasse (Mode)

majority_class <- function(y, idx) {
  tab <- table(y[idx]) #zählt: wie oft kommt jede Klasse vor
  names(tab)[which.max(tab)]  # liefert Klassenlabel (character) mit der meisten Häufigkeit,
  #Frage: bei Gleichheit einfach die erste? also passt das oder eher per Zufall
}

# Fehlklassifikationen im Blatt, wenn man Majority predicten würde
misclass_of_indices <- function(y, idx) {
  if (length(idx) == 0) return(Inf)
  tab <- table(y[idx])
  length(idx) - max(tab)   # Gesamtzahl minus Anzahl in der Mehrheitsklasse -> "falsche Punkte"
  #brauchen wir für Algorithmus (6.11) #A * (1-(#Mehrheitsklasse/#A))
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

best_split_for_leaf_class <- function(X, y, idx, min_leaf_size = 1, mtry = NULL) { #nach Buch

  X <- as.matrix(X)  # sicherheitshalber
  if (is.null(nrow(X))) X <- matrix(X, nrow = 1)
  d <- ncol(X)



  best <- list(
    score = Inf,
    j = NA_integer_,
    s = NA_real_,
    left_idx = NULL,
    right_idx = NULL,
    # ĉ1(j,s) # hat hier Bedeutung: Mehrheitsklasse links
    c1 = NA_character_,
    c2 = NA_character_
  )

  # Änderung: für Random Forest Algorithmus nur noch mtry features nutzen
  feature_set <- if (is.null(mtry) || mtry >= d) {
    seq_len(d)
  } else {
    sample.int(d, size = mtry, replace = FALSE)
  }

  for (j in feature_set) { #Hier nur noch für alle j in feature set mit Größe mtry machen

    xj_all <- X[idx, j]                 # Feature j im Blatt
    s_candidates <- candidate_splits(xj_all)
    if (length(s_candidates) == 0) next

    for (s in s_candidates) {

      left_local  <- which(xj_all <  s)
      right_local <- which(xj_all >= s)

      left_idx  <- idx[left_local]      # A1(j,s) auf Datenebene
      right_idx <- idx[right_local]     # A2(j,s) auf Datenebene

      if (length(left_idx) < min_leaf_size || length(right_idx) < min_leaf_size) next
#hier beginnt unterschiedlich zu Regression

      # ĉ1(j,s), ĉ2(j,s): Mehrheitsklasse in den Kindern (Def. 6.16 / (6.13)-(6.14))
      c1 <- majority_class(y, left_idx)
      c2 <- majority_class(y, right_idx)

      # score gemäß (6.11): Fehlklassifikationen links + rechts
      err_left  <- misclass_of_indices(y, left_idx)
      err_right <- misclass_of_indices(y, right_idx)
      #bis hier unterschiedlich
      score <- err_left + err_right #(6.11) im Algorithmus
      #Gesamtzahl falscher Klassifikationen nach dem Split, wenn jedes Kindblatt seine Mehrheitsklasse vorhersagt.


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


new_node_class <- function(idx, y) {
  list(
    idx = idx,
    is_leaf = TRUE,
    pred = majority_class(y, idx), #in Klassifikation brauchen wir hier das Mehrheitsvote
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
#' Fit greedy CART classification tree
#' @export
fit_greedy_cart_classification <- function(X, y,
                                       max_splits = 10^9, #Buch Abbruch: nur wenn kein Blatt mehr splitbar ist
                                       min_leaf_size = 1,
                                       min_improve = 1e-12,
                                       print_splits = TRUE, #nach Buch sonst -infinity?
                                       mtry = NULL) {

  y <- factor(y)
  X <- as.matrix(X)
  n <- nrow(X)

  nodes <- list()
  nodes[[1]] <- new_node_class(idx = seq_len(n), y = y)

  leaf_ids <- function(nodes) {
    which(vapply(nodes, function(nd) nd$is_leaf, logical(1)))
  }

  for (k in seq_len(max_splits)) {

    leaves <- leaf_ids(nodes)

    # Wir suchen jetzt global den Split mit GRÖSSTER Verbesserung
    best_global <- list(
      improvement = -Inf,
      leaf_id = NA_integer_,
      split = NULL,
      parent_err = NA_real_
    )

    for (lid in leaves) {
      idx <- nodes[[lid]]$idx

      # Buch: wenn Blatt nur 1 Punkt hat, kann man nicht mehr splitten
      if (length(idx) <= 1) next

      # Praktisch: min_leaf_size beachten
      if (length(idx) < 2 * min_leaf_size) next

      parent_err <- misclass_of_indices(y, idx) ###für Klassifikation
      split <- best_split_for_leaf_class(X, y, idx, min_leaf_size = min_leaf_size,
                                         mtry = mtry)

      # Falls kein gültiger Split existiert:
      if (is.na(split$j)) next

      improvement <- parent_err - split$score

      if (improvement > best_global$improvement) {
        best_global$improvement <- improvement
        best_global$leaf_id <- lid
        best_global$split <- split
        best_global$parent_err <- parent_err
      }
    }

    # Abbruch: kein Split gefunden
    if (is.na(best_global$leaf_id)) break

    # Optionaler Abbruch über "Verbesserung":
    if (best_global$improvement < min_improve) break

    # Split ausführen
    lid <- best_global$leaf_id
    sp  <- best_global$split

    left_id  <- length(nodes) + 1
    right_id <- length(nodes) + 2
    nodes[[left_id]]  <- new_node_class(sp$left_idx,  y)
    nodes[[right_id]] <- new_node_class(sp$right_idx, y)

    nodes[[lid]]$is_leaf <- FALSE
    nodes[[lid]]$j <- sp$j
    nodes[[lid]]$s <- sp$s
    nodes[[lid]]$left <- left_id
    nodes[[lid]]$right <- right_id

    if (print_splits) {
      cat(sprintf("Split %d: leaf %d by j=%d at s=%.4f | score=%.4f | improvement=%.4f\n",
                k, lid, sp$j, sp$s, sp$score, best_global$improvement))
    }
  }

  structure(list(nodes = nodes, levels = levels(y)), class = "greedy_cart_clas")
}


# Prediction für greedy_cart_clas
# Traversiere den Baum: bei innerem Knoten splitte nach (j,s),
# bis ein Blatt erreicht ist, dann gib pred zurück.
#' @export
predict.greedy_cart_clas <- function(object, newdata, ...) {
  nodes <- object$nodes
  X <- as.matrix(newdata)
  n <- nrow(X)

  preds <- character(n)  # erst anlegen!

  for (i in seq_len(n)) {
    node_id <- 1L

    repeat {
      nd <- nodes[[node_id]]

      if (nd$is_leaf) {
        preds[i] <- nd$pred
        break
      }

      j <- nd$j
      s <- nd$s

      if (X[i, j] < s) node_id <- nd$left else node_id <- nd$right
    }
  }

  factor(preds, levels = object$levels)
}


