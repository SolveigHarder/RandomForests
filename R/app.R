library(shiny)

# Die restlichen Skripte nehmen an, dass wir im Projekt-Root sind und nicht im R/ Unterordner.
# RStudio's "Run App" Button startet allerdings dieses Skript im Unterordner, daher gehe in den Root-Ordner falls nötig.
if (basename(getwd()) == "R") {
  setwd("..")
}

# Algorithmen
source("R/GierigesVerf_Klassifikation.R")
source("R/GierigesVerf_Regression.R")
source("R/Pruning.R")
source("R/Bagging_6_25.R")
source("R/Bagging_6_26.R")
source("R/Bagging_6_28.R")
source("R/Rdm_Forest_Algo.R")
# Plotting
source("R/Plot_Klassifikation.R")
source("R/Plot_Regression.R")

ui <- fluidPage(
  br(),
  sidebarLayout(
    sidebarPanel(
      selectInput("task", "Aufgabe:",
                  choices = c("Gierige Verfahren", "Cost-Complexity Pruning", "Bagging", "Random Forest")),

      selectInput("mode", "Modus:", choices = c("Regression", "Klassifikation")),
      uiOutput("func_select_ui"),

      # Feature-Auswahl für Iris
      conditionalPanel(
        condition = "input.func_type == 'Iris-Datensatz'",
        selectInput("iris_x", "Feature X-Achse:", choices = names(iris)[1:4], selected = "Sepal.Length"),
        selectInput("iris_y", "Feature Y-Achse:", choices = names(iris)[1:4], selected = "Petal.Length")
      ),

      conditionalPanel(
        condition = "input.func_type != 'Iris-Datensatz'",
        sliderInput("n_samples", "Anzahl Datenpunkte (n):", min = 20, max = 500, value = 150, step = 10),
        checkboxInput("add_noise", "Rauschen hinzufügen", value = TRUE)
      ),

      numericInput("global_seed", "Random Seed:", value = 1, min = 1),

      # Parameter für Klassifikationen
      conditionalPanel(
        condition = "input.mode == 'Klassifikation'",
        sliderInput("plot_res", "Plot-Auflösung (Rastergröße):", min = 4, max = 250, value = 200, step = 1)
      ),

      # Parameter für Gierige Verfahren
      conditionalPanel(
        condition = "input.task == 'Gierige Verfahren'",
        numericInput("max_splits", "Max Splits:", value = 20, min = 1),
        numericInput("min_leaf_size", "Min Leaf Size:", value = 5, min = 1),
        numericInput("min_improve", "Min Improve (Verbesserung):", value = 0.01, min = 0, step = 0.005)
      ),

      # Parameter für Cost-Complexity Pruning
      conditionalPanel(
        condition = "input.task == 'Cost-Complexity Pruning'",
        checkboxInput("auto_lambda", "Automatische Lambda-Bestimmung (CV)", value = FALSE)
      ),

      conditionalPanel(
        condition = "input.task == 'Cost-Complexity Pruning' && input.auto_lambda == true",
        sliderInput("cv_folds", "CV Folds (M):", min = 2, max = 10, value = 5, step = 1)
      ),

      # Parameter für Bagging
      conditionalPanel(
        condition = "input.task == 'Bagging'",
        sliderInput("B_trees", "Anzahl Bäume (B):", min = 1, max = 200, value = 50, step = 5),
        numericInput("bag_max_splits", "Max Splits pro Baum:", value = 30, min = 1),
        numericInput("bag_min_leaf_size", "Min Leaf Size pro Baum:", value = 3, min = 1),

        conditionalPanel(
          condition = "input.mode == 'Klassifikation'",
          selectInput("bag_class_method", "Klassifikations-Methode:",
                      choices = c("Majority Vote (Def 6.26)", "Probabilities (Def 6.28)"))
        )
      ),

      # Parameter für Random Forest
      conditionalPanel(
        condition = "input.task == 'Random Forest'",
        sliderInput("rf_B_trees", "Anzahl Bäume (B):", min = 1, max = 200, value = 50, step = 5),
        numericInput("rf_mtry", "mtry (Features pro Split):", value = 1, min = 1),
        uiOutput("rf_An_ui"),
        numericInput("rf_max_splits", "Max Splits pro Baum:", value = 30, min = 1),
        numericInput("rf_min_leaf_size", "Min Leaf Size pro Baum:", value = 3, min = 1)
      ),

      conditionalPanel(
        condition = "input.task == 'Bagging' && input.mode == 'Regression'",
        checkboxInput("show_individual", "Einzelbäume anzeigen (erste 25)", value = FALSE)
      ),

      conditionalPanel(
        condition = "input.task != 'Gierige Verfahren'",
        hr(),
        actionButton("run_model", "Berechnen & Plotten",
                     class = "btn-primary", icon = icon("play")),
        br(),
        helpText("Hinweis: Berechnungen können bei vielen Datenpunkten/Bäumen eine Weile dauern.")
      )
    ),

    mainPanel(
      uiOutput("lambda_slider_ui"),
      plotOutput("treePlot", height = "600px"),
      br(),
      conditionalPanel(
        condition = "input.task == 'Random Forest' || input.task == 'Bagging'",
        plotOutput("errorPlot", height = "350px")
      )
    )
  )
)

server <- function(input, output, session) {

  output$func_select_ui <- renderUI({
    choices <- c("Sinus", "Cosinus", "Stufenfunktion")
    if (input$mode == "Klassifikation") {
      choices <- c(choices, "Iris-Datensatz")
    }
    if (input$task == "Random Forest") {
      choices <- c(choices, "5D-Raum")
    }
    selectInput("func_type",  "Funktion / Datensatz:", choices = choices)
  })

  output$rf_An_ui <- renderUI({
    numericInput("rf_A_n", "A_n (Bootstrap Sample-Größe):",
                 value = input$n_samples, min = 1, max = input$n_samples)
  })

  # Wenn Bagging+Probabilities ausgewählt wird, setze die Auflösung des Plots
  # automatisch herunter, da die Vorhersagen sonst sehr zeitaufwändig werden.
  observeEvent(input$bag_class_method, {
    if (input$task == "Bagging" && input$mode == "Klassifikation") {
      if (input$bag_class_method == "Probabilities (Def 6.28)") {
        updateSliderInput(session, "plot_res", value = 35)
      }
    }
  })
  # Random Forests ist für mehrdimensionale Daten interessant; wähle 5D-Raum automatisch aus
  observeEvent(input$task, {
    if (input$task == "Random Forest") {
      updateSelectInput(session, "func_type", selected = "5D-Raum")
      updateCheckboxInput(session, "add_noise", value = FALSE)
    }
  })

  generate_data <- function(n, mode, func_type, add_noise, seed) {
    set.seed(seed)

    if (func_type == "Iris-Datensatz") {
      data(iris)
      return(list(X = iris[, 1:4], y = iris$Species, f = NULL, mode = "Klassifikation", is_iris = TRUE, dims = 4))
    }

    if (func_type == "5D-Raum") {
      X <- as.data.frame(matrix(runif(n * 5, -1, 1), ncol = 5))
      colnames(X) <- paste0("X", 1:5)
      f_val <- X$X1^2 + sin(pi * X$X2) + (X$X3 * X$X4) + ifelse(X$X5 > 0, 0.5, -0.5)

      if (mode == "Regression") {
        y <- f_val + (if(add_noise) rnorm(n, sd = 0.3) else 0)
        return(list(X = X, y = y, f = NULL, mode = "Regression", dims = 5))
      } else {
        prob <- 1 / (1 + exp(-2 * f_val))
        y_true <- ifelse(runif(n) < prob, 1, 2)
        if (add_noise) {
          flip <- runif(n) < 0.05
          y_true[flip] <- ifelse(y_true[flip] == 1, 2, 1)
        }
        return(list(X = X, y = as.factor(y_true), f = NULL, mode = "Klassifikation", dims = 5))
      }
    }

    if (func_type == "Sinus") {
      f <- function(x) sin(x)
      min_x <- -pi; max_x <- pi
    } else if (func_type == "Cosinus") {
      f <- function(x) cos(x)
      min_x <- -pi; max_x <- pi
    } else {
      f <- function(x) ifelse(x < -0.2, 1, ifelse(x < 0.4, 3, 0))
      min_x <- -1; max_x <- 1
    }

    if (mode == "Regression") {
      x <- sort(runif(n, min_x, max_x))
      y <- f(x)
      if (add_noise) {
        y <- y + rnorm(n, sd = 0.2)
      }
      X <- data.frame(x = x)
      dims <- 1
    } else {
      grid_vals <- f(seq(min_x, max_x, length.out = 1000))
      min_y <- min(grid_vals) - 0.5
      max_y <- max(grid_vals) + 0.5

      X1 <- runif(n, min_x, max_x)
      X2 <- runif(n, min_y, max_y)
      X <- data.frame(X1 = X1, X2 = X2)

      y_true <- ifelse(X2 < f(X1), 1, 2)
      if (add_noise) {
        flip <- runif(n) < 0.07
        y_true[flip] <- ifelse(y_true[flip] == 1, 2, 1)
      }
      y <- as.factor(y_true)
      dims <- 2
    }

    list(X = X, y = y, f = f, mode = mode, min_x = min_x, max_x = max_x, dims = dims)
  }

  plot_data_greedy <- reactive({
    dat <- generate_data(input$n_samples, input$mode, input$func_type, input$add_noise, input$global_seed)
    # Für Iris: reduziere X auf die 2 gewählten Dimensionen
    if (!is.null(dat$is_iris) && dat$is_iris) {
      dat$X <- dat$X[, c(input$iris_x, input$iris_y)]
    }
    dat$max_splits <- input$max_splits
    dat$min_leaf_size <- input$min_leaf_size
    dat$min_improve <- input$min_improve
    dat
  })

  pruning_computation <- eventReactive(input$run_model, {
    req(input$task == "Cost-Complexity Pruning")
    dat <- generate_data(input$n_samples, input$mode, input$func_type, input$add_noise, input$global_seed)
    # Feature-Auswahl für Iris
    if (!is.null(dat$is_iris) && dat$is_iris) {
      dat$X <- dat$X[, c(input$iris_x, input$iris_y)]
    }

    is_auto <- input$auto_lambda

    withProgress(message = paste('Berechne', dat$mode, '...'), value = 0, {
      set.seed(input$global_seed)
      incProgress(0.2, detail = "Vollständiger Baum wird erzeugt...")

      if (dat$mode == "Regression") {
        fit <- fit_greedy_cart_regression(dat$X, dat$y, min_improve = 0, print_splits = FALSE)
        pruning_seq <- cost_complexity_sequence(fit$nodes, dat$y, "regression")
      } else {
        fit <- fit_greedy_cart_classification(dat$X, dat$y, min_improve = 0, print_splits = FALSE)
        pruning_seq <- cost_complexity_sequence(fit$nodes, dat$y, "classification")
      }

      cv_result <- NULL
      if (is_auto) {
        incProgress(0.4, detail = "Cross-Validation für bestes Lambda...")
        cv_mode <- ifelse(dat$mode == "Regression", "regression", "classification")
        cv_result <- cv_optimal_lambda(dat$X, dat$y, fit, pruning_seq, mode = cv_mode, M = input$cv_folds)
      }

      incProgress(0.9, detail = "Bereite Visualisierung vor...")
      list(dat = dat, fit = fit, pruning_seq = pruning_seq, cv_result = cv_result, auto_lambda_state = is_auto)
    })
  }, ignoreNULL = TRUE)

  bagging_computation <- eventReactive(input$run_model, {
    req(input$task == "Bagging")
    dat <- generate_data(input$n_samples, input$mode, input$func_type, input$add_noise, input$global_seed)
    if (!is.null(dat$is_iris) && dat$is_iris) {
      dat$X <- dat$X[, c(input$iris_x, input$iris_y)]
    }

    saved_B_trees <- input$B_trees
    saved_bag_class_method <- input$bag_class_method

    withProgress(message = paste('Berechne Bagging', dat$mode, '...'), value = 0.3, {
      set.seed(input$global_seed)

      if (dat$mode == "Regression") {
        fit <- bagging_regression(dat$X, dat$y, B = saved_B_trees,
                                  max_splits = input$bag_max_splits,
                                  min_leaf_size = input$bag_min_leaf_size,
                                  print_splits = FALSE)
      } else {
        if (saved_bag_class_method == "Majority Vote (Def 6.26)") {
          fit <- bagging_classification(dat$X, dat$y, B = saved_B_trees,
                                        max_splits = input$bag_max_splits,
                                        min_leaf_size = input$bag_min_leaf_size,
                                        print_splits = FALSE)
        } else {
          fit <- bagging_classification_proba(dat$X, dat$y, B = saved_B_trees,
                                              max_splits = input$bag_max_splits,
                                              min_leaf_size = input$bag_min_leaf_size,
                                              print_splits = FALSE)
        }
      }

      incProgress(0.6, detail = "Berechne Fehlerrate...")

      all_preds <- matrix(NA, nrow = nrow(dat$X), ncol = saved_B_trees)
      for(b in 1:saved_B_trees) {
        all_preds[, b] <- as.numeric(predict(fit$trees[[b]], dat$X))
      }

      error_vals <- numeric(saved_B_trees)
      if (dat$mode == "Regression") {
        for(b in 1:saved_B_trees) {
          collected_pred <- rowMeans(all_preds[, 1:b, drop = FALSE])
          error_vals[b] <- mean((collected_pred - dat$y)^2)
        }
      } else {
        y_numeric <- as.numeric(dat$y)
        for(b in 1:saved_B_trees) {
          collected_pred <- apply(all_preds[, 1:b, drop = FALSE], 1, function(x) {
            as.numeric(names(which.max(table(x))))
          })
          error_vals[b] <- mean(collected_pred != y_numeric)
        }
      }

      list(dat = dat, fit = fit, B_trees = saved_B_trees, class_method = saved_bag_class_method, show_individual = input$show_individual, errors = error_vals)
    })
  }, ignoreNULL = TRUE)

  rf_computation <- eventReactive(input$run_model, {
    req(input$task == "Random Forest")
    dat <- generate_data(input$n_samples, input$mode, input$func_type, input$add_noise, input$global_seed)
    if (!is.null(dat$is_iris) && dat$is_iris) {
      dat$X <- dat$X[, c(input$iris_x, input$iris_y)]
    }

    saved_B_trees <- input$rf_B_trees
    saved_mtry <- input$rf_mtry
    saved_A_n <- input$rf_A_n

    d <- ncol(dat$X)
    if (saved_mtry > d) saved_mtry <- d
    if (is.null(saved_A_n) || saved_A_n > nrow(dat$X)) saved_A_n <- nrow(dat$X)

    withProgress(message = 'Berechne Random Forest...', value = 0, {
      set.seed(input$global_seed)
      if (input$mode == "Regression") {
        fit <- random_forest_regression(dat$X, dat$y, B = saved_B_trees,
                                        mtry = saved_mtry, A_n = saved_A_n,
                                        max_splits = input$rf_max_splits,
                                        min_leaf_size = input$rf_min_leaf_size,
                                        print_splits = FALSE)
      } else {
        fit <- random_forest_classification(dat$X, dat$y, B = saved_B_trees,
                                            mtry = saved_mtry,
                                            max_splits = input$rf_max_splits,
                                            min_leaf_size = input$rf_min_leaf_size,
                                            print_splits = FALSE)
      }

      incProgress(0.6, detail = "Berechne Fehlerrate...")

      all_preds <- matrix(NA, nrow = nrow(dat$X), ncol = saved_B_trees)
      for(b in 1:saved_B_trees) {
        all_preds[, b] <- as.numeric(predict(fit$trees[[b]], dat$X))
      }

      error_vals <- numeric(saved_B_trees)
      if (input$mode == "Regression") {
        for(b in 1:saved_B_trees) {
          collected_pred <- rowMeans(all_preds[, 1:b, drop = FALSE])
          error_vals[b] <- mean((collected_pred - dat$y)^2)
        }
      } else {
        y_numeric <- as.numeric(dat$y)
        for(b in 1:saved_B_trees) {
          collected_pred <- apply(all_preds[, 1:b, drop = FALSE], 1, function(x) {
            as.numeric(names(which.max(table(x))))
          })
          error_vals[b] <- mean(collected_pred != y_numeric)
        }
      }

      # RF ist für 1D-Daten identisch zu Bagging
      # Für Hyperdimensionale Daten zeige Feature Usage im Vergleich zu Bagging
      bag_fit <- NULL
      if (ncol(dat$X) > 2) {
        set.seed(input$global_seed)
        if (input$mode == "Regression") {
          bag_fit <- bagging_regression(dat$X, dat$y, B = saved_B_trees,
                                        max_splits = input$rf_max_splits,
                                        min_leaf_size = input$rf_min_leaf_size,
                                        print_splits = FALSE)
        } else {
          bag_fit <- bagging_classification(dat$X, dat$y, B = saved_B_trees,
                                            max_splits = input$rf_max_splits,
                                            min_leaf_size = input$rf_min_leaf_size,
                                            print_splits = FALSE)
        }
      }

      list(dat = dat, fit = fit, B_trees = saved_B_trees, mtry = saved_mtry,
           A_n = saved_A_n, show_individual = input$show_individual,
           errors = error_vals, bag_fit = bag_fit)
    })
  }, ignoreNULL = TRUE)

  # Zählt wie oft Feature in allen Bäumen als Split benutzt wurde
  count_feature_usage <- function(model, d) {
    counts <- integer(d)
    for (tr in model$trees) {
      nodes <- tr$nodes
      for (nd in nodes) {
        if (!is.null(nd$j) && !is.na(nd$j)) {
          counts[nd$j] <- counts[nd$j] + 1
        }
      }
    }
    counts
  }

  # --- Pruning Schritt Slider ---
  output$lambda_slider_ui <- renderUI({
    req(input$task == "Cost-Complexity Pruning")
    res <- pruning_computation()
    req(res)

    n_lambdas <- length(res$pruning_seq$lambdas)
    start_idx <- 1
    if (res$auto_lambda_state && !is.null(res$cv_result)) {
      # Falls CV aktiviert: wähle Index des optimalen Lambdas automatisch aus
      start_idx <- which.min(abs(res$pruning_seq$lambdas - res$cv_result$best_lambda))
    }

    sliderInput("lambda_index", "Wähle Pruning-Schritt (1 = ungestutzt):",
                min = 1, max = n_lambdas, value = start_idx, step = 1, width = "100%",
                animate = animationOptions(interval = 800, loop = FALSE))
  })

  # --- Plot Hilfsfkt.
  plot_multi_tree_regression <- function(fit, dat, title_text, show_individual) {
    par(oma = c(2, 0, 0, 0))
    x_vals <- dat$X[[1]]
    plot(x_vals, dat$y, main = title_text, xlab = "x", ylab = "y", pch = 19, col = "gray", cex = 0.8)

    x_grid <- seq(min(x_vals), max(x_vals), length.out = 2000)
    if (!is.null(dat$f)) lines(x_grid, dat$f(x_grid), col = "gray", lwd = 2)

    X_grid <- data.frame(x = x_grid)
    names(X_grid) <- names(dat$X)

    # Max. 25 Einzelbäume -> weniger Rechenaufwand & übersichtlicher
    if (show_individual) {
      n_trees_to_plot <- min(fit$B, 25)
      for (b in 1:n_trees_to_plot) {
        tree_fit <- fit$trees[[b]]
        tree_preds <- predict(tree_fit, X_grid)
        lines(x_grid, tree_preds, col = rgb(0.2, 0.5, 0.8, alpha = 0.25), lwd = 1, type = "s")
      }
    }

    lines(x_grid, predict(fit, X_grid), col = "red", lwd = 2.5, type = "s")

    par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
    plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")
    legend("bottom", legend = c("Daten", "Originale Funktion", "Einzelbäume", "Vorhersage"),
           col = c("gray", "gray", rgb(0.2, 0.5, 0.8, alpha = 0.6), "red"),
           lty = c(NA, 1, 1, 1),
           pch = c(19, NA, NA, NA),
           lwd = c(NA, 2, 1, 2.5),
           bg = "white", xpd = TRUE, bty = "n", horiz=TRUE)
  }

  add_regression_legend <- function() {
    par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
    plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")
    legend("bottom", legend = c("Daten", "Originale Funktion", "Vorhersage", "Splits"),
           col = c("gray", "gray", "red", "blue"), lty = c(NA, 1, 1, 2),
           pch = c(19, NA, NA, NA), lwd = c(NA, 2, 2, 1),
           bg = "white", xpd = TRUE, bty = "n", horiz=TRUE)
  }

  add_classification_legend <- function(levels_y) {
    par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
    plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")
    n_classes <- length(levels_y)
    pt_colors <- c("red", "blue", "darkgreen")[1:n_classes]
    legend("bottom", legend = c(paste("Klasse", levels_y), "Entscheidungsrand"),
           col = c(pt_colors, "black"), pch = c(rep(1, n_classes), NA),
           lty = c(rep(NA, n_classes), 2), lwd = c(rep(1, n_classes), 2),
           horiz = TRUE, bty = "n", cex = 1.1)
  }

  show_message_plot <- function(msg) {
    plot.new()
    text(0.5, 0.5, msg, cex = 1.2)
  }

  output$errorPlot <- renderPlot({
    req(input$task %in% c("Random Forest", "Bagging"))

    if (input$task == "Random Forest") {
      res <- rf_computation()
    } else {
      res <- bagging_computation()
    }
    req(res)

    y_label <- if(res$dat$mode == "Regression") "Mean Squared Error" else "Fehlklassifikationsrate"
    plot_title <- paste("Error Graph -", input$task, "-", res$dat$mode)

    plot(1:res$B_trees, res$errors, type = "l", lwd = 2, col = "darkblue",
         xlab = "Anzahl der Bäume (B)", ylab = y_label,
         main = plot_title)
    points(1:res$B_trees, res$errors, pch = 20, col = "darkblue", cex = 0.5)
    grid()
    abline(h = tail(res$errors, 1), lty = 2, col = "red")
    legend("topright", legend = c("Fehler", paste("Endfehler:", round(tail(res$errors, 1), 4))),
           col = c("darkblue", "red"), lty = c(1, 2), bty = "n")
  })

  # --- Main Tree Plot ---
  output$treePlot <- renderPlot({
    task <- input$task

    if (task %in% c("Cost-Complexity Pruning", "Bagging", "Random Forest") && input$run_model == 0) {
      return(show_message_plot("Bitte Parameter wählen und auf 'Berechnen & Plotten' klicken."))
    }

    # Stellt orig. Plotparam. am Ende der Funktion wiederher
    opar <- par(no.readonly = TRUE)
    on.exit(par(opar))

    withProgress(message = 'Erstelle Plot...', detail = 'Vorhersagen werden berechnet', value = 0.5, {

      if (task == "Gierige Verfahren") {
        dat <- plot_data_greedy()

        if (dat$mode == "Regression") {
          fit <- fit_greedy_cart_regression(dat$X, dat$y, max_splits = dat$max_splits,
                                            min_leaf_size = dat$min_leaf_size,
                                            min_improve = dat$min_improve, print_splits = FALSE)
          par(oma = c(2, 0, 0, 0))
          plot_regression_fit(fit, dat$X, dat$y, "CART (Gieriges Verfahren)", dat$f, TRUE)
          add_regression_legend()

        } else {
          fit <- fit_greedy_cart_classification(dat$X, dat$y, max_splits = dat$max_splits,
                                                min_leaf_size = dat$min_leaf_size,
                                                min_improve = dat$min_improve, print_splits = FALSE)
          par(oma = c(3, 0, 0, 0))
          plot_classification_fit(fit, dat$X, dat$y, "CART (Gieriges Verfahren)", dat$f, res = input$plot_res)
          add_classification_legend(levels(dat$y))
        }

      } else if (task == "Cost-Complexity Pruning") {
        res <- pruning_computation()
        req(res, input$lambda_index)

        dat <- res$dat; fit <- res$fit; pruning_seq <- res$pruning_seq
        idx <- min(input$lambda_index, length(pruning_seq$lambdas))

        fit_pruned_nodes <- pruning_seq$trees[[idx]]
        lam <- pruning_seq$lambdas[idx]

        is_cv_optimal <- res$auto_lambda_state && !is.null(res$cv_result) &&
          idx == which.min(abs(pruning_seq$lambdas - res$cv_result$best_lambda))

        title_pruned <- sprintf("Gestutzt (%s, Lambda = %.6f)",
                                ifelse(is_cv_optimal, "CV Optimal", "Manuell"), lam)

        if (dat$mode == "Regression") {
          par(mfrow = c(1, 2), oma = c(2, 0, 0, 0))
          fit_pruned <- structure(list(nodes = fit_pruned_nodes), class = "greedy_cart_reg")
          plot_regression_fit(fit, dat$X, dat$y, "Ungestutzt (Voll ausgewachsen)", dat$f, TRUE)
          plot_regression_fit(fit_pruned, dat$X, dat$y, title_pruned, dat$f, TRUE)
          add_regression_legend()

        } else {
          par(mfrow = c(1, 2), oma = c(3, 0, 0, 0))
          fit_pruned <- structure(list(nodes = fit_pruned_nodes, levels = fit$levels), class = "greedy_cart_clas")
          plot_classification_fit(fit, dat$X, dat$y, "Ungestutzt (Voll ausgewachsen)", dat$f, res = isolate(input$plot_res))
          plot_classification_fit(fit_pruned, dat$X, dat$y, title_pruned, dat$f, res = isolate(input$plot_res))
          add_classification_legend(levels(dat$y))
        }

      } else if (task == "Bagging") {
        res <- bagging_computation()
        req(res)
        dat <- res$dat; fit <- res$fit

        if (dat$mode == "Regression") {
          title_bag <- sprintf("Bagging Regression (B = %d Bäume)", res$B_trees)
          plot_multi_tree_regression(fit, dat, title_bag, res$show_individual)

        } else {
          par(oma = c(3, 0, 0, 0))
          meth <- ifelse(res$class_method == "Majority Vote (Def 6.26)", "Majority", "Proba")
          title_bag <- sprintf("Bagging Klassifikation (B = %d, %s)", res$B_trees, meth)

          plot_classification_fit(fit, dat$X, dat$y, title_bag, dat$f, res = isolate(input$plot_res))
          add_classification_legend(levels(dat$y))
        }

      } else if (task == "Random Forest") {
        res <- rf_computation()
        req(res)
        dat <- res$dat

        if (ncol(dat$X) > 2) {
          bag_fit <- res$bag_fit

          d <- ncol(dat$X)
          names_feats <- colnames(dat$X)
          bag_counts <- count_feature_usage(bag_fit, d)
          rf_counts  <- count_feature_usage(res$fit, d)

          par(mfrow = c(1, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 3, 0))
          barplot(bag_counts, names.arg = names_feats, las = 2,
                  main = "Feature-Usage: Bagging", ylim = c(0, max(c(bag_counts, rf_counts))+1))
          barplot(rf_counts, names.arg = names_feats, las = 2,
                  main = "Feature-Usage: Random Forest", ylim = c(0, max(c(bag_counts, rf_counts))+1))

        } else if (res$dat$mode == "Regression") {
          title_rf <- sprintf("Random Forest Regression (B = %d, mtry = %d, A_n = %d)",
                              res$B_trees, res$mtry, res$A_n)
          plot_multi_tree_regression(res$fit, res$dat, title_rf, res$show_individual)
        } else {
          par(oma = c(3, 0, 0, 0))
          plot_classification_fit(res$fit, res$dat$X, res$dat$y, "Random Forest Klassifikation", res$dat$f, res = isolate(input$plot_res))
          add_classification_legend(levels(res$dat$y))
        }
      }
    })
  })
}

shinyApp(ui = ui, server = server)
