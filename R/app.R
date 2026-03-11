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

  sidebarLayout(
    sidebarPanel(
      selectInput("task", "Aufgabe:",
                  choices = c("Gierige Verfahren", "Cost-Complexity Pruning", "Bagging", "Random Forest")),

      selectInput("mode", "Modus:", choices = c("Regression", "Klassifikation")),
      selectInput("func_type", "Wahre Funktion:", choices = c("Sinus", "Cosinus", "Stufenfunktion")),
      sliderInput("n_samples", "Anzahl Datenpunkte (n):", min = 20, max = 500, value = 150, step = 10),

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
        checkboxInput("auto_lambda", "Automatische Lambda-Bestimmung (CV)", value = TRUE)
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

      # Bei Aufgaben, die länger brauchen, muss mit einem Button bestätigt werden, dass eine neue Berechnung gestartet werden soll.
      conditionalPanel(
        condition = "input.task != 'Gierige Verfahren'",
        hr(),
        actionButton("run_model", "Modell trainieren & Plotten",
                     class = "btn-primary", icon = icon("play")),
        helpText("Hinweis: Berechnungen können bei vielen Datenpunkten/Bäumen einen Moment dauern.")
      )
    ),

    mainPanel(
      uiOutput("lambda_slider_ui"),
      plotOutput("treePlot", height = "600px")
    )
  )
)

server <- function(input, output, session) {

  # Wenn Bagging+Probabilities ausgewählt wird, setze die Auflösung des Plots
  # automatisch herunter, da die Vorhersagen sonst sehr zeitaufwändig werden.
  observeEvent(input$bag_class_method, {
    if (input$task == "Bagging" && input$mode == "Klassifikation") {
      if (input$bag_class_method == "Probabilities (Def 6.28)") {
        updateSliderInput(session, "plot_res", value = 35)
      }
    }
  })

  generate_data <- function(n, mode, func_type) {
    set.seed(1)

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
      y <- f(x) + rnorm(n, sd = 0.2)
      X <- data.frame(x = x)
    } else {
      grid_vals <- f(seq(min_x, max_x, length.out = 1000))
      min_y <- min(grid_vals) - 0.5
      max_y <- max(grid_vals) + 0.5

      X1 <- runif(n, min_x, max_x)
      X2 <- runif(n, min_y, max_y)
      X <- data.frame(X1 = X1, X2 = X2)

      y_true <- ifelse(X2 < f(X1), 1, 2)
      flip <- runif(n) < 0.07
      y_true[flip] <- ifelse(y_true[flip] == 1, 2, 1)
      y <- as.factor(y_true)
    }

    list(X = X, y = y, f = f, mode = mode, min_x = min_x, max_x = max_x)
  }

  plot_data_greedy <- reactive({
    dat <- generate_data(input$n_samples, input$mode, input$func_type)
    dat$max_splits <- input$max_splits
    dat$min_leaf_size <- input$min_leaf_size
    dat$min_improve <- input$min_improve
    dat
  })

  pruning_computation <- eventReactive(input$run_model, {
    req(input$task == "Cost-Complexity Pruning")
    dat <- generate_data(input$n_samples, input$mode, input$func_type)
    is_auto <- input$auto_lambda

    withProgress(message = paste('Trainiere', dat$mode, 'smodell...'), value = 0, {
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
    dat <- generate_data(input$n_samples, input$mode, input$func_type)

    saved_B_trees <- input$B_trees
    saved_bag_class_method <- input$bag_class_method

    withProgress(message = paste('Trainiere Bagging', dat$mode, '...'), value = 0.5, {

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

      list(dat = dat, fit = fit, B_trees = saved_B_trees, class_method = saved_bag_class_method)
    })
  }, ignoreNULL = TRUE)

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

    sliderInput("lambda_index", "Wähle Pruning-Schritt (1 = ungestutzt, höhere Werte = stärker gestutzt):",
                min = 1, max = n_lambdas, value = start_idx, step = 1, width = "100%",
                animate = animationOptions(interval = 800, loop = FALSE))
  })

  # --- Plot Hilfsfkt.
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

  # --- Plot ---
  output$treePlot <- renderPlot({
    task <- input$task

    if (task == "Random Forest") {
      return(show_message_plot(paste(task, "ist noch nicht implementiert.")))
    }

    # Überprüfe für aufwendige Aufgaben, ob der Button gedrückt wurde
    if (task %in% c("Cost-Complexity Pruning", "Bagging") && input$run_model == 0) {
      return(show_message_plot("Bitte Parameter wählen und auf 'Trainieren & Plotten' klicken."))
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
          plot_classification_fit(fit, dat$X, dat$y, "Ungestutzt (Voll ausgewachsen)", dat$f, res = input$plot_res)
          plot_classification_fit(fit_pruned, dat$X, dat$y, title_pruned, dat$f, res = input$plot_res)
          add_classification_legend(levels(dat$y))
        }

      } else if (task == "Bagging") {
        res <- bagging_computation()
        req(res)
        dat <- res$dat; fit <- res$fit

        if (dat$mode == "Regression") {
          par(oma = c(2, 0, 0, 0))
          title_bag <- sprintf("Bagging Regression (B = %d Bäume)", res$B_trees)

          x_vals <- dat$X[[1]]
          plot(x_vals, dat$y, main = title_bag, xlab = "x", ylab = "y", pch = 19, col = "gray", cex = 0.8)

          x_grid <- seq(min(x_vals), max(x_vals), length.out = 2000)
          lines(x_grid, dat$f(x_grid), col = "gray", lwd = 2)

          X_grid <- data.frame(x = x_grid)
          names(X_grid) <- names(dat$X)
          lines(x_grid, predict(fit, X_grid), col = "red", lwd = 2, type = "s")

          par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
          plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")
          legend("bottom", legend = c("Daten", "Originale Funktion", "Aggregierte Vorhersage"),
                 col = c("gray", "gray", "red"), lty = c(NA, 1, 1),
                 pch = c(19, NA, NA), lwd = c(NA, 2, 2), bg = "white", xpd = TRUE, bty = "n", horiz=TRUE)

        } else {
          par(oma = c(3, 0, 0, 0))
          meth <- ifelse(res$class_method == "Majority Vote (Def 6.26)", "Majority", "Proba")
          title_bag <- sprintf("Bagging Klassifikation (B = %d, %s)", res$B_trees, meth)

          plot_classification_fit(fit, dat$X, dat$y, title_bag, dat$f, res = input$plot_res)
          add_classification_legend(levels(dat$y))
        }
      }
    })
  })
}

shinyApp(ui = ui, server = server)
