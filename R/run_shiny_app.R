#' run shiny app
#' @export
run_shiny_app <- function() {
  if (!requireNamespace("shiny", quietly = TRUE)) {
    stop("Bitte installiere das Paket 'shiny' (Suggests).")
  }
  app_dir <- system.file("shiny", package = "RandomForests")
  if (app_dir == "") stop("Keine Shiny-App unter inst/shiny gefunden.")
  shiny::runApp(app_dir)
}

