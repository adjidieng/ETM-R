#' Start Python Environment
#' @description
#' This is always the first step when using `ETMr`.  `start_etm`
#' first launches a virtual environment that accesses Python packages, then
#' loads the necessary Python packages directly into the R environment.
#' @examples
#' start_python()
#' @export
start_etm <-
  function(){
    use_virtualenv("r-reticulate")
    os <- import("os")
  }
