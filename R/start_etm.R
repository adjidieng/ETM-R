#' Start Python Environment
#' @description
#' This is always the first step when using `ETMr`.  `start_etm`
#' first launches a virtual environment in which Python can be used, then
#' verifies that all necessary Python packages are available.
#' @param env_name By default, ETMr automatically loads a virtual environment
#'  within which R and Python packages can be simultaneously loaded.  The default virtual environment name
#'  is "r-reticulate".  Advanced users are welcome to change the virtual environment name;
#'  just remember to re-install all necessary Python packages using the function `install_py_package`.
#' @examples
#' \dontrun{
#' start_etm()
#' start_etm("new_environment_name")
#' }
#' @importFrom reticulate use_virtualenv
#' @importFrom reticulate py_module_available
#' @export

start_etm <-
  function(env_name = "r-reticulate"){
    # start environment
    reticulate::use_virtualenv(env_name)
    message(paste("Starting virtual environment:", env_name))
    message(paste("Verifying that all Python packages are installed...", env_name))
    # check for Python packages
    if(sum(sapply(c("gensim","matplotlib","numpy",
                    "sklearn","scipy", "torch"),
                  reticulate::py_module_available
                  )) != 6){
      message("Error: one or more packages not installed.  Run ETMr::install_py_package.")
    } else{
      message("All Python packages are installed.")
    }
  }

