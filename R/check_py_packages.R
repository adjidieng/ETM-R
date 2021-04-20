#' Check for Python Packages
#'
#' @description
#' `check_py_packages` checks if all Python packages are installed.
#' All Python packages should already be installed when installing `ETMr`.
#' @returns
#' `check_py_packages` returns a list of package names and values of
#' `TRUE` or `FALSE` depending on whether they are installed or not.
#' @examples
#' check_py_packages()
#' @export
check_py_packages <-
  function(){
  sapply(c("gensim", "matplotlib",
         "numpy", "pickle",
         "scipy", "torch"),
       py_module_available)
  }
