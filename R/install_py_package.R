#' Install all Python Packages
#' @description
#' Installs a specific uninstalled Python package.  Most users of `ETMr` will not
#' need to use this function because `ETMr` should automatically
#' installs all necessary Python packages.  The reasons to manually install
#' Python packages are: 1) to use GPUs with PyTorch; or 2) to use a virtual
#' environment outside of the one launched with the `start_etm` function
#' (in other words, a virtual environment not named `r-reticulate`).  `install_py_package`
#' installs one package at a time.  Advanced users are encouraged to use
#' the `reticulate::py_install` function directly.
#'
#' @param package_name The Python package to install
#' @examples
#' install_py_package("gensim")
#'
#' @export
install_py_package <-
  function(package_name){
    py_install(package_name,
               method = "auto",
               envname = "r-reticulate")
  }
