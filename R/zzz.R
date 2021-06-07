.onLoad <- function(libname, pkgname) {
  # use superassignment to update global reference to os
  subprocess <<- reticulate::import("subprocess", delay_load = TRUE)
  reticulate::configure_environment(pkgname)
}
