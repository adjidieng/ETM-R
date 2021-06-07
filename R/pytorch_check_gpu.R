#' Check that GPU support is enabled for PyTorch
#'
#' @description
#' `pytorch_check_gpu` checks if GPU support is enabled for PyTorch.
#' @return Boolean (TRUE or FALSE) entry, depending on if GPU support
#' is available.  Note that GPU support is not available for Mac OS users.
#' @examples
#' \dontrun{
#' pytorch_check_gpu()
#' }
#' @export

pytorch_check_gpu <- function() {
  subprocess$call(paste("python pytorch_check_cuda.py"),
                  shell = TRUE)
}
