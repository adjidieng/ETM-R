#' Install GPU-enabled PyTorch
#'
#' @description
#' `install_pytorch_gpu` installs a version of PyTorch that enables GPUs for general purpose
#' processing.  GPU support for PyTorch is only available for Windows and Linux users
#' (not Mac OS users).  `install_pytorch_gpu`  will first uninstall all CPU-based versions of
#' PyTorch, and then install the GPU-based version of PyTorch.
#' @param os Your operating system, either Windows or Linux (there is no GPU support for
#' Mac OS users).  Default operating system is Windows.
#' @examples
#' \dontrun{
#' install_pytorch_gpu(os = "windows")
#' install_pytorch_gpu(os = "linux")
#' }
#' @export
install_pytorch_gpu <- function(os = "windows") {
  if(os %in% c("windows", "Windows", "WINDOWS")){
    subprocess$call("inst/python/pytorch_cuda_windows.sh")
  } else if (os %in% c("linux", "Linux", "LINUX") ){
    subprocess$call("inst/python/pytorch_cuda_linux.sh")
  } else if (os %in% c("mac", "Mac", "MAC")){
    message("Error: GPU support for PyTorch is not available for Mac OS.")
  } else{
    message("Error: os input invalid.  Please input either Windows or Linux as your os.")
  }
}
