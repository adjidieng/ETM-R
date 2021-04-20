#' Use Skipgrams to Fit Word Embeddings to Text
#'
#' @description
#' `etm_skipgram` fits word embeddings on your dataset using
#' simple skipgrams.
#' @param data_path Path to the dataset file.
#' @param embed_path Path to the embeddings file.
#' @param dim_rho Number of dimensions for rho, which is the number of
#' embedding representations of the vocabulary.  Default is 300 dimensions.
#' @param iters Number of iterations.  Default is 50 iterations.
#' @param window_size The window size to determine context.  In other words,
#' the number of words surrounding the word of interest.  Default size is 4.
#' @examples
#' etm_skipgram(data_path = "Desktop/", embed_path = "Desktop/embed_file.txt", dim_rho = 300, iters = 50, window_size = 4)
#' @export

etm_skipgram <- function(data_path,
                         embed_path,
                         dim_rho = 300,
                         iters = 50,
                         window_size = 4) {
  os$system(paste("python skipgram.py --data_file ", data_path, "--embed_file ", -embed_path, "--dim_rho", dim_rho, "--iters ", iters, "--window_size ", window_size))

}
