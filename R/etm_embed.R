#' Fit Word Embeddings to Text
#'
#' @description
#' `etm_embed` fits word embeddings on your corpus using the popular neural network model
#' Word2Vec.  Choose between two word embedding algorithms: skip-grams, or continuous bag of words.
#' `etm_embed` produces a text file with the word embeddings.
#' @param data_path Path to the corpus text file.
#' @param output_name File name for the output, which is the word embeddings text file.
#' Default is "embeddings.txt".
#' @param dim_rho Number of dimensions for rho, which is the number of
#' embedding representations of the vocabulary.  Default is 300 dimensions.
#' @param min_count Minimum term frequency (to define the vocabulary).  Default is 2.
#' @param iters Number of iterations.  Default is 50 iterations.
#' @param window_size The window size to determine context.  In other words,
#' the number of words surrounding the word of interest.  Default size is 4.
#' @param neg_samples The number of negative samples allowed.  In other words, the number
#' of "noise words" that are drawn.  Default is 10.
#' @param num_cores The number of CPU cores to use.  To automatically use the optimal
#' number of CPU cores available, use "detect".  Otherwise, specify an integer (e.g. 4).
#' The default number of cores is 25.
#' @param algorithm The word embedding algorithm to use.  "skipgram" for skip-grams,
#' or "cbow" for continuous bag of words.  Default is skip-gram.
#' @examples
#' \dontrun{
#' etm_embed(data_path = "Desktop/corpus.txt",
#' output_name = "Desktop/embeddings.txt",
#' dim_rho = 300, iters = 50, window_size = 4, neg_samples = 10,
#' num_cores = "detect", algorithm = "skipgram")
#'
#' etm_embed(data_path = "Desktop/news.txt", output_name = "Documents/embed_file.txt",
#' num_cores = 4, algorithm = "cbow")
#' }
#' @importFrom parallel detectCores
#' @export

etm_embed <- function(data_path,
                      output_name,
                      dim_rho = 300,
                      min_count = 2,
                      iters = 50,
                      window_size = 4,
                      neg_samples = 10,
                      num_cores = 25,
                      algorithm = "skipgram") {
  if(num_cores == "detect"){
    n_cores <- parallel::detectCores() - 1
    if(algorithm  == "skipgram"){
      subprocess$call(paste("python inst/python/skipgram.py --data_file ", data_path, "--embed_file ", -output_name, "--dim_rho ", dim_rho, "--min_count ", min_count, "--sg 1 --workers", n_cores, "--negative_samples", neg_samples,
                            "--window_size ", window_size, "--iters ", iters),
                      shell = TRUE)
    }
    else if(algorithm == "cbow") {
      subprocess$call(paste("python inst/python/skipgram.py --data_file ", data_path, "--embed_file ", -output_name, "--dim_rho ", dim_rho, "--min_count ", min_count, "--sg 0 --workers", n_cores, "--negative_samples", neg_samples,
                            "--window_size ", window_size, "--iters ", iters),
                      shell = TRUE)
    }
    else{
      message("Error: algorithm not properly specified.  Choose either skipgram or cbow.")
    }
  } else{
    if(algorithm  == "skipgram"){
      subprocess$call(paste("python inst/python/skipgram.py --data_file ", data_path, "--embed_file ", -output_name, "--dim_rho ", dim_rho, "--min_count ", min_count, "--sg 1 --workers", num_cores, "--negative_samples", neg_samples,
                            "--window_size ", window_size, "--iters ", iters),
                      shell = TRUE)
    }
    else if(algorithm == "cbow") {
      subprocess$call(paste("python inst/python/skipgram.py --data_file ", data_path, "--embed_file ", -output_name, "--dim_rho ", dim_rho, "--min_count ", min_count, "--sg 0 --workers", num_cores, "--negative_samples", neg_samples,
                            "--window_size ", window_size, "--iters ", iters),
                      shell = TRUE)
    }
    else{
      message("Error: algorithm not properly specified.  Choose either skipgram or cbow.")
    }
  }
}
