#' Use Pre-Fitted Word Embeddings with the ETM algorithm
#'
#' @description
#' `etm_prefit` runs the ETM algorithm on text with pre-fitted word embeddings
#' (e.g. using skiggrams with the `etm_skipgram` function).
#' @param dataset_name The name of the dataset.
#' @param data_path The directory or location to the dataset.
#' @param embed_path The directory or location to the embeddings file.
#' @param num_topics The number of topics to learn.  The default number of topics is 50.
#' @param epochs Number of times the dataset is fed into the algorithm for
#' learning.  The default number of epochs is 1000.  Minimum number of epochs is 1.
#' @examples
#' \dontrun{
#' etm_prefit(dataset_name = "corpus_1", data_path = "Desktop/", embed_path = "Desktop/embed/",
#'  num_topics = 50, epochs = 1000)
#' }
#' @export
etm_prefit <- function(dataset_name,
                       data_path,
                       embed_path,
                       num_topics = 50,
                       epochs = 1000) {
  subprocess$call(paste("python inst/main_etm.py --mode train --dataset ",dataset_name, "--data_path ",data_path, "--emb_path", embed_path, "--num_topics ", num_topics, "--train_embeddings 0 --epochs ", epochs),
                  shell = TRUE)

}
