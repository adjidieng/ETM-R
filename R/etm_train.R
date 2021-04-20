#' Training Text using Embedded Topic Modeling
#'
#' @description
#' `etm_train` performs two steps: first, it creates interpretable word embeddings
#' on a given text using Word2Vec; and second, it runs the ETM algorithm
#' to learn the topics of the text.
#' @param dataset_name The name of the dataset.
#' @param data_path The directory or location to the dataset.
#' @param num_topics The number of topics to learn.  The default number of topics is 50.
#' @param epochs Number of times the dataset is fed into the algorithm for
#' learning.  The default number of epochs is 1000.  Minimum number of epochs is 1.
#' @param save_path The directory of location of the trained embeddings.
#' @returns
#' A text file with the trained embeddings and topics,
#' located at the designated save path.
#' @examples
#' etm_train(dataset_name = "corpus_1", data_path = "Desktop/", num_topics = 10,
#' epochs = 1000, save_path = "Desktop/"
#' )
#' @export

etm_train <- function(dataset_name,
                     data_path,
                     num_topics = 50,
                     epochs = 1000,
                     save_path) {
  os$system(paste("python Python/main_etm.py --mode train --dataset ",dataset_name, "--data_path ",data_path, "--num_topics ", num_topics,"--train_embeddings 1 --epochs ", epochs,"--save_path ", save_path))

}
