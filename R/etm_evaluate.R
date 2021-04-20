#' Evaluate Results Embedded Topic Modeling
#'
#' @description
#' `etm_evaluate` evaluates perplexity on document completion,
#' topic coherence and topic diversity.  It then visualizes the topics and embeddings.
#' @param dataset_name The name of the dataset.
#' @param data_path The directory or location to the dataset.
#' @param num_topics The number of topics to learn.
#' @param topic_coherence Whether to compute topic coherence or not (1 for yes,
#' 0 for no).
#' @param topic_diversity Whether to compute topic diversity or not (1 for yes,
#' 0 for no).
#' @param checkpoint_path The directory of location to the checkpoint file, which
#' contains the trained weights of the model.
#' @examples
#' etm_evaluate(dataset_name = "corpus_1", data_path = "Desktop/", num_topics = 10,
#' topic_coherence = 1, topic_diversity = 1, checkpoint_path = "Desktop/file"
#'
#' )
#' @export

etm_evaluate <- function(dataset_name,
                      data_path,
                      num_topics,
                      topic_coherence = 1,
                      topic_diversity = 1,
                      checkpoint_path) {
  os$system(paste("python R/ETM-master/main_etm.py --mode eval --dataset ",dataset_name, "--data_path ",data_path, "--num_topics", num_topics,"--train_embeddings 1 --tc ", topic_coherence, "--td ", topic_diversity, "--load_from ", checkpoint_path))

}
