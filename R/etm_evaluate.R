#' Evaluate Results of the Embedded Topic Modeling
#'
#' @description
#' `etm_evaluate` evaluates the results of the embedded topic model.  `etm_evaluate` can evaluate
#' perplexity on document completion, topic coherence and topic diversity.  This function can
#' also visualize the topics and word embeddings.
#' @param dataset_name The name of the dataset.
#' @param data_path The directory or location to the dataset.
#' @param num_topics The number of topics to learn.
#' @param topic_coherence Whether to compute topic coherence or not (1 for yes,
#' 0 for no).
#' @param topic_diversity Whether to compute topic diversity or not (1 for yes,
#' 0 for no).
#' @param viz_num_words Number of top words to visualize for each topic.  Default is 10.
#' @param checkpoint_path The directory of location to the checkpoint file, which
#' is the output from `etm_model`.  The checkpoint file contains the trained weights of the
#' embedded topic model.
#' @examples
#' \dontrun{
#' etm_evaluate(dataset_name = "corpus_1", data_path = "Desktop/", num_topics = 10,
#' topic_coherence = 1, topic_diversity = 1, checkpoint_path = "Desktop/file")
#' }
#' @export

etm_evaluate <- function(dataset_name,
                         data_path,
                         num_topics,
                         viz_num_words = 10,
                         topic_coherence = 1,
                         topic_diversity = 1,
                         checkpoint_path) {
  subprocess$call(paste("python inst/main_etm.py --mode eval --dataset ",dataset_name, "--data_path ",data_path, "--num_topics", num_topics, "--num_words", viz_num_words, "--train_embeddings 1 --tc ", topic_coherence, "--td ", topic_diversity, "--load_from ", checkpoint_path),
                  shell = TRUE)

}
