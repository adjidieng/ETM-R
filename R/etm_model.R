#' Create Word Embeddings and Run the Embedded Topic Model on a Corpus
#'
#' @description
#' `etm_model` combines two steps into one: first, it creates word embeddings on a
#' corpus, and second, it then runs the embedded topic model.  Before using `etm_model`,
#' use `etm_preprocess` to process the corpus into a bag-of-words representation (
#' tokens and counts).  An optional step before using `etm_model` is to use `etm_embed` (or another means)
#' to pre-fit word embeddings on the corpus, and hence, skip the first step of `etm_model`.
#' @param dataset_name The folder name of the preprocessed data (e.g. "20ng").  Remember to run `etm_preprocess`
#' to produce this folder of bag-of-words representations.
#' @param data_path The directory or location to the preprocessed data (e.g. "~/Desktop/20ng").
#' @param num_topics The number of topics to learn.  The default number of topics is 50.
#' @param epochs Number of times the dataset is fed into the algorithm for
#' learning.  The default number of epochs is 1000.  Minimum number of epochs is 1.
#' @param save_path The directory for the output.  The default is "./results".
#' @param use_embed Boolean entry (TRUE or FALSE) to skip creating word embeddings
#' and use pre-fitted embeddings.
#' @param embed_path The path to the pre-fitted word embeddings file, if `use_embed` is TRUE.  Otherwise,
#' leave blank.
#' @return
#' A text file, known as the checkpoint file, which contains the trained embeddings and topics.  This
#' file is saved at the designated save
#' @examples
#' \dontrun{
#' # most minimal example
#' etm_train(dataset_name = "bow_output", data_path = "Desktop/bow_output", num_topics = 2,
#' epochs = 2, save_path = "Desktop/", use_embed = FALSE)
#'
#' # create word embeddings and then run the embedded topic model
#' etm_train(dataset_name = "bow_output", data_path = "Desktop/bow_output", num_topics = 10,
#' epochs = 1000, save_path = "Desktop/", use_embed = FALSE)
#'
#'# use pre-fitted word embeddings, and then run the embedded topic model
#' etm_train(dataset_name = "prefitted_data", data_path = "Desktop/", num_topics = 10,
#' epochs = 1000, save_path = "Desktop/",
#' use_embed = TRUE, embed_path = "Desktop/embeddings/")
#' }
#' @export

etm_model <- function(dataset_name,
                      data_path,
                      num_topics = 50,
                      epochs = 1000,
                      save_path,
                      use_embed,
                      embed_path) {
  if(use_embed == FALSE){
    subprocess$call(paste("python inst/python/main_etm.py --mode train --dataset ",dataset_name, "--data_path ",data_path, "--num_topics ", num_topics,"--train_embeddings 1 --epochs ", epochs,"--save_path ", save_path),
                    shell = TRUE)
  } else if(use_embed == TRUE){
    subprocess$call(paste("python inst/python/main_etm.py --mode train --dataset ",dataset_name, "--data_path ",data_path, "--emb_path", embed_path, "--num_topics ", num_topics, "--train_embeddings 0 --epochs ", epochs),
                    shell = TRUE)
  } else{
    message("Error: for the parameter use_embed, specify either TRUE or FALSE")
  }

}
