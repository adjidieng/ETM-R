#' Preprocess the Corpus
#'
#' @description
#' `etm_preprocess` preprocesses the corpus to convert the text into bag-of-words
#' representation (tokens and counts).  `etm_preprocess` is the precursor step before
#' using word embeddings (`etm_embed`) or modeling the embedded topic model (`etm_main`).
#' @param data_path Path to the corpus text file.
#' @param max_df The desired value for maximum document frequency.  Default is 0.7.
#' @param min_df The desired value for minimum document frequency.  Default is 100.
#' @return A folder with ".mat" files of bag-of-words representation.  The folder will be
#' located in the same directory as the corpus text file.
#' @examples
#' \dontrun{
#' etm_preprocess(data_path = "Desktop/corpus.txt")
#'
#' etm_preprocess(data_path = "Documents/news.txt", max_df = 0.8, min_df = 100)
#' }
#' @export

etm_preprocess <- function(data_path,
                           max_df = 0.7,
                           min_df = 100) {
  subprocess$call(paste("python inst/python/data_nyt_args.py --data_file ", data_path, "--max_df ", max_df, "--min_df ", min_df),
                  shell = TRUE)
}

