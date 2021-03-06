% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/etm_evaluate.R
\name{etm_evaluate}
\alias{etm_evaluate}
\title{Evaluate Results of the Embedded Topic Modeling}
\usage{
etm_evaluate(dataset_name, data_path, num_topics, viz_num_words = 10,
  topic_coherence = 1, topic_diversity = 1, checkpoint_path)
}
\arguments{
\item{dataset_name}{The name of the dataset.}

\item{data_path}{The directory or location to the dataset.}

\item{num_topics}{The number of topics to learn.}

\item{viz_num_words}{Number of top words to visualize for each topic.  Default is 10.}

\item{topic_coherence}{Whether to compute topic coherence or not (1 for yes,
0 for no).}

\item{topic_diversity}{Whether to compute topic diversity or not (1 for yes,
0 for no).}

\item{checkpoint_path}{The directory of location to the checkpoint file, which
is the output from `etm_model`.  The checkpoint file contains the trained weights of the
embedded topic model.}
}
\description{
`etm_evaluate` evaluates the results of the embedded topic model.  `etm_evaluate` can evaluate
perplexity on document completion, topic coherence and topic diversity.  This function can
also visualize the topics and word embeddings.
}
\examples{
\dontrun{
etm_evaluate(dataset_name = "corpus_1", data_path = "Desktop/", num_topics = 10,
topic_coherence = 1, topic_diversity = 1, checkpoint_path = "Desktop/file")
}
}
