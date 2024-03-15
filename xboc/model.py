"""
Bag-Of-Concepts implementation following the code from the original authors and
improving it. For reference: https://github.com/hank110/bagofconcepts/blob/master/bagofconcepts/boc.py
"""

import logging
import pickle
from collections import Counter
from typing import Any, Type

import numpy as np
import pandas as pd
import scipy.sparse
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables.base import RunnableSerializable
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.mixture import GaussianMixture
from sklearn.utils.extmath import safe_sparse_dot
from soyclustering import SphericalKMeans
from typing_extensions import Self

from xboc.prompts import get_labeling_config
from xboc.types import ClusteringMethod, LabelingImplementation, LLMModel, Tokenizer
from xboc.utils import setup_logging


class XBOCModel:
    """
    A Bag-Of-Concepts model that is implemented following the original paper:
    https://www.sciencedirect.com/science/article/abs/pii/S0925231217308962

    The model is also compatible with the BERTopic pipeline and can be used
    to embed documents.

    """

    def __init__(
        self: Self,
        corpus: list[list[str]] | np.ndarray,
        wv: np.ndarray,
        idx2word: dict[int, str],
        tokenizer: Tokenizer | None = None,
        clustering_method: ClusteringMethod = ClusteringMethod.Spherical_KMeans,
        n_concepts: int = 100,
        iterations: int = 100,
        random_state: int = 42,
        verbose: bool = False,
        label_impl: LabelingImplementation | None = None,
        llm_model: LLMModel = LLMModel.OPENAI_GPT3_5,
        custom_chain: RunnableSerializable[dict[Any, Any], BaseMessage] | None = None,
        n_top_words_label: int = 10,
        log_level: int = logging.INFO,
    ) -> None:
        """Initializes the Bag-Of-Concept model.

        Parameters
        ----------
        corpus : list[list[str]] | np.ndarrayy
            The *preprocessed* corpus to train the model on.
        wv : np.ndarray
            Word vector representations
        idx2word : dict[int, str]
            Index to word mapping
        tokenizer : Tokenizer | None
            The tokenizer used to tokenize the corpus. If not provided
            no new documents can be embedded. by default None
        clustering_method : ClusteringMethod
            The clustering method to use, by default Spherical KMeans
        n_concepts : int, optional
            Number of concepts, by default 100
        iterations : int, optional
            Numbeer of iterations., by default 100
        random_state : int, optional
            Random state to avoid stochasticity., by default 42
        verbose : bool, optional
            Verbose or not, by default False
        label_impl: LabelingImplementation | None
            Whether to use template chains or a custom chain and whether to
            label concepts at all, by default None
        llm_model: LLMModel
            Large language model to be used for the supported templates, by default OPENAI_GPT3_5
        custom_chain: RunnableSerializable[dict[Any, Any], BaseMessage] | None
            The custom chain to be used if the user has specified it, by default None
        n_top_words_label : int
            How many top words to use to label the concepts, by default 10
        log_level: int
            Log level if verbose is True, by default INFO.
        """
        self.corpus = corpus
        self.wv = wv
        self.idx2word = idx2word
        self.n_concepts = n_concepts
        self.iterations = iterations
        self.random_state = random_state
        self.tokenizer = tokenizer
        self.clustering_method = clustering_method
        self.verbose = verbose
        self.label_impl = label_impl
        self.llm_model = llm_model
        self.custom_chain = custom_chain
        self.n_top_words_label = n_top_words_label

        self.wv_cluster_id: np.ndarray = np.array([], dtype=np.int16)
        self.bow: scipy.sparse.spmatrix = scipy.sparse.csr_matrix([])
        self.w2c: scipy.sparse.spmatrix = scipy.sparse.csr_matrix([])
        self.boc: np.ndarray | scipy.sparse.spmatrix = scipy.sparse.csr_matrix([])
        self.icf: np.ndarray = np.array([], dtype=np.float64)
        self.word2idx: dict[str, int] = self._get_word2idx(idx2word)
        self.ranked_cluster_indicies: list[np.ndarray] = []
        self.concept_labels: list[str] = [""] * self.n_concepts

        if label_impl == LabelingImplementation.CUSTOM_CHAIN and custom_chain is None:
            raise AttributeError(
                "If you want to use a custom chain, you must provide a custom_chain implementation!"
            )

        self.logger: logging.Logger | None = None
        if self.verbose:
            self.logger = setup_logging(__name__, log_level=log_level)

    def fit(
        self: Self,
    ) -> tuple[
        np.ndarray | scipy.sparse.spmatrix, list[tuple[str, int]], dict[int, str]
    ]:
        """Fit the model on the training data

        Returns
        -------
        tuple[np.ndarray | scipy.sparse.spmatrix, list[str, int], dict[int, str]]
            1. Bag of Concept matrix, where the rows represent the document's embeddings.
            2. A list of word to concept mappings.
            3. A index to word converter as a dictionary.
        """
        # Verbose logging
        self._log("1. Cluster the word vectors.")
        self._cluster_wv(self.wv, self.n_concepts, max_iter=self.iterations)

        self._log("2. Create a bag of words.")
        self._create_bow()

        self._log("3. Map words to concepts.")
        self._create_w2c()

        self._log("4. Compute the CF-IDF")
        self._apply_cfidf(safe_sparse_dot(self.bow, self.w2c))  # type: ignore

        if self.label_impl is not None:
            self._log("5. Automatically label the concepts")
            self._label_concepts()

        return (
            self.boc,
            [wc_pair for wc_pair in zip(self.idx2word, self.wv_cluster_id)],  # type: ignore
            self.idx2word,
        )

    def encode(self: Self, text: list[str] | str) -> scipy.sparse.spmatrix | np.ndarray:
        """Encodes new documents

        Parameters
        ----------
        text : list[str] | str
            Text to be encoded

        Returns
        -------
        scipy.sparse.spmatrix | np.ndarray
            The document embedding

        Raises
        ------
        ValueError
            Model must be first trained before it can encode documents.
        AttributeError
            A tokenizer must be specified in order to encode new documents.
        """
        if (
            self.boc.shape == (0, 0)
            or self.icf.shape == (0, 0)
            or self.w2c.shape == (0, 0)
        ):
            raise ValueError("Please fit the model before trying to encode any data.")

        if self.tokenizer is None:
            raise AttributeError(
                "If you want to use the BOC model to encode new document, you must set the tokenizer attribute"
            )

        text_processed = text
        # If the text is not tokenized, tokenize it.
        # We assume that if it is a list of strings then the text is not tokenized.
        if isinstance(text, str):
            text_processed = self.tokenizer(text)

        # Create the bag of words representation
        doc_bow = np.zeros(len(self.idx2word))
        for token in text_processed:
            if token in self.word2idx:
                doc_bow[self.word2idx[token]] += 1.0
        doc_bow = scipy.sparse.csr_matrix(doc_bow)

        # Convert to bag of concepts representation
        doc_boc = safe_sparse_dot(doc_bow, self.w2c)

        # Apply the ICF
        return safe_sparse_dot(doc_boc, scipy.sparse.diags(self.icf))

    def save(self: Self, folder_path: str) -> None:
        """Saves the model using pickle.

        IMPORTANT
        ----------
        Custom chains cannot be saved. Please make sure that
        after you load your BoC model, you set manually the
        custom chain.

        Parameters
        ----------
        folder_path : str
            The folder where the model should be saved.
        """
        file_name = f"BoC_c{self.n_concepts}.pickle"
        file_path = f"{folder_path}/{file_name}"

        # Cannot save custom Langchain chains.
        temporary_custom_chain = self.custom_chain
        self.custom_chain = None

        with open(file_path, "wb") as file:
            pickle.dump(self, file)
        self._log(f"Successfully saved the model at: {file_path}")

        self.custom_chain = temporary_custom_chain

    def get_top_n_words(
        self, n: int, cluster: int | None = None
    ) -> list[list[str]] | list[str]:
        """Gets the top N words for all clusters or for a given cluster if provided.

        Parameters
        ----------
        n : int
            Number of words to get
        cluster : int | None, optional
            Cluster of interest, by default None

        Returns
        -------
        list[list[str]] | list[str]
            Top N words for all clusters or for a given cluster

        Raises
        ------
        AttributeError
            Something went wrong during training.
        IndexError
            The number N is higher than the minimum number of words assigned to clusters.
        IndexError
            The cluster of interest is out of range.
        """
        if self.ranked_cluster_indicies is None:
            raise AttributeError("The word to cluster ranking attribute is None.")

        lengths = np.min([len(x) for x in self.ranked_cluster_indicies])
        if n > np.min(lengths):
            raise IndexError(
                f"The number of provided words exceeds the minimum \
                number of words assigned to one of the clusters. Lengths: {lengths}"
            )

        # Get the top N word vectors for each cluster
        top_n_indicies = [indices[:n] for indices in self.ranked_cluster_indicies]

        # Get the corresponding words using the ranked indices
        top_n_words_per_cluster = [
            [self.idx2word[i] for i in per_cluster_ranking]
            for per_cluster_ranking in top_n_indicies
        ]

        # If a cluster of interest is provided, return only its top N words
        if cluster is not None:
            if cluster >= self.n_concepts or cluster < 0:
                raise IndexError(
                    f"The cluster index can only be in the range [0, {self.n_concepts})"
                )

            return top_n_words_per_cluster[cluster]

        return top_n_words_per_cluster

    def get_concept_label(self: Self, index: int | None = None) -> list[str] | str:
        """Return all concept labels or specific label.

        Parameters
        ----------
        index : int | None, optional
            Concept index, by default None

        Returns
        -------
        np.ndarray | str
            The concept label or all concept labels

        Raises
        ------
        IndexError
            If concept index is out of range
        """
        if index is None:
            return self.concept_labels

        if index >= len(self.concept_labels) or index < 0:
            raise IndexError(
                f"Concept index out of range: {index}; Max: {self.n_concepts - 1}"
            )

        return self.concept_labels[index]

    @classmethod
    def load(cls: Type[Self], file_path: str) -> "XBOCModel":
        """Loads the model using pickle.

        Parameters
        ----------
        file_path : str
            The path to the file.
        """
        with open(file_path, "rb") as file:
            model: XBOCModel = pickle.load(file)

        return model

    @classmethod
    def calculate_scores_for_k_range(
        cls: Type[Self],
        k_range: list[int],
        wv: np.ndarray,
        clustering_method: ClusteringMethod = ClusteringMethod.Spherical_KMeans,
        max_iter: int = 100,
        verbose: bool = False,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Calculates the BIC, AIC using GMMs on the provided word vectors
        using the K number of clusters in the provided list. Additionaly,
        this function calculates the silhouette, davies and calinski scores
        for the same list of k number of clusters, using the provided
        clustering method.

        Parameters
        ----------
        k_range : list[int]
            The grid parameter search range
        wv : np.ndarray
            Word Vectors
        clustering_method : ClusteringMethod
            The clustering method to use for calculating
            the silhouette, davies and calinski scores.
            by default Spherical_KMeans
        max_iter : int, optional
            Maximum number of iteration for the clustering model,
            by default 100
        verbose : bool, optional
            Verbosity, by default False
        random_state : int, optional
            Random state to be used, by default 42

        Returns
        -------
        pd.DataFrame
            Dataframe consisting all the results
        """
        k_arr = list(k_range)
        bic_arr = []
        aic_arr = []
        sil_arr = []
        db_arr = []
        cal_arr = []

        for n in k_range:
            _, y_pred = XBOCModel.fit_predict_clustering_method(
                wv, n, clustering_method, max_iter, verbose, random_state
            )

            # Fit GMMs
            gmm = GaussianMixture(
                n_components=n,
                init_params="k-means++",
                max_iter=max_iter,
                verbose=verbose,
            )
            gmm.fit(wv)

            # Calculate BIC and AIC
            bic, aic = gmm.bic(wv), gmm.aic(wv)

            bic_arr.append(bic)
            aic_arr.append(aic)

            # Calculate other scores
            sil = silhouette_score(wv, y_pred)
            db = davies_bouldin_score(wv, y_pred)
            cal = calinski_harabasz_score(wv, y_pred)

            sil_arr.append(cal)
            db_arr.append(db)
            cal_arr.append(cal)

            print(f"K={n}: BIC: {bic}; AIC: {aic}; SIL:{sil}; DB: {db}; CAL: {cal}")

        return pd.DataFrame(
            {
                "k": k_arr,
                "BIC": bic_arr,
                "AIC": aic_arr,
                "silhouette": sil_arr,
                "davies": db_arr,
                "calinski": cal_arr,
            }
        )

    @classmethod
    def fit_predict_clustering_method(
        cls: Type[Self],
        wv: np.ndarray,
        k: int,
        clustering_method: ClusteringMethod,
        max_iter: int = 100,
        verbose: bool = False,
        random_state: int = 42,
    ) -> tuple[
        KMeans | SpectralClustering | SphericalKMeans,
        np.ndarray,
    ]:
        """Fits a cluster model on the provided data, given the number of
        clusters K to use. Then predicts the clusters for the provided data
        and returns both the fitted cluster model and the predictions for the data.

        Parameters
        ----------
        wv : np.ndarray
            Word Vectors
        k : int
            Number of clusters
        clustering_method : ClusteringMethod
            Clustering method to be used
        max_iter : int, optional
            Maximum number of iteration for the clustering model,
            by default 100
        verbose : bool, optional
            Verbosity, by default False
        random_state : int, optional
            Random state to be used, by default 42

        Returns
        -------
        tuple[ KMeans | SpectralClustering | SphericalKMeans, np.ndarray, ]
            The fitted cluster model and the predictions for each word vector.
        """
        if clustering_method == ClusteringMethod.KMeans:
            cluster_model = KMeans(
                n_clusters=k,
                n_init="auto",
                max_iter=max_iter,
                verbose=verbose,
                random_state=random_state,
            )
        elif clustering_method == ClusteringMethod.Spectral:
            cluster_model = SpectralClustering(
                n_clusters=k,
                verbose=verbose,
                random_state=random_state,
            )
        else:
            cluster_model = SphericalKMeans(
                n_clusters=k,
                max_iter=max_iter,
                verbose=verbose,
                init="similar_cut",
                sparsity="None",
                random_state=random_state,
            )

        sM = scipy.sparse.csr_matrix(wv)
        return cluster_model, cluster_model.fit_predict(sM)

    def _cluster_wv(
        self: Self, wv: np.ndarray, num_concept: int, max_iter: int = 10
    ) -> None:
        """Cluster word vector representations using the provided
        clustering method.

        Parameters
        ----------
        wv : np.ndarray
            The word vector representations.
        num_concept : int
            Number of concepts to cluster.
        max_iter : int, optional
            Maximum number of iterations for the clustering method., by default 10

        Raises
        ------
        ValueError
            If the word vectors that have to be clustered contain any NaN values.
        """

        if np.isnan(wv).any():
            raise ValueError(
                "NaN values found in word vectors. Handling strategy needed."
            )

        sM = scipy.sparse.csr_matrix(wv)
        cluster_model, self.wv_cluster_id = XBOCModel.fit_predict_clustering_method(
            wv,
            num_concept,
            self.clustering_method,
            max_iter,
            self.verbose,
            self.random_state,
        )

        # The following code block ensures the top N words functionality
        centroids = None
        if isinstance(cluster_model, KMeans):
            centroids = cluster_model.cluster_centers_
        else:
            # For Spectral, and SphericalKMeans, compute centroids manually
            centroids = np.array(
                [
                    sM[self.wv_cluster_id == i].mean(axis=0)
                    for i in range(self.n_concepts)
                ]
            )
            centroids = centroids.squeeze(axis=1)

        # Compute distances from each word vector to its cluster centroid
        distances = cosine_distances(sM, centroids)

        # Get indices of word vectors within each cluster, sorted by distance
        self.ranked_cluster_indicies = [
            np.argsort(distances[:, i]) for i in range(self.n_concepts)
        ]

        # Ranking of words per cluster only for the words assigned to that cluster
        for index, word2concept_rank in enumerate(self.ranked_cluster_indicies):
            new_array = np.array(
                [x for x in word2concept_rank if self.wv_cluster_id[x] == index]
            )
            self.ranked_cluster_indicies[index] = new_array

    def _create_bow(self: Self) -> None:
        """Create the bag of words that is needed for compute the CF-IDF"""
        rows = []
        cols = []
        vals = []
        word2idx = {word: idx for idx, word in enumerate(self.idx2word)}

        for i, doc in enumerate(self.corpus):
            tokens_count = Counter(
                [word2idx[token] for token in doc if token in word2idx]
            )
            for idx, count in tokens_count.items():
                rows.append(i)
                cols.append(idx)
                vals.append(float(count))
        self.bow = csr_matrix((vals, (rows, cols)), shape=(i + 1, len(word2idx)))  # type: ignore

    def _create_w2c(self: Self) -> None:
        """Create the word to concept mapping.

        Raises
        ------
        IndexError
            When the dimensions between words and labels do not match.
        """
        if len(self.idx2word) != len(self.wv_cluster_id):  # type: ignore
            raise IndexError(
                f"Dimensions between words and labels mismatched: {len(self.idx2word)} != {len(self.wv_cluster_id)}"
            )

        rows = [i for i, _ in enumerate(self.idx2word)]
        cols = [j for j in self.wv_cluster_id]  # type: ignore
        vals = [1.0 for i in self.idx2word]

        self.w2c = csr_matrix(
            (vals, (rows, cols)), shape=(len(self.idx2word), self.n_concepts)
        )

    def _apply_cfidf(
        self: Self, csr_matrix: np.ndarray | scipy.sparse.spmatrix
    ) -> None:
        """Applies the Concept-Frequency Inverse-Concept-Frequency.

        Parameters
        ----------
        csr_matrix : np.ndarray | scipy.sparse.spmatrix
            The embedded documents using the concepts.
        """
        num_docs, num_concepts = csr_matrix.shape
        _, nz_concept_idx = csr_matrix.nonzero()  # type: ignore
        cf = np.bincount(nz_concept_idx, minlength=num_concepts)
        icf = np.log(num_docs / cf)
        icf[np.isinf(icf)] = 0

        # Save ICF
        self.icf = icf

        self.boc = safe_sparse_dot(csr_matrix, scipy.sparse.diags(icf))

    def _get_word2idx(self: Self, idx2word: dict[int, str]) -> dict[str, int]:
        """Creates the reverse mapping of index to word.

        Parameters
        ----------
        idx2word : dict[int, str]
            Index to word mapping

        Returns
        -------
        dict[str, int]
            Word to index mapping
        """
        return {str(word): idx for idx, word in enumerate(idx2word)}

    def _label_concepts(self: Self) -> None:
        """Automatically labels the concepts using the top 10 words."""

        if (
            self.label_impl == LabelingImplementation.CUSTOM_CHAIN
            and self.custom_chain is not None
        ):
            chain = self.custom_chain
        else:
            # Get prompt and model config and then create the chain
            prompt, model = get_labeling_config(self.llm_model)
            chain = prompt | model

        # Get the top words for each concept
        top_words_per_cluster = self.get_top_n_words(self.n_top_words_label)

        for index, keywords in enumerate(top_words_per_cluster):
            keywords_str = ", ".join(keywords)

            try:
                label = chain.invoke({"keywords": keywords_str}).content
            except Exception as e:
                raise Exception(f"Could not label concept {index}. Exception: {e}")

            # Verbose logging
            self._log(f"Conept label: {label}\nKeywords: {keywords_str}")

            self.concept_labels[index] = label  # type: ignore

    def _log(self: Self, msg: str) -> None:
        """Log messasges if verbose is enabled.

        Parameters
        ----------
        msg : str
            Message to be logged
        """
        if self.verbose and self.logger is not None:
            self.logger.info(msg)
