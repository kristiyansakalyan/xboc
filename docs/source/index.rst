.. Explainable Bag-of-Concepts documentation master file, created by
   sphinx-quickstart on Fri Mar 15 13:44:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Explainable Bag-of-Concepts's documentation!
=======================================================

Bag-Of-Concepts (BOC) Implementation
--------------------------------------------------

The Bag-Of-Concepts (BOC) implementation is an advanced text processing module designed to enhance document embedding techniques by adding explainability. 

In comparsion to BERTopic
--------------------------------------------------
- c-CF-IDF normalization
- Explainable AI - compatibility with SHAP
- Calculate BIC, AIC using GMMs, silhouette, davies and calinski scores using a user-specified clustering method for a given list of values for K (number of concepts).

Limitations
--------------------------------------------------
- Spherical KMeans is slow.
- Cluster pollution of names in vector space (probably make 2D plots)
- Not the best scores most likely due to word vectors (in comparison to the BoC)

Changelog of the project in comparsion to BoC
--------------------------------------------------
This project implements a flexible BoC module with automatic concept labelling using LLMs.

- Automatic Concept Labeling
   - The user can use our predefined prompts for OpenAI's GPT3.5-Turbo
   - The user can provide his custom LangChain chain, that we invoke with the words that have to be labelled
   - The user can specify how many of the top N words belonging to a cluster to use
- Flexible Clustering
   - Spheircal KMeans (default one; used in the BoC paper)
   - KMeans
   - Spectral
- Ability to encode new documents
- Ability to save and load the model
- Get the top N words for a concept.
- Calculate BIC, AIC using GMMs, silhouette, davies and calinski scores using a user-specified clustering method for a given list of values for K (number of concepts).
- The output is compatible with SHAP values visualizations
   - The user can train any kind of model and use SHAP to visualize the feature importance. Examples:


.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

Contents
--------

.. toctree::

   usage
   explainability
   xboc