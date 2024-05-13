---
title: 'XBOC: Explainable Bag-Of-Concepts'
tags:
  - Python
  - natural language processing
  - document classification
  - document embeddings
  - explainable AI
  - topic modelling
authors:
  - name: Kristiyan Sakalyan
    equal-contrib: true
    affiliation: 1
  - name: Gerhard Johann Hagerer
    equal-contrib: true 
    affiliation: 1
  - name: Ahmed Mosharafa
    equal-contrib: true 
    affiliation: 1
affiliations:
 - name: Technical University of Munich (TUM), Germany
   index: 1
date: 08 April 2024
bibliography: paper.bib
---

# Summary
This paper introduces a novel approach to generating explainable document embeddings. It leverages enhanced Bag-of-Concepts methodology with a focus on transparency, allowing for clear insight into machine learning decision-making processes. This model not only maintains high accuracy and efficiency but also improves interpretability, making it suitable for applications requiring high levels of trust and accountability, such as legal or medical document analysis.

# Statement of Need
Traditional document embedding techniques often yield opaque outputs, limiting their usability in fields requiring transparency. This work addresses the need for explainable AI in natural language processing by providing a framework that enhances the interpretability of document embeddings without compromising performance. Our approach positions itself crucially among existing works like doc2vec, which focus more on performance than on explainability.

Document embeddings: 
- BERT (and everything else based on encoding Transformer models)
- doc2vec
- traditional NLP: bag-of-words, tf-idf based classification

Problems with these approaches:
- lack of explainable features: when you have a classifier and analyze SHAP and LIME values, these do not yield insights about how document contents relate to the classification outcome
- the features themselves do not explain what the document contents are about
- solution: word2vec based stuff for example bag-of-concepts which is build upon w2v based topic modeling. other examples are unsupervised aspect detection or graph-based w2v clustering, that give the most coherent clusters of texts and according topic distributions (unsupervised aspect extraction and graph-based topic extraction). these relate oftentimes well to pre-defined classed, but can also be used as features for classification, see bag-of-concepts.
- bag-of-concepts-like approaches have these explainable features. however, it cannot optimize the clustering automatically (number of clusters and the clustering algorithm). furthermore, the value of explainable features stays questionable if these compatible to explainable AI frameworks, such as, SHAP or LIME. last but not least, feature mining here involves manual labor to label the topics or concepts coherently and consistently.
- modern topic modelling approaches, such as, BERTopic, utilize large language models (LLMs) to automate the process of automated topic labeling. BERTopic, in turn, is primarily a topic modelling toolkit based on deep pre-trained text encoders that creates topic models based on short text embeddings. It is lacking out-of-the-box functionality to cluster texts which are longer than a typical context window. Furthermore, using BERTopic to process large corpora is compute intensive due to deep learning models, and GPUs are a necessity, which in industrial settings can be a hurdle even nowadays.

Our solution:
- Best of both worlds:
- From BERTopic we take topic labeling, i.e., interpretable features
- From bag-of-concepts we take efficiency (long texts, no GPUs)

# Features and Functionality

Our enhanced implementation of the Bag-of-Concepts framework introduces several key features that significantly extend its functionality and user flexibility, compared to the original implementation. 
These advancements are designed to provide a more robust, versatile, and user-friendly experience. 
Below, we delineate the novel features integrated into our module:

- Clustering Methods Selection: Users have the flexibility to choose from various clustering algorithms, enabling the selection of the most appropriate method based on the specific characteristics of their dataset.
    
- Document Encoding Post-Training: Our system allows for the encoding of new documents after the model has already been trained, facilitating the dynamic application of the model to new data without the need for retraining.
    
- Model Persistence: Users can save and subsequently load the trained model, enhancing the efficiency of repeated analyses and the deployment of models in different contexts.
    
- Retrieval of Top N Words: The module provides functionality to retrieve the top N words associated with each concept, offering insights into the semantic composition of the concept space.
    
- Concept Space Labeling: Through integration with a pre-defined or custom large language model pipeline, our framework supports the labeling of the concept space, thereby enriching the interpretability of the model's output.
    
- Optimality Scores Calculation: It incorporates the calculation of Bayesian Information Criterion (BIC), Akaike Information Criterion (AIC), and other scores to aid in determining the most optimal number of concepts for a given dataset.
    
- Interpretable Concept Space: The interpretable nature of the concept space enables the explanation of decisions taken by machine learning models applied on the Concept Frequency-Inverse Document Frequency (CF-IDF) values, thereby enhancing the transparency of model predictions.
    
- SHAP Values Visualization: Our implementation is compatible with the visualization of SHapley Additive exPlanations (SHAP) values (cite SHAP)

These features collectively aim to augment the analytical capabilities of researchers and practitioners, offering a comprehensive toolset for semantic analysis and model interpretation within the Bag-of-Concepts framework.

Our additions:
- we add compatibility to SHAP and LIME
- model persistence
- retrieval of top words
- topic labelling
- added inference

# Example Use Cases
The code repository of this paper contains an example for classification of news reports of the BBC News dataset. The newspaper articles are assigned to a given category by a classifier. Experiments show the accuracies shown in []{label="floatlabel"}. The notebooks also depict the SHAP values and explain which topics are related to which newspaper categories. This can be used to understand how topics are inter-related amongst each other.

| Model                | Accuracy (%) | Recall (%) | Precision (%) | F1-Score (%) |
|----------------------|--------------|------------|---------------|--------------|
| Logistic Regression  | 95.3         | 95.21      | 95.23         | 95.21        |
| Random Forest        | 95.5         | 95.31      | 95.37         | 95.32        |
[Performance metrics for Logistic Regression and Random Forest models using explainable document embeddings. Results reflect the models' ability to accurately classify documents in the test set.]{label="floatlabel"}



# Key References
- BOC
- BERTopic

# Acknowledgements
Special thanks to the contributors and colleagues from the Technical University of Munich (TUM) who provided insights and expertise that greatly assisted the research.
