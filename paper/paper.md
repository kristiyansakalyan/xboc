---
title: 'XBOC: Explainable Bag-Of-Concepts'
tags:
  - Python
  - xai
  - nlp
  - ml
  - document embeddings
  - bag-of-concepts
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

This work enhances the Bag-of-Concepts framework by introducing an automated approach to concept labeling for document embeddings, achieving a balance between model performance and explainability. By applying spherical k-means clustering and a large language model pipeline, we provide a clear and automated interpretation of the latent space. We incorporate SHAP values to reveal the influence of each concept on model predictions across various document classifications. Our findings demonstrate that it is possible to create machine learning models that are both accurate and transparent, contributing to the advancement of trustworthy artificial intelligence.

# Statement of need

In the rapidly evolving landscape of machine learning, there is a pressing need for models that not only excel in performance but are also transparent and interpretable. `XBOC` is a package that enhances the Bag-of-Concepts framework to offer automated concept labeling for document embeddings. This approach introduces explainability by utilizing spherical k-means clustering alongside a large language model pipeline to interpret the latent spaces. It is designed to be used by NLP researchers and students in the field of Machine Learning.