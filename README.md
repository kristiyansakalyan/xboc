# Changelog of the project in comparsion to BoC
This project implements a flexible BoC module with automatic concept labelling using LLMs.

- Automatic Concept Labeling
  - The user can use our predefined prompts for OpenAI's GPT3.5-Turbo
  - The user can provide his custom LangChain chain, that we invoke with the words that have to be labelled
  - The user can specify how many of the top N words belonging to a cluster to use
- Flexible Clustering
  - Spheircal KMeans (default one; used in the BoC paper)
  - KMeans
  - Agglomerative Clustering
  - Spectral
- Ability to encode new documents
- Ability to save and load the model
- Get the top N words for a concept.
- Calculate BIC, AIC using GMMs, silhouette, davies and calinski scores using a user-specified clustering method for a given list of values for K (number of concepts).
- The output is compatible with SHAP values visualizations
  - The user can train any kind of model and use SHAP to visualize the feature importance. Examples:

```
# Logistic Regression
explainer = shap.LinearExplainer(log_reg, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=boc_model.get_concept_label())

# Support Vectors
X_train_summary = shap.kmeans(docs_train_embedded, 50)
explainer = shap.KernelExplainer(svm.predict, X_train_summary)
shap_values = explainer.shap_values(docs_test_np)
shap.summary_plot(shap_values, docs_test_np, feature_names=boc_model.get_concept_label())

# Random Forest
explainer = shap.TreeExplainer(random_forest)
shap_values = explainer.shap_values(docs_test_np)
shap.summary_plot(shap_values, docs_test_np, feature_names=boc_model.get_concept_label())

# XGBoost
explainer = shap.TreeExplainer(xgb_classifier)
shap_values = explainer.shap_values(docs_test_np)
shap.summary_plot(shap_values, docs_test_np, feature_names=boc_model.get_concept_label())

# KNN
explainer = shap.KernelExplainer(knn.predict, docs_train_embedded) 
shap_values = explainer.shap_values(docs_test_np)
shap.summary_plot(shap_values, docs_test_np, feature_names=boc_model.get_concept_label())
```

# In comparsion to BERTopic
- c-CF-IDF normalization
- Explainable AI - compatibility with SHAP
- Calculate BIC, AIC using GMMs, silhouette, davies and calinski scores using a user-specified clustering method for a given list of values for K (number of concepts).

# Limitations
- Spherical KMeans is slow.
- Cluster pollution of names in vector space (probably make 2D plots)
- Not the best scores most likely due to word vectors (in comparison to the BoC)
