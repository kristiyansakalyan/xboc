Explainability with SHAP values
================================

Here you can see a few examples on how to fit a 
machine learning model and then use SHAP values to analyze its' decisions.

Logistic Regression
----------------------
.. code-block:: python
        
    explainer = shap.LinearExplainer(log_reg, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=boc_model.get_concept_label())

Support Vectors
----------------------
.. code-block:: python
    
    X_train_summary = shap.kmeans(docs_train_embedded, 50)
    explainer = shap.KernelExplainer(svm.predict, X_train_summary)
    shap_values = explainer.shap_values(docs_test_np)
    shap.summary_plot(shap_values, docs_test_np, feature_names=boc_model.get_concept_label())


Random Forest
----------------------
.. code-block:: python
    
    explainer = shap.TreeExplainer(random_forest)
    shap_values = explainer.shap_values(docs_test_np)
    shap.summary_plot(shap_values, docs_test_np, feature_names=boc_model.get_concept_label())

XGBoost
----------------------
.. code-block:: python
    
    explainer = shap.TreeExplainer(xgb_classifier)
    shap_values = explainer.shap_values(docs_test_np)
    shap.summary_plot(shap_values, docs_test_np, feature_names=boc_model.get_concept_label())


KNN
----------------------
.. code-block:: python
    
    explainer = shap.KernelExplainer(knn.predict, docs_train_embedded) 
    shap_values = explainer.shap_values(docs_test_np)
    shap.summary_plot(shap_values, docs_test_np, feature_names=boc_model.get_concept_label())
