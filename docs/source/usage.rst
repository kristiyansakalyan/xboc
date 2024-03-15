Usage
=====

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install xboc


Training
------------

The default usage is to just fit the model to a corpus. The **boc_matrix** then contains the document embeddings of each document.

.. code-block:: python
    
    boc_model = BOCModel(
        docs_train,
        word_vectors,
        idx2word,
    )
    boc_matrix, word2concept_list, idx2word_converter = boc_model.fit()

Training with automatic concept labeling
-----------------------------------------

.. code-block:: python
    
    boc_model = BOCModel(
        docs_train,
        word_vectors,
        idx2word, 
        tokenizer=CustomTokenizer(),
        n_concepts=20,
        label_impl=LabelingImplementation.TEMPLATE_CHAIN,
        llm_model=LLMModel.OPENAI_GPT3_5
    )
    boc_matrix, word2concept_list, idx2word_converter = boc_model.fit()

Further usage examples
----------------------

For more details on how to use the BoC model, please take a look at [the DEMO notebook.](notebooks/DEMO-Notebook.ipynb).

you can use the ``xboc.BOCModel()`` function:

.. autofunction:: xboc.BOCModel()