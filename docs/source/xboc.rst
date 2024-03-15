API
===

XBOCModel
----------

.. autofunction:: xboc.XBOCModel()
.. autofunction:: xboc.XBOCModel.__init__
.. autofunction:: xboc.XBOCModel.fit
.. autofunction:: xboc.XBOCModel.encode
.. autofunction:: xboc.XBOCModel.save
.. autofunction:: xboc.XBOCModel.get_top_n_words
.. autofunction:: xboc.XBOCModel.get_concept_label
.. autofunction:: xboc.XBOCModel.load
.. autofunction:: xboc.XBOCModel.calculate_scores_for_k_range
.. autofunction:: xboc.XBOCModel._cluster_wv
.. autofunction:: xboc.XBOCModel._create_bow
.. autofunction:: xboc.XBOCModel._create_w2c
.. autofunction:: xboc.XBOCModel._apply_cfidf
.. autofunction:: xboc.XBOCModel._get_word2idx
.. autofunction:: xboc.XBOCModel._log

Types
----------

.. autofunction:: xboc.LLMModel()
.. autofunction:: xboc.LabelingImplementation()
.. autofunction:: xboc.ClusteringMethod()
.. autofunction:: xboc.Tokenizer()
.. autofunction:: xboc.Tokenizer.__call__


Prompts
----------

.. autofunction:: xboc.prompts.UniversalPrompt()
.. autofunction:: xboc.prompts.OpenAIPrompt()