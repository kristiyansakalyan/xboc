"""Define BoC-related types."""

from abc import ABC, abstractmethod
from enum import Enum, auto


class LLMModel(Enum):
    """All model-specific prompts implementations"""

    OPENAI_GPT3_5 = auto()


class LabelingImplementation(Enum):
    """Labeling LangChain Implementation to use. The user can specify
    whether to use our pre-defined templates or a custom
    langchain that he provides.
    """

    CUSTOM_CHAIN = auto()
    TEMPLATE_CHAIN = auto()


class ClusteringMethod(Enum):
    """Define all supported clustering methods."""

    KMeans = auto()
    Spherical_KMeans = auto()
    Spectral = auto()


class Tokenizer(ABC):
    """
    Abstract wrapper around tokenizers.
    The call method should be implemented to allow
    tokenization of new documents.

    Ensures compatibility with pickle.
    """

    @abstractmethod
    def __call__(self, text: str) -> list[str]:
        """Tokenizes the input text and returns a list of tokens.

        Parameters
        ----------
        text : str
            The input text to tokenize.

        Returns
        -------
        list[str]
            A list of tokens extracted from the input text.
        """
        pass
