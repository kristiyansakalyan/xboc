"""Define the prompts that can be used with different models."""

from abc import ABC

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_openai.chat_models import ChatOpenAI

from xboc.types import LLMModel


class UniversalPrompt(ABC):
    """An abstract class that defines that each model-specific
    prompt implementation should have the following attribute:

    Attributes
    ----------
    prompt : str
        The main prompt to be used. MUST CONTAIN "{keywords}"!
    """

    prompt: str


class OpenAIPrompt(UniversalPrompt):
    """OpenAI specific prompt.

    Attributes
    ----------
    prompt : str
        The main prompt to be used
    """

    _system_prompt = """
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant for labeling multiple keywords into an abstract concept.
    <</SYS>>
    """

    _example_prompt = """
    I have an abstract concept that contains the following keywords:
    france, germany, italy, spain, portugal, denmark, switzerland, netherlands, belgium, britain

    Based on the information about the abstract concept above, please create a short label of this concept. Make sure you to only return the label and nothing more.
    [/INST] Name of countries
    """

    _example_prompt_2 = """
    [INST]
    I have an abstract concept that contains the following keywords:
    apple, orange, banana, grape, mango, pineapple, kiwi, strawberry, peach, plum

    Based on the information about the abstract concept above, please create a short label of this concept. Make sure you only return the label and nothing more.
    [/INST] Fruits
    """

    _main_prompt = """
    [INST]
    I have an abstract concept that contains the following keywords:
    {keywords}

    Based on the information about the abstract concept above, please create a short label of this concept. Make sure you to only return the label and nothing more.
    [/INST]
    """
    prompt = _system_prompt + _example_prompt + _example_prompt_2 + _main_prompt


def get_labeling_config(
    llm_model: LLMModel,
) -> tuple[ChatPromptTemplate, BaseChatModel]:
    """Get the model and the model-specific prompt template.

    Parameters
    ----------
    model_prompt : ModelPrompts
        The model-specific prompt to be used.

    Returns
    -------
    tuple[ChatPromptTemplate, BaseChatModel]
        The prompt template and model to be used.

    Raises
    ------
    NotImplementedError
        If there is no such implementation yet.
    """
    if llm_model == LLMModel.OPENAI_GPT3_5:
        return (
            ChatPromptTemplate.from_template(OpenAIPrompt.prompt),
            ChatOpenAI(temperature=0.1),
        )

    raise NotImplementedError(f"{llm_model} not in {LLMModel.__members__}")
