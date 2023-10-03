from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from typing import List, Optional, Union

from langchain.callbacks.openai_info import MODEL_COST_PER_1K_TOKENS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
from gpt_engineer.hf_llm_config import (
    REDPAJAMA_3B,
    REDPAJAMA_7B,
    VICUNA_7B,
    LMSYS_VICUNA_1_5_7B,
    LMSYS_VICUNA_1_5_16K_7B,
    LMSYS_LONGCHAT_1_5_32K_7B,
    LMSYS_VICUNA_1_5_7B_Q8,
    LMSYS_VICUNA_1_5_16K_7B_Q8,
    LMSYS_VICUNA_1_5_13B_Q6,
    LMSYS_VICUNA_1_5_16K_13B_Q6,
    SANTA_CODER_1B,
    STARCHAT_BETA_16B_Q5,
    WIZARDCODER_3B,
    WIZARDCODER_15B_Q8,
    WIZARDCODER_PY_7B,
    WIZARDCODER_PY_7B_Q6,
    WIZARDCODER_PY_13B_Q6,
    WIZARDCODER_PY_34B_Q5,
    WIZARDLM_FALCON_40B_Q6K, 
    LLMConfig,
)
from gpt_engineer.hf_chat_model import ChatHuggingFace
from gpt_engineer.hf_chatbot_base import HuggingFaceChatBotBase

Message = Union[AIMessage, HumanMessage, SystemMessage]

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    step_name: str
    in_step_prompt_tokens: int
    in_step_completion_tokens: int
    in_step_total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int


class HF:
    def __init__(self, llm_config : LLMConfig = None, temperature=0.1):
        """
        Initialize the AI class.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to use, by default "gpt-4".
        temperature : float, optional
            The temperature to use for the model, by default 0.1.
        """
        self.temperature = temperature
        if llm_config is None:
            logger.info("No LLM config found. Exiting.")
            exit()
        self.llm_config = llm_config
        self.llm_config.temperature = temperature
        self.chatbot = HuggingFaceChatBotBase(llm_config=self.llm_config, disable_mem=True, server_mode=False)
        self.tokenizer = self.chatbot.tokenizer
        self.llm : ChatHuggingFace = create_chat_model(self, self.chatbot)

        logger.debug(f"Using model {self.llm_config.model}")

        # initialize token usage log
        self.cumulative_prompt_tokens = 0
        self.cumulative_completion_tokens = 0
        self.cumulative_total_tokens = 0
        self.token_usage_log = []

    def start(self, system: str, user: str, step_name: str) -> List[Message]:
        """
        Start the conversation with a system message and a user message.

        Parameters
        ----------
        system : str
            The content of the system message.
        user : str
            The content of the user message.
        step_name : str
            The name of the step.

        Returns
        -------
        List[Message]
            The list of messages in the conversation.
        """
        messages: List[Message] = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        return self.next(messages, step_name=step_name)

    def fsystem(self, msg: str) -> SystemMessage:
        """
        Create a system message.

        Parameters
        ----------
        msg : str
            The content of the message.

        Returns
        -------
        SystemMessage
            The created system message.
        """
        return SystemMessage(content=msg)

    def fuser(self, msg: str) -> HumanMessage:
        """
        Create a user message.

        Parameters
        ----------
        msg : str
            The content of the message.

        Returns
        -------
        HumanMessage
            The created user message.
        """
        return HumanMessage(content=msg)

    def fassistant(self, msg: str) -> AIMessage:
        """
        Create an AI message.

        Parameters
        ----------
        msg : str
            The content of the message.

        Returns
        -------
        AIMessage
            The created AI message.
        """
        return AIMessage(content=msg)

    def next(
        self,
        messages: List[Message],
        prompt: Optional[str] = None,
        *,
        step_name: str,
    ) -> List[Message]:
        """
        Advances the conversation by sending message history
        to LLM and updating with the response.

        Parameters
        ----------
        messages : List[Message]
            The list of messages in the conversation.
        prompt : Optional[str], optional
            The prompt to use, by default None.
        step_name : str
            The name of the step.

        Returns
        -------
        List[Message]
            The updated list of messages in the conversation.
        """
        """
        Advances the conversation by sending message history
        to LLM and updating with the response.
        """
        if prompt:
            messages.append(self.fuser(prompt))

        logger.debug(f"Creating a new chat completion: {messages}")

        # callsbacks = [StreamingStdOutCallbackHandler()]
        # response = self.llm(messages, callbacks=callsbacks)  # type: ignore
        response = self.llm(messages)

        self.update_token_usage_log(
            messages=messages, answer=response.content, step_name=step_name
        )
        messages.append(response)
        logger.debug(f"Chat completion finished: {messages}")

        return messages

    @staticmethod
    def serialize_messages(messages: List[Message]) -> str:
        """
        Serialize a list of messages to a JSON string.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to serialize.

        Returns
        -------
        str
            The serialized messages as a JSON string.
        """
        return json.dumps(messages_to_dict(messages))

    @staticmethod
    def deserialize_messages(jsondictstr: str) -> List[Message]:
        """
        Deserialize a JSON string to a list of messages.

        Parameters
        ----------
        jsondictstr : str
            The JSON string to deserialize.

        Returns
        -------
        List[Message]
            The deserialized list of messages.
        """
        return list(messages_from_dict(json.loads(jsondictstr)))  # type: ignore

    def update_token_usage_log(
        self, messages: List[Message], answer: str, step_name: str
    ) -> None:
        """
        Update the token usage log with the number of tokens used in the current step.

        Parameters
        ----------
        messages : List[Message]
            The list of messages in the conversation.
        answer : str
            The answer from the AI.
        step_name : str
            The name of the step.
        """
        prompt_tokens = self.num_tokens_from_messages(messages)
        completion_tokens = self.num_tokens(answer)
        total_tokens = prompt_tokens + completion_tokens

        self.cumulative_prompt_tokens += prompt_tokens
        self.cumulative_completion_tokens += completion_tokens
        self.cumulative_total_tokens += total_tokens

        self.token_usage_log.append(
            TokenUsage(
                step_name=step_name,
                in_step_prompt_tokens=prompt_tokens,
                in_step_completion_tokens=completion_tokens,
                in_step_total_tokens=total_tokens,
                total_prompt_tokens=self.cumulative_prompt_tokens,
                total_completion_tokens=self.cumulative_completion_tokens,
                total_tokens=self.cumulative_total_tokens,
            )
        )

    def format_token_usage_log(self) -> str:
        """
        Format the token usage log as a CSV string.

        Returns
        -------
        str
            The token usage log formatted as a CSV string.
        """
        result = "step_name,"
        result += "prompt_tokens_in_step,completion_tokens_in_step,total_tokens_in_step"
        result += ",total_prompt_tokens,total_completion_tokens,total_tokens\n"
        for log in self.token_usage_log:
            result += log.step_name + ","
            result += str(log.in_step_prompt_tokens) + ","
            result += str(log.in_step_completion_tokens) + ","
            result += str(log.in_step_total_tokens) + ","
            result += str(log.total_prompt_tokens) + ","
            result += str(log.total_completion_tokens) + ","
            result += str(log.total_tokens) + "\n"
        return result

    def usage_cost(self) -> float:
        """
        Return the total cost in USD of the api usage.

        Returns
        -------
        float
            Cost in USD.
        """
        prompt_price = MODEL_COST_PER_1K_TOKENS[self.model_name]
        completion_price = MODEL_COST_PER_1K_TOKENS[self.model_name + "-completion"]

        result = 0
        for log in self.token_usage_log:
            result += log.total_prompt_tokens / 1000 * prompt_price
            result += log.total_completion_tokens / 1000 * completion_price
        return result

    def num_tokens(self, txt: str) -> int:
        """
        Get the number of tokens in a text.

        Parameters
        ----------
        txt : str
            The text to count the tokens in.

        Returns
        -------
        int
            The number of tokens in the text.
        """
        return len(self.tokenizer.encode(txt))

    def num_tokens_from_messages(self, messages: List[Message]) -> int:
        """
        Get the total number of tokens used by a list of messages.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to count the tokens in.

        Returns
        -------
        int
            The total number of tokens used by the messages.
        """
        """Returns the number of tokens used by a list of messages."""
        n_tokens = 0
        for message in messages:
            n_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            n_tokens += self.num_tokens(message.content)
        n_tokens += 2  # every reply is primed with <im_start>assistant
        return n_tokens

def create_chat_model(self, chatbot : HuggingFaceChatBotBase) -> BaseChatModel:
    """
    Create a chat model with the specified model name and temperature.

    Parameters
    ----------
    model : str
        The name of the model to create.
    temperature : float
        The temperature to use for the model.

    Returns
    -------
    BaseChatModel
        The created chat model.
    """
    return ChatHuggingFace(
        chatbot=chatbot,
    )

def serialize_messages(messages: List[Message]) -> str:
    return HF.serialize_messages(messages)
