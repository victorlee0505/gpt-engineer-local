# Modified from https://github.com/opencopilotdev/opencopilot
import asyncio
import time
import json
from threading import Lock, Thread
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from urllib.parse import urljoin

import aiohttp
import requests
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage
from langchain.schema import BaseMessage
from langchain.schema import ChatGeneration
from langchain.schema import ChatResult
from pydantic import Extra
from gpt_engineer.error_messages import LOCAL_LLM_CONNECTION_ERROR, WEAVIATE_INVALID_URL, WEAVIATE_ERROR_EXTRA, COPILOT_IS_NOT_RUNNING_ERROR, INVALID_MODEL_ERROR, INVALID_LOGS_DIR_ERROR
from gpt_engineer.hf_prompts import NO_MEM_PROMPT
from gpt_engineer.hf_chatbot_base import HuggingFaceChatBotBase
from gpt_engineer.hf_streaming_util import run_generation

class ChatHuggingFace(BaseChatModel):

    chatbot: HuggingFaceChatBotBase = None

    def __init__(self, chatbot: HuggingFaceChatBotBase = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chatbot = chatbot

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore
        allow_population_by_field_name = True

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = ""
        for message in messages:
            prompt += message.content
            prompt += "\n"
        self.chatbot.user_input(prompt)
        final = self.chatbot.bot_response()
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=final),
                )
            ]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        final = ""
        try:
            async for text in self._get_async_stream(
                messages[0],
            ):
                final += text
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=final),
                    )
                ]
            )
        except Exception as exc:
            print(LOCAL_LLM_CONNECTION_ERROR)

    def get_token_ids(self, text: str) -> List[int]:
        try:
            return self.chatbot.tokenizer.tokenize(text)
        except Exception as exc:
            print(LOCAL_LLM_CONNECTION_ERROR)

    @property
    def _llm_type(self) -> str:
        return "local-llm"

    def _get_stream(self, prompt):
        thread = Thread(target=run_generation, args=(prompt, self.chatbot))
        thread.start()
        time.sleep(20)
        # Use a while loop to continuously yield the generated text
        while True:
            try:
                # This is a blocking call until there's a new chunk of text or a stop signal
                new_text = next(self.chatbot.streamer)
                yield new_text
            except StopIteration:
                # If we receive a StopIteration, it means the stream has ended
                break
            # await asyncio.sleep(0.5)

    async def _get_async_stream(self, prompt):
        thread = Thread(target=run_generation, args=(prompt, self.chatbot))
        thread.start()
        time.sleep(20)
        # Use a while loop to continuously yield the generated text
        while True:
            try:
                # This is a blocking call until there's a new chunk of text or a stop signal
                new_text = next(self.chatbot.streamer)
                yield new_text
            except StopIteration:
                # If we receive a StopIteration, it means the stream has ended
                break
            await asyncio.sleep(0.5)
