from threading import Lock, Thread
from gpt_engineer.hf_chatbot_base import HuggingFaceChatBotBase

def text_streamer(prompt, bot : HuggingFaceChatBotBase):
    # Start the generation in a separate thread
    thread = Thread(target=run_generation, args=(prompt, bot))
    thread.start()

    # Use a while loop to continuously yield the generated text
    while True:
        try:
            # This is a blocking call until there's a new chunk of text or a stop signal
            new_text = next(bot.streamer)
            yield new_text
        except StopIteration:
            # If we receive a StopIteration, it means the stream has ended
            break
        # await asyncio.sleep(0.5)

def run_generation(prompt, bot : HuggingFaceChatBotBase):
    # llama.pipe(prompt)[0]['generated_text']
    bot.user_input(prompt)
    bot.bot_response()