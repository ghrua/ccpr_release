import fire
import json
import os
import logging
import random
from openai import OpenAI
client = OpenAI(organization=os.getenv("OPENAI_ORGANIZATION"))
random.seed(10086)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ],
)

class OpenAIModel:
    GPT3_5 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"

def test():
    model = "gpt-3.5-turbo"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
    res = call_chat(model, messages)
    msg_str = json.dumps(messages)
    logging.info("model: {}".format(model))
    logging.info("messages: \n{}\n".format(msg_str))
    logging.info("return: \n{}".format(json.dumps(res)))


def call_chat(model, messages, temperature=1.0, top_p=1.0, n=1, extract_response=False):
    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=temperature,
                                              top_p=top_p,
                                              n=n)
    if extract_response:
        response = response['choices'][0]['message']['content']

    return response
    

def main():
    fire.Fire({
        "test": test,
    })

if __name__ == "__main__":
    main()
