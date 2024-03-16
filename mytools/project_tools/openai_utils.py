"""Tools to generate from OpenAI prompts."""

import asyncio
import logging
import os
from typing import Any

import aiolimiter
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
from mytools.tool_utils import FileUtils
import fire

async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(10):
            try:
                return await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 30 seconds."
                )
                await asyncio.sleep(30)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.error.InvalidRequestError:
                logging.warning("OpenAI API Invalid Request: Prompt was filtered")
                return {
                    "choices": [
                        {"message": {"content": "Invalid Request: Prompt was filtered"}}
                    ]
                }
            except openai.error.APIConnectionError:
                logging.warning(
                    "OpenAI API Connection Error: Error Communicating with OpenAI"
                )
                await asyncio.sleep(10)
            except openai.error.Timeout:
                logging.warning("OpenAI APITimeout Error: OpenAI Timeout")
                await asyncio.sleep(10)
            except openai.error.ServiceUnavailableError as e:
                logging.warning(f"OpenAI service unavailable error: {e}")
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    api_key,
    model,
    prompts,
    temperature: float = 1.,
    max_tokens: int = 2048,
    top_p: float = 1.,
    requests_per_minute: int = 15,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    openai.api_key = api_key
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    await openai.aiosession.get().close()  # type: ignore
    ret = []
    for x in responses:
        try:
            content = x["choices"][0]["message"]["content"]
            finish_reason = x["choices"][0]['finish_reason']
        except:
            print (x)
            content = None
            finish_reason = None
        ret.append({'content':content, 'finish_reason':finish_reason})
    return ret


def revise_prompts(it):
    suffix = 'When discussing images that have appeared in the dialogue, the human and the assistant should refer to them by their order, such as "the fist image" and the "the second image", rather than copying the images again.'
    return it + " " + suffix



def mim(prompt_file_path, save_dir, start_id=0, end_id=float("inf"), batch_size=100, top_p=1.0, temperature=1.0):
    data = FileUtils.load_file(prompt_file_path)
    FileUtils.check_dirs(save_dir)
    mini_batch_size = 10
    for i in range(0, len(data), batch_size):
        if i < start_id or i >= end_id:
            continue
        ex_list = data[i : i + batch_size]
        prompts = [revise_prompts(ex['prompt']) for ex in ex_list]
        responses = []
        for j in range(0, len(ex_list), mini_batch_size):
            responses += asyncio.run(
                generate_from_openai_chat_completion(
                    os.getenv("OPENAI_API_KEY"),
                    "gpt-4",
                    prompts[j:j+mini_batch_size],
                    temperature=temperature,
                    top_p=top_p
                )
            )
        ret = []
        for ex, r in zip(ex_list, responses):
            ex['response'] = r['content']
            ex['finish_reason'] = r['finish_reason']
            ret.append(ex)
        FileUtils.save_file(ret, "{}/{}.{}.json".format(save_dir, i, i+batch_size), 'json')

def cai_example():
    import json
    fname = "log-07-10_to_07-20.txt"
    data = []
    for line in open(fname).readlines():
        line = line.strip()
        if line.startswith("log") and not line.endswith("recieve a write request =>"):
            x = eval(line[line.find("{"):])
            data.append(x)
    print (f"total number of requests: {len(data)}")
    data = [x['description'].strip() for x in data if x['interval'] !=0]
    print (f"after remove interval!=0: {len(data)}")
    data = [x for x in data if x]
    print (f"after empty: {len(data)}")
    data = [x for x in data if 5<=len(x)<=100]
    print (f"after remove too long or too short: {len(data)}")
    data = list(set(data))
    data.sort()
    print (f"after deduplicate: {len(data)}")

    #data = data[:5]
    batch_size = 500
    
    for i in range(0, len(data), batch_size):
        prompts = data[i : i + batch_size]
        responses =  asyncio.run(
            generate_from_openai_chat_completion(
                "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",
                "gpt-4",
                prompts
            )
        )
        with open(f"output/query{i}.json", "w") as fo:
            for instruction, response in zip(prompts, responses):
                fo.write(
                    json.dumps(
                        {
                            'instruction': instruction,
                            'content' : response['content'],
                            'finish_reason' : response['finish_reason']
                        }
                    , ensure_ascii=False) + "\n"
                )
            
        

if __name__ == "__main__":
    fire.Fire({
        "cai_example": cai_example,
        "mim": mim
    })
