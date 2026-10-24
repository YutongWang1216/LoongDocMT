import importlib
import re
import os
import requests
import time
from typing import Literal
import re
import asyncio
import aiohttp
import logging

logger = logging.getLogger(__name__)


PrepareTools = Literal['view_summaries', 'view_pages', 'look_up_entities']

LLM_MODEL = os.getenv('LLM_MODEL', 'qwen')

def llm_invoke_generate(client, prompt: str, temperature: float=0.7, top_p: float=1.0):
    completion = client.completions.create(
        model=LLM_MODEL,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=1024,
        extra_body={"chat_template_kwargs":{"enable_thinking": False}}
    )
    return completion.choices[0].text


def llm_invoke(client, messages: str | list[dict], tools: list=None, temperature: float=0.0, top_p: float=1.0,
               timeout: float=200.0, max_retries: int=5, retry_delay: float=2.0, call_by_entity: bool = False):

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    for attempt in range(max_retries):
        try:
            if tools:
                if call_by_entity:
                    completion = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                        max_tokens=2048,
                        frequency_penalty=0.2,
                        extra_body={"chat_template_kwargs":{"enable_thinking": False}}
                    )
                else:
                    completion = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=2048,
                        timeout=timeout,
                        extra_body={"chat_template_kwargs":{"enable_thinking": False}}
                    )
            else:
                if call_by_entity:
                    completion = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                        frequency_penalty=0.2,
                        extra_body={"chat_template_kwargs":{"enable_thinking": False}}
                    )
                else:
                    completion = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=2048,
                        timeout=timeout,
                        extra_body={"chat_template_kwargs":{"enable_thinking": False}}
                    )
            return completion.choices[0].message.model_dump()
        except Exception as e:
            err_name = type(e).__name__
            is_timeout = 'Timeout' in err_name or 'timeout' in str(e).lower()
            if is_timeout and attempt < max_retries - 1:
                wait = retry_delay * (attempt + 1)
                print(f"[Warning] llm_invoke timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait}s... [{e}]")
                time.sleep(wait)
            else:
                raise


def get_function_by_name(function_name: str, module_name: str):
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        if callable(func):
            return func
        else:
            raise ValueError(f"{function_name} is not a function.")
    except ModuleNotFoundError:
        raise ValueError(f"Module {module_name} not found.")
    except AttributeError:
        raise ValueError(f"Function {function_name} not found in module {module_name}.")


def get_comet_score(instances: list[dict], timeout=200, max_retries=10, comet_api: str=None, system_level=False):
    if comet_api is not None:
        url = f"http://{comet_api}/evaluate"
    else:
        url = f"http://{os.getenv('COMET_API')}/evaluate"
    payload = {'instances': instances, 'gpus': 1}

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, json=payload, timeout=timeout)

            if response.status_code == 200:
                sentence_level_scores = response.json()['scores']
                if system_level:
                    return sum(sentence_level_scores) / len(sentence_level_scores) if len(sentence_level_scores) > 0 else 0.0
                else:
                    return sentence_level_scores
            else:
                print(f"Request failed with status code: {response.status_code}")
        except requests.Timeout:
            retries += 1
            print(f"Request timed out. Retrying... ({retries}/{max_retries})")
            time.sleep(5)
        except requests.RequestException as e:
            raise RuntimeError(f"Request failed due to: {e}")
            
    raise RuntimeError("Max retries exceeded. Request failed.")


async def async_get_comet_score(instances: list[dict], timeout=200, max_retries=10, comet_api: str=None, system_level=False):
    if comet_api is not None:
        url = f"http://{comet_api}/evaluate"
    else:
        url = f"http://{os.getenv('COMET_API')}/evaluate"
    payload = {'instances': instances, 'gpus': 1}

    retries = 0
    begin_time = time.time()
    while retries < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status == 200:
                        data = await response.json()
                        sentence_level_scores = data['scores']
                        if system_level:
                            end_time = time.time()
                            print(f"[{comet_api}]: {end_time - begin_time:.2f} seconds, {len(instances)} sentences.")
                            return sum(sentence_level_scores) / len(sentence_level_scores) if len(sentence_level_scores) > 0 else 0.0
                        else:
                            end_time = time.time()
                            print(f"[{comet_api}]: {end_time - begin_time:.2f} seconds, {len(instances)} sentences.")
                            return sentence_level_scores
                    else:
                        print(f"Request failed with status code: {response.status}")
        except asyncio.TimeoutError:
            retries += 1
            print(f"Request timed out. Retrying... ({retries}/{max_retries})")
            await asyncio.sleep(5)
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Request failed due to: {e}")

    raise RuntimeError("Max retries exceeded. Request failed.")


def build_observation_prompt(target_lines: str, candidate_info: str|list, tool_name: PrepareTools):
    if os.getenv('WITH_INTER_TRANS', 'false').lower() == 'true':
        target_text = '\n'.join(target_lines)
        prompt = f'<Current Translation>\n{target_text}\n\n'
    else:
        prompt = ''
    if tool_name == 'view_summaries':
        prompt += f'Now, here are several summaries of previous pages, each labeled with a unique number in square brackets:\n<Summaries>\n'
        for idx, summary in enumerate(candidate_info):
            prompt += f'[{idx + 1}] {summary}\n'
        if len(candidate_info) == 0:
            prompt += 'No summaries available.\n'
        prompt += '\n'
    elif tool_name == 'view_pages':
        prompt += f'Now, here are several related source-and-target sentence pairs from previous pages, each labeled with a unique number in square brackets:\n<Sentence Paires>\n'
        for idx, (source, target) in enumerate(candidate_info):
            prompt += f'[{idx + 1}] [Source] {source} // [Target] {target}\n'
        if len(candidate_info) == 0:
            prompt += 'No sentence pairs available.\n'
        prompt += '\n'
    elif tool_name == 'look_up_entities':
        for info in candidate_info:
            assert len(info) == 3, "Each entity record should contain (name, name_translation, tldr)."
        prompt += f'Now, here are several records of entities mentioned in the source text, each labeled with a unique number in square brackets:\n<Entity Records>\n'
        for idx, (name, name_translation, tldr) in enumerate(candidate_info):
            prompt += f'[{idx + 1}] {name} / {name_translation}: {tldr}\n'
        if len(candidate_info) == 0:
            prompt += 'No entity records available.\n'
        prompt += '\n'
    
    info_map = {
        'view_summaries': 'summary',
        'view_pages': 'sentence pair',
        'look_up_entities': 'record'
    }
    
    info_map_cap = {
        'view_summaries': 'Summary',
        'view_pages': 'Sentence Pair',
        'look_up_entities': 'Record'
    }
    
    prompt += f"Briefly analyse whether each {info_map[tool_name]} should be selected to improve the translation and output your final selection. Follow the output format strictly, and DO NOT inlude any additional content in the output:\n<Output Format>\n[Analysis]\n{info_map_cap[tool_name]} 1: This {info_map[tool_name]} ... So it should/shouldn't be selected.\n{info_map_cap[tool_name]} 2: This {info_map[tool_name]} ... So it should/shouldn't be selected.\n...\n\n[Selection]\n1, 3, ... (or \"N/A\" if none of them are selected)"
    
    return prompt


def get_new_info(candidate_info: list, action: dict, tool_name: str):
    llm_response = action['content']
    
    selection_match = re.search(r'\[Selection\]\s*([\s\S]*?)(?:\n\[|$)', llm_response)
    if selection_match is None:
        print(f"Warning: No selection found in the response for tool {tool_name}.")
        print(llm_response)
        return {tool_name: []}
        # raise ValueError(f"No selection found in the response for tool {tool_name}.\nResponse:\n{llm_response}")
    selection = selection_match.group(1).strip()
    print(f'Selection: {selection}')
    
    if selection == "N/A":
        return {tool_name: []}

    chosen_ids = re.findall(r'\d+', selection)
    
    new_info = []
    for j, idx in enumerate(chosen_ids):
        real_idx = int(idx) - 1
        if real_idx < len(candidate_info):
            new_info.append(candidate_info[real_idx])
    return {tool_name: new_info}


def force_generate(messages, selection, tool_name: PrepareTools, tokenizer, client, temperature, candidate_num: int):
    selection = [str(s) if isinstance(s, int) else s for s in selection]

    observation = messages[-1]['content']
    lines = observation.splitlines()
    instruct_begin_line_id = None
    for i, line in enumerate(lines):
        if line.startswith("Briefly analyse whether each"):
            if instruct_begin_line_id is not None:
                raise ValueError("Multiple instruction lines found in the observation.")
            instruct_begin_line_id = i
    if instruct_begin_line_id is None:
        raise ValueError("No instruction line found in the observation.")
    
    info_map = {
        'view_summaries': 'summaries',
        'view_pages': 'sentence pairs',
        'look_up_entities': 'records'
    }

    info_map_single = {
        'view_summaries': 'summary',
        'view_pages': 'sentence pair',
        'look_up_entities': 'record'
    }
    
    info_map_cap = {
        'view_summaries': 'Summary',
        'view_pages': 'Sentence Pair',
        'look_up_entities': 'Record'
    }
    
    new_instruct = f'''Select the numbers of the {info_map[tool_name]} that you think would be helpful in improving the translation and briefly analyse the reasons in English. You MUST provide an analysis for each and every {info_map_single[tool_name]} item, whether selected or not. Follow the output format of the given example, and DO NOT inlude any additional content in the output:
<Output Example>
[Selection]
2, 4, ... (\"N/A\" means no items are selected)

[Rejection]
1, 3, ... (\"N/A\" means no items are rejected)

[Analysis]
{info_map_cap[tool_name]} 1 (not selected): This {info_map_single[tool_name]} ... Therefore, it should not be selected.
{info_map_cap[tool_name]} 2 (selected): This {info_map_single[tool_name]} ... Therefore, it should be selected.
{info_map_cap[tool_name]} 3 (not selected): This {info_map_single[tool_name]} ... Therefore, it should not be selected.
{info_map_cap[tool_name]} 4 (selected): This {info_map_single[tool_name]} ... Therefore, it should be selected.
...
'''
    candidate_ids = [str(i + 1) for i in range(candidate_num)]
    rejection = [cid for cid in candidate_ids if cid not in selection]
    observation = '\n'.join(lines[:instruct_begin_line_id]) + '\n' + new_instruct
    messages[-1]['content'] = observation

    selection_prompt = f"[Selection]\n{', '.join(selection) if selection else 'N/A'}"
    rejection_prompt = f"[Rejection]\n{', '.join(rejection) if rejection else 'N/A'}"
    messages.append({
        "role": "assistant",
        "content": f"<think>\n\n</think>\n\n{selection_prompt}\n\n{rejection_prompt}\n\n[Analysis]\n"
    })
    
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )

    if prompt.endswith("<|im_end|>\n"):
        prompt = prompt[:-len("<|im_end|>\n")]
    else:
        raise ValueError("Prompt does not end with <|im_end|>.\n" + prompt)

    completion = llm_invoke_generate(
        client=client,
        prompt=prompt,
        temperature=temperature,
    )
    messages[-1]['content'] += completion.strip()
    
    def swap_selection_analyse(text):
        pattern = re.compile(r'(\[Selection\][^\[]*?)\n(\[Rejection\][^\[]*?)\n(\[Analysis\][\s\S]*)', re.MULTILINE)
        def repl(m):
            return f"{m.group(3).strip()}\n\n{m.group(1).strip()}\n\n{m.group(2).strip()}"
        
        return pattern.sub(repl, text)

    final_message = swap_selection_analyse(messages[-1]['content'])
    
    return final_message



async def async_llm_invoke_generate(async_client, prompt: str, temperature: float = 0.7, top_p: float = 1.0,
                                    timeout: float = 200.0, max_retries: int = 5, retry_delay: float = 2.0):
    for attempt in range(max_retries):
        try:
            completion = await async_client.completions.create(
                model=LLM_MODEL,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=1024,
                timeout=timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            return completion.choices[0].text
        except Exception as e:
            err_name = type(e).__name__
            is_timeout = 'Timeout' in err_name or 'timeout' in str(e).lower()
            if is_timeout and attempt < max_retries - 1:
                wait = retry_delay * (attempt + 1)
                print(f"[Warning] async_llm_invoke_generate timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait}s... [{e}]")
                await asyncio.sleep(wait)
            else:
                raise


async def async_llm_invoke(async_client, messages, tools: list = None, temperature: float = 0.0, top_p: float = 1.0,
                           timeout: float = 200.0, max_retries: int = 5, retry_delay: float = 2.0, call_by_entity: bool = False):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    for attempt in range(max_retries):
        try:
            if tools:
                if call_by_entity:
                    completion = await async_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                        frequency_penalty=0.2,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    )
                else:
                    completion = await async_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    )
            else:
                if call_by_entity:
                    completion = await async_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                        max_tokens=2048,
                        frequency_penalty=0.2,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    )
                else:
                    completion = await async_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                        max_tokens=2048,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    )
            return completion.choices[0].message.model_dump()
        except Exception as e:
            err_name = type(e).__name__
            is_timeout = 'Timeout' in err_name or 'timeout' in str(e).lower()
            if is_timeout and attempt < max_retries - 1:
                wait = retry_delay * (attempt + 1)
                print(f"[Warning] async_llm_invoke timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait}s... [{e}]")
                await asyncio.sleep(wait)
            else:
                raise


async def async_force_generate(messages, selection, tool_name: PrepareTools, tokenizer, async_client, temperature, candidate_num: int):
    """Async version of force_generate – same logic, LLM call is awaited."""
    selection = [str(s) if isinstance(s, int) else s for s in selection]

    observation = messages[-1]['content']
    lines = observation.splitlines()
    instruct_begin_line_id = None
    for i, line in enumerate(lines):
        if line.startswith("Briefly analyse whether each"):
            if instruct_begin_line_id is not None:
                raise ValueError("Multiple instruction lines found in the observation.")
            instruct_begin_line_id = i
    if instruct_begin_line_id is None:
        raise ValueError("No instruction line found in the observation.")

    info_map = {
        'view_summaries': 'summaries',
        'view_pages': 'sentence pairs',
        'look_up_entities': 'records'
    }
    info_map_single = {
        'view_summaries': 'summary',
        'view_pages': 'sentence pair',
        'look_up_entities': 'record'
    }
    info_map_cap = {
        'view_summaries': 'Summary',
        'view_pages': 'Sentence Pair',
        'look_up_entities': 'Record'
    }

    new_instruct = f'''Select the numbers of the {info_map[tool_name]} that you think would be helpful in improving the translation and briefly analyse the reasons in English. You MUST provide an analysis for each and every {info_map_single[tool_name]} item, whether selected or not. Follow the output format of the given example, and DO NOT inlude any additional content in the output:
<Output Example>
[Selection]
2, 4, ... (\"N/A\" means no items are selected)

[Rejection]
1, 3, ... (\"N/A\" means no items are rejected)

[Analysis]
{info_map_cap[tool_name]} 1 (not selected): This {info_map_single[tool_name]} ... Therefore, it should not be selected.
{info_map_cap[tool_name]} 2 (selected): This {info_map_single[tool_name]} ... Therefore, it should be selected.
{info_map_cap[tool_name]} 3 (not selected): This {info_map_single[tool_name]} ... Therefore, it should not be selected.
{info_map_cap[tool_name]} 4 (selected): This {info_map_single[tool_name]} ... Therefore, it should be selected.
...
'''

    candidate_ids = [str(i + 1) for i in range(candidate_num)]
    rejection = [cid for cid in candidate_ids if cid not in selection]
    observation = '\n'.join(lines[:instruct_begin_line_id]) + '\n' + new_instruct
    messages[-1]['content'] = observation

    selection_prompt = f"[Selection]\n{', '.join(selection) if selection else 'N/A'}"
    rejection_prompt = f"[Rejection]\n{', '.join(rejection) if rejection else 'N/A'}"
    messages.append({
        "role": "assistant",
        "content": f"<think>\n\n</think>\n\n{selection_prompt}\n\n{rejection_prompt}\n\n[Analysis]\n"
    })

    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )

    if prompt.endswith("<|im_end|>\n"):
        prompt = prompt[:-len("<|im_end|>\n")]
    else:
        raise ValueError("Prompt does not end with <|im_end|>.\n" + prompt)

    completion = await async_llm_invoke_generate(
        async_client=async_client,
        prompt=prompt,
        temperature=temperature,
    )
    messages[-1]['content'] += completion.strip()

    def swap_selection_analyse(text):
        pattern = re.compile(r'(\[Selection\][^\[]*?)\n(\[Rejection\][^\[]*?)\n(\[Analysis\][\s\S]*)', re.MULTILINE)
        def repl(m):
            return f"{m.group(3).strip()}\n\n{m.group(1).strip()}\n\n{m.group(2).strip()}"
        return pattern.sub(repl, text)

    return swap_selection_analyse(messages[-1]['content'])
