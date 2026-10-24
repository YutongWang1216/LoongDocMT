from utils import llm_invoke, async_llm_invoke
from entity_records import EntityRecords
from information_pool import InformationPool
import torch
import re
import os
import asyncio
from copy import deepcopy
import json


sep_map = {
    'en': ' ',
    'zh': '',
    'fr': ' ',
    'de': ' ',
    'cs': ' ',
    'es': ' ',
    'it': ' ',
    'pt': ' ',
    'ru': ' ',
    'ja': '',
    'ar': ' ',
    'English': ' ',
    'Chinese': '',
    'French': ' ',
    'German': ' ',
    'Czech': ' ',
    'Spanish': ' ',
    'Italian': ' ',
    'Portuguese': ' ',
    'Russian': ' ',
    'Japanese': '',
    'Arabic': ' '
}


def view_summaries(info_pool: InformationPool, node, **kargs):
    candidate_summary_number = kargs.get('candidate_summary_number', 4)
    src_summaries = info_pool.page_summaries.src_summaries
    if len(src_summaries) <= candidate_summary_number:
        return src_summaries
    else:
        sentence_encoder = info_pool.encoder
        summary_embeddings = info_pool.page_summaries.summary_embeddings
        query_embedding = sentence_encoder.encode(sep_map[info_pool.src_lang].join(node.source_lines))
        similarities = sentence_encoder.similarity(query_embedding, summary_embeddings)
        top_values, top_indices = torch.topk(similarities, k=min(candidate_summary_number, similarities.shape[1]))
        top_indices = top_indices.tolist()[0]
        candidate_summaries = [src_summaries[i] for i in top_indices]
        return candidate_summaries


def view_pages(info_pool: InformationPool, node, **kargs):
    candidate_sent_number = kargs.get('candidate_page_numbers', 4)

    candidate_pairs = info_pool.sentence_pairs.get_exemplars(node.source_lines, top_k=candidate_sent_number)
    
    return candidate_pairs


def look_up_entities(info_pool: InformationPool, node, **kargs):

    entity_records: EntityRecords = info_pool.entity_records
    source_lines = node.source_lines
    candidate_entities = entity_records.get_records('\n'.join(source_lines))
    return candidate_entities


def translate(client, source_lines, info_dict, src_language, tgt_language, temperature, ensure_alignment, sentence_ids=None) -> list[str]:
    
    if len(source_lines) == 0:
        return []
    
    mid_flag = True
    if sentence_ids is None:
        mid_flag = False
        sentence_ids = list(range(len(source_lines)))
    
    label_dict = {"look_up_entities": "<Entity Records>", "view_summaries": "<Summaries of previous pages>", "view_pages": "<Original texts of previous pages>"}
    
    src_input = ''
    for idx, line in enumerate(source_lines):
        src_input += f'#{idx} <s>{line}</s>\n'

    auxiliary_prompt = ""
    for tool_name in info_dict:
        if len(info_dict[tool_name]) > 0:
            auxiliary_prompt += f"{label_dict[tool_name]}\n"
            for i, info in enumerate(info_dict[tool_name]):
                if tool_name == 'view_summaries':
                    auxiliary_prompt += f'[{i + 1}] {info}\n'
                elif tool_name == 'view_pages':
                    auxiliary_prompt += f'[{i + 1}] [Source] {info[0]} // [Target] {info[1]}\n'
                elif tool_name == 'look_up_entities':
                    auxiliary_prompt += f'[{i + 1}] {info[0]} / {info[1]}: {info[2]}\n'
            auxiliary_prompt += '\n'


    if auxiliary_prompt != "":
        prompt = f"Given some auxiliary information, translate the current page of source text from {src_language} to {tgt_language}.\n\n" + auxiliary_prompt + f"Now please translate the given {src_language} text into {tgt_language}. "
    else:
        prompt = f"Translate the current page of source text from {src_language} to {tgt_language}. "

    prompt += f'Make sure to obay the TRANSLATION TASK RULES.\n\n<TRANSLATION TASK RULES>\n1. Each sentence in the text is marked with "#i" to indicate its order.\n2. The beginning and end of an independent sentences are marked by "<s>" and "</s>", respectively.\n3. Output MUST:\n- Preserve ALL sequence, beginning and end marks ("#i", "<s>" and "</s>")\n- Maintain EXACT 1:1 sentence correspondence\n- NEVER merge/split/reorder/omit sentences.\n\n'
    prompt += f"<{src_language} source text>\n{src_input}"
    

    messages = [
        {"role": "user", "content": prompt}
    ]

    if ensure_alignment:
        max_attempts = 10
        attempt = 0
        while True:
            # if cur_temperature > 1.0:
            if attempt >= max_attempts:
                if len(source_lines) == 1:
                    raise RuntimeError("Alignment failed: \n" + hyp)
                print(f"Splitting input into {sentence_ids[:int(len(source_lines) / 2)]} and {sentence_ids[:int(len(source_lines) / 2)]} lines for retrying...")
                target_lines_part1 = translate(
                    client=client,
                    source_lines=source_lines[:int(len(source_lines) / 2)],
                    info_dict=info_dict,
                    src_language=src_language,
                    tgt_language=tgt_language,
                    temperature=temperature,
                    sentence_ids=sentence_ids[:int(len(source_lines) / 2)],
                    ensure_alignment=ensure_alignment
                )
                
                target_lines_part2 = translate(
                    client=client,
                    source_lines=source_lines[int(len(source_lines) / 2):],
                    info_dict=info_dict,
                    src_language=src_language,
                    tgt_language=tgt_language,
                    temperature=temperature,
                    sentence_ids=sentence_ids[int(len(source_lines) / 2):],
                    ensure_alignment=ensure_alignment
                )
                
                return target_lines_part1 + target_lines_part2

            if mid_flag:
                print(f'Source lines:', sentence_ids)
            response = llm_invoke(client=client, messages=messages, temperature=temperature)
            hyp = response["content"]
            hyp = hyp.strip()
            if hyp[-4:] != '</s>':
                hyp += '</s>'
            
            pattern = r'\d+\s*<s>(.*?)</s>'
            sentences = re.findall(pattern, hyp)

            if len(sentences) != len(source_lines):
                print(hyp)
                attempt += 1
                continue
            else:
                target_lines = [s.strip() for s in sentences]
                return target_lines
    else:
        response = llm_invoke(client, messages=messages, temperature=temperature)
        messages.append({'role': 'assistant', 'content': response['content']})
        hyp = response["content"]
        hyp = hyp.strip()
        if hyp[-4:] != '</s>':
            hyp += '</s>'
        
        pattern = r'\d+\s*<s>(.*?)</s>'
        sentences = re.findall(pattern, hyp)
        
        target_lines = [s.strip() for s in sentences]
        return target_lines, messages


async def async_translate(async_client, source_lines, info_dict, src_language, tgt_language, temperature, ensure_alignment, sentence_ids=None) -> list:
    """Async version of translate – LLM calls are awaited; recursive splits run concurrently."""
    if len(source_lines) == 0:
        return [] if ensure_alignment else ([], [])
    
    mid_flag = True
    if sentence_ids is None:
        mid_flag = False
        sentence_ids = list(range(len(source_lines)))

    label_dict = {
        "look_up_entities": "<Entity Records>",
        "view_summaries": "<Summaries of previous pages>",
        "view_pages": "<Original texts of previous pages>"
    }

    src_input = ''
    for idx, line in enumerate(source_lines):
        src_input += f'#{idx} <s>{line}</s>\n'

    auxiliary_prompt = ""
    for tool_name in info_dict:
        if len(info_dict[tool_name]) > 0:
            auxiliary_prompt += f"{label_dict[tool_name]}\n"
            for i, info in enumerate(info_dict[tool_name]):
                if tool_name == 'view_summaries':
                    auxiliary_prompt += f'[{i + 1}] {info}\n'
                elif tool_name == 'view_pages':
                    auxiliary_prompt += f'[{i + 1}] [Source] {info[0]} // [Target] {info[1]}\n'
                elif tool_name == 'look_up_entities':
                    auxiliary_prompt += f'[{i + 1}] {info[0]} / {info[1]}: {info[2]}\n'
            auxiliary_prompt += '\n'

    if auxiliary_prompt != "":
        prompt = f"Given some auxiliary information, translate the current page of source text from {src_language} to {tgt_language}.\n\n" + auxiliary_prompt + f"Now please translate the given {src_language} text into {tgt_language}. "
    else:
        prompt = f"Translate the current page of source text from {src_language} to {tgt_language}. "

    prompt += f'Make sure to obay the TRANSLATION TASK RULES.\n\n<TRANSLATION TASK RULES>\n1. Each sentence in the text is marked with "#i" to indicate its order.\n2. The beginning and end of an independent sentences are marked by "<s>" and "</s>", respectively.\n3. Output MUST:\n- Preserve ALL sequence, beginning and end marks ("#i", "<s>" and "</s>")\n- Maintain EXACT 1:1 sentence correspondence\n- NEVER merge/split/reorder/omit sentences.\n\n'
    prompt += f"<{src_language} source text>\n{src_input}"

    messages = [{"role": "user", "content": prompt}]

    if ensure_alignment:
        max_attempts = 10
        attempt = 0
        hyp = ""
        while True:
            if attempt >= max_attempts:
                if len(source_lines) == 1:
                    raise RuntimeError("Alignment failed: \n" + hyp)
                half = int(len(source_lines) / 2)
                print(f"Splitting input into {sentence_ids[:half]} and {sentence_ids[half:]} lines for retrying...")
                target_lines_part1, target_lines_part2 = await asyncio.gather(
                    async_translate(
                        async_client=async_client,
                        source_lines=source_lines[:half],
                        info_dict=info_dict,
                        src_language=src_language,
                        tgt_language=tgt_language,
                        temperature=temperature,
                        sentence_ids=sentence_ids[:half],
                        ensure_alignment=ensure_alignment,
                    ),
                    async_translate(
                        async_client=async_client,
                        source_lines=source_lines[half:],
                        info_dict=info_dict,
                        src_language=src_language,
                        tgt_language=tgt_language,
                        temperature=temperature,
                        sentence_ids=sentence_ids[half:],
                        ensure_alignment=ensure_alignment,
                    ),
                )
                return target_lines_part1 + target_lines_part2

            if mid_flag:
                print(f'Source lines:', sentence_ids)
            response = await async_llm_invoke(async_client=async_client, messages=messages, temperature=temperature)
            hyp = response["content"]
            hyp = hyp.strip()
            if hyp[-4:] != '</s>':
                hyp += '</s>'
            
            save_file = os.getenv('TRANS_MES_FILE', '')
            if save_file != '':
                if os.path.exists(save_file):
                    with open(save_file, 'r') as f:
                        existing_content = json.load(f)
                else:
                    existing_content = []
                existing_content.append(messages + [{'role': 'assistant', 'content': hyp}])
                with open(save_file, 'w') as f:
                    json.dump(existing_content, f, indent=4, ensure_ascii=False)

            pattern = r'\d+\s*<s>(.*?)</s>'
            sentences = re.findall(pattern, hyp)

            if len(sentences) != len(source_lines):
                print(hyp)
                attempt += 1
                continue
            else:
                target_lines = [s.strip() for s in sentences]
                return target_lines
    else:
        response = await async_llm_invoke(async_client, messages=messages, temperature=temperature)
        messages.append({'role': 'assistant', 'content': response['content']})
        hyp = response["content"]
        hyp = hyp.strip()
        if hyp[-4:] != '</s>':
            hyp += '</s>'

        pattern = r'\d+\s*<s>(.*?)</s>'
        sentences = re.findall(pattern, hyp)

        target_lines = [s.strip() for s in sentences]
        return target_lines, messages
