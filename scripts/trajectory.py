import random
import asyncio
from transformers import AutoTokenizer
from utils import (
    llm_invoke,
    get_comet_score,
    async_get_comet_score,
    get_function_by_name,
    build_observation_prompt,
    get_new_info,
    force_generate,
    async_llm_invoke,
    async_force_generate,
)

from tools import translate, translate_cot, async_translate

from information_pool import InformationPool

from copy import deepcopy
import graphviz
from tqdm import tqdm
import os
import json
import re
from filelock import FileLock
from typing import Literal
import itertools
import os

from logger import get_logger
from datetime import datetime


logger = get_logger(__name__)

verbose_memory_map = {
    "view_pages": False,
    "view_summaries": False,
    "look_up_entities": False,
    "translate": True,
    "update_entities": False,
    "write_summary": False
}

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

PrepareTools = Literal['view_summaries', 'view_pages', 'look_up_entities', 'END']

class Node:
    def __init__(self, tool_name: PrepareTools, existed_info: dict=None, parent=None, children: list=None, source_lines: list[str]=None, target_lines: list[str]=None, reference_lines: list[str]=None, score=None, history=None, selection=None, base_trans_samples=None, cot_trans_samples=None):
        self.tool_name: PrepareTools = tool_name
        self.existed_info = existed_info if existed_info is not None else {'view_summaries': [], 'view_pages': [], 'look_up_entities': []}
        self.parent = parent
        self.children = children if children is not None else []
        self.source_lines = source_lines
        self.target_lines = target_lines
        self.reference_lines = reference_lines
        self.score = score
        self.history = history if history is not None else []
        self.selection = selection
        self.base_trans_samples = base_trans_samples if base_trans_samples is not None else []
        self.cot_trans_samples = cot_trans_samples if cot_trans_samples is not None else []

    def to_dict(self):
        return {
            "tool_name": self.tool_name,
            "existed_info": self.existed_info,
            "parent": None if self.parent is None else id(self.parent),
            "children": [child.to_dict() for child in self.children],
            "source_lines": self.source_lines,
            "target_lines": self.target_lines,
            "reference_lines": self.reference_lines,
            "score": self.score,
            "history": self.history,
            "selection": self.selection,
            "base_trans_samples": self.base_trans_samples,
            "cot_trans_samples": self.cot_trans_samples
        }
    
    @classmethod
    def from_dict(cls, data, parent=None):
        if data == None:
            return None
        node = cls(
            tool_name=data["tool_name"],
            existed_info=data["existed_info"],
            parent=parent,
            source_lines=data["source_lines"],
            target_lines=data["target_lines"],
            reference_lines=data["reference_lines"],
            score=data["score"],
            history=data["history"],
            selection=data.get("selection", None),
            base_trans_samples=data["base_trans_samples"],
            cot_trans_samples=data["cot_trans_samples"]
        )
        for child_data in data["children"]:
            child_node = cls.from_dict(data=child_data, parent=node)
            node.children.append(child_node)
        return node

    def select_descendant(self):
        node: Node = max(self.children, key=lambda child: (child.score, len(str(child.history).split())))
        return node
    
class Trajectory:
    def __init__(self, info_pool: InformationPool, root_list=None, schedule_client=None, inference_client=None, async_schedule_client=None, async_inference_client=None, save_path="results", comet_api=None, hyp_file: str=None, stage: str='train', sample_strategy: Literal['generate', 'random']='random', tokenizer_path=None, schedule_temperature=0.7, infer_temperature=0.7, base_target_lines=None, translate_style='base'):
        self.info_pool = deepcopy(info_pool)
        self.root_list: list[Node] = root_list if root_list is not None else []
        self.schedule_client = schedule_client
        self.inference_client = inference_client
        self.async_schedule_client = async_schedule_client
        self.async_inference_client = async_inference_client
        self.save_path = save_path
        self.comet_api = comet_api
        self.hyp_file = hyp_file
        self.stage = stage
        self.sample_strategy = sample_strategy
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if tokenizer_path else None
        self.schedule_temperature = schedule_temperature
        self.infer_temperature = infer_temperature
        self.base_target_lines = [] if base_target_lines is None else base_target_lines
        self.translate_style = translate_style  # base or cot
        random.seed(42)

    def to_dict(self):
        return {
            "info_pool": self.info_pool.to_dict(),
            "root_list": [root.to_dict() if root else None for root in self.root_list],
            "save_path": self.save_path,
            "stage": self.stage,
            "sample_strategy": self.sample_strategy,
            "schedule_temperature": self.schedule_temperature,
            "infer_temperature": self.infer_temperature,
            "base_target_lines": self.base_target_lines,
            "translate_style": self.translate_style
        }
    
    @classmethod
    def from_dict(cls, data, schedule_client=None, inference_client=None, comet_api=None, encoder=None, tokenizer_path=None):
        root_list = [Node.from_dict(data=node_data) for node_data in data["root_list"]]
        trajectory = cls(
            info_pool=InformationPool.from_dict(data=data["info_pool"], client=inference_client, encoder=encoder),
            root_list=root_list,
            schedule_client=schedule_client,
            inference_client=inference_client,
            save_path=data["save_path"],
            comet_api=comet_api,
            stage=data["stage"],
            sample_strategy=data["sample_strategy"],
            tokenizer_path=tokenizer_path,
            schedule_temperature=data["schedule_temperature"],
            infer_temperature=data["infer_temperature"],
            base_target_lines=data["base_target_lines"],
            translate_style=data["translate_style"]
        )
        return trajectory

    def get_possible_actions(self, node: Node, candidate_info, sample_times):
        tool_name = node.tool_name
        candidate_num = len(candidate_info)
        
        if self.stage == 'train':
            if self.sample_strategy == 'random':
                candidate_ids = [i + 1 for i in range(candidate_num)]
                possible_selections = [list(c) for r in range(1, len(candidate_ids)+1) for c in itertools.combinations(candidate_ids, r)]
                possible_selections = [[]] + random.sample(possible_selections, min(len(possible_selections), sample_times))
                if candidate_ids not in possible_selections:
                    possible_selections.append(candidate_ids)
                
                actions = []
                for i in range(len(possible_selections)):
                    print(f'Sample {i + 1}: {possible_selections[i]} / {candidate_num}')
                    force_action = force_generate(messages=deepcopy(node.history), selection=possible_selections[i], tool_name=tool_name, tokenizer=self.tokenizer, client=self.schedule_client, temperature=self.schedule_temperature, candidate_num=candidate_num)
                    actions.append(force_action)
                assert len(actions) == len(possible_selections)
                return actions, possible_selections
            elif self.sample_strategy == 'generate':
                actions = []
                sampled_selections = []
                for i in range(sample_times):
                    response = llm_invoke(self.schedule_client, messages=node.history, temperature=self.schedule_temperature)['content']
                    
                    selection_match = re.search(r'\[Selection\]\s*([\s\S]*?)(?:\n\[|$)', response)
                    
                    if selection_match is None:
                        continue
                    
                    selection = selection_match.group(1).strip()
                    if selection == "N/A":
                        chosen_ids = []
                    else:
                        chosen_ids = re.findall(r'\d+', selection)
                        chosen_ids = [int(i) if i.isdigit() else i for i in chosen_ids]
                        
                    print(f'Sample {i + 1}: {chosen_ids} / {candidate_num}')
                        
                    if chosen_ids not in sampled_selections:
                        sampled_selections.append(chosen_ids)
                        actions.append(response)
                
                if [] not in sampled_selections:
                    force_action = force_generate(messages=deepcopy(node.history), selection=[], tool_name=tool_name, tokenizer=self.tokenizer, client=self.schedule_client, temperature=self.schedule_temperature, candidate_num=candidate_num)
                    actions.append(force_action)
                    sampled_selections.append([])
                
                return actions, sampled_selections
        else:
            action = llm_invoke(self.schedule_client, messages=node.history, temperature=self.schedule_temperature)['content']

            selection_match = re.search(r'\[Selection\]\s*([\s\S]*?)(?:\n\[|$)', action)
                    
            if selection_match is None:
                print(f"Selection match not found in action: {action}")
                return [action], [[]]
            
            selection = selection_match.group(1).strip()
            if selection == "N/A":
                chosen_ids = []
                print(f"{node.tool_name}: N/A / {candidate_num}")
            else:
                chosen_ids = re.findall(r'\d+', selection)
                chosen_ids = [int(i) if i.isdigit() else i for i in chosen_ids]
                print(f"{node.tool_name}: {chosen_ids} / {candidate_num}")
            
            return [action], [chosen_ids]

    def expand(self, node: Node, sample_times=6, mid_trans_sample_times=5, final_trans_sample_times=5):

        tool_function = get_function_by_name(node.tool_name, 'tools')
        logger.info(f"Begin getting candidate info for tool {node.tool_name}.")
        print(f"##### {node.tool_name} #####")
        candidate_info = tool_function(self.info_pool, node)
        logger.info(f"Finished getting candidate info for tool {node.tool_name}.")

        new_nodes: list[Node] = []
        prepare_tools_list = list(PrepareTools.__args__)
        next_tool_name = prepare_tools_list[prepare_tools_list.index(node.tool_name) + 1]

        observation = build_observation_prompt(target_lines=node.target_lines, candidate_info=candidate_info, tool_name=node.tool_name)
        if len(node.history) == 1:
            source_content = '\n'.join(node.source_lines)
            node.history.append({'role': 'user', 'content': f'<Source Content>\n{source_content}\n\n{observation}'})
        else:
            node.history.append({'role': 'user', 'content': observation})  # 当前翻译 + tool信息
        possible_actions, possible_selections = self.get_possible_actions(node=node, candidate_info=candidate_info, sample_times=sample_times)
        logger.info(f"Finished getting possible actions for tool {node.tool_name}.")
        
        for action, selection in zip(possible_actions, possible_selections):
            new_node = Node(
                tool_name=next_tool_name,
                existed_info=deepcopy(node.existed_info),
                parent=node,
                source_lines=deepcopy(node.source_lines),
                target_lines=deepcopy(node.target_lines),
                reference_lines=deepcopy(node.reference_lines),
                history=deepcopy(node.history),
                selection=selection
            )
            action = {'role': 'assistant', 'content': action}
            new_node.history.append(action)
            
            new_info = get_new_info(candidate_info=candidate_info, action=action, tool_name=node.tool_name)  # 选则有效信息的内容
            new_node.existed_info.update(new_info)

            if self.stage == 'train':
                for _ in range(mid_trans_sample_times):
                    logger.info(f"Begin sampling base translation for tool {node.tool_name} with selection {selection}.")
                    sample_target_lines, sample_messages = translate(
                        client=self.inference_client,
                        source_lines=new_node.source_lines,
                        info_dict=new_node.existed_info,
                        src_language=self.info_pool.src_language,
                        tgt_language=self.info_pool.tgt_language,
                        temperature=self.infer_temperature,
                        ensure_alignment=False
                    )
                    if len(sample_target_lines) == len(node.source_lines):
                        sample_score = get_comet_score([{'src': i, 'mt': j, 'ref': k} for i, j, k in zip(new_node.source_lines, sample_target_lines, new_node.reference_lines)], comet_api=self.comet_api, system_level=True)
                    else:
                        sample_score = float('-inf')
                    new_node.base_trans_samples.append({'score': sample_score, 'target_lines': sample_target_lines, 'messages': sample_messages})
                    logger.info(f"Finished sampling base translation with score {sample_score}.")

                tmp_scores = [sample['score'] for sample in new_node.base_trans_samples if sample['score'] != float('-inf')]
                new_node.score = sum(tmp_scores) / len(tmp_scores) if tmp_scores != [] else float('-inf')
                if tmp_scores:
                    new_node.target_lines = max(new_node.base_trans_samples, key=lambda x: x['score'])['target_lines']
                else:
                    print(f"All samples misaligned for selection {selection}; inheriting target lines from parent node ({len(node.target_lines)}/{len(new_node.source_lines)}).")
                    new_node.target_lines = deepcopy(node.target_lines)

                write_data = {
                    'source_lines': new_node.source_lines,
                    'reference_lines': new_node.reference_lines,
                    'base_target_lines': self.base_target_lines[-1],
                    'trans_samples': new_node.base_trans_samples
                }
                write_file = os.path.join(self.save_path, f"base_trans_{self.info_pool.current_page_id}.json")
                written_data = []
                if os.path.exists(write_file):
                    with open(write_file, 'r') as f:
                        written_data = json.load(f)
                written_data.append(write_data)
                with open(write_file, 'w') as f:
                    json.dump(written_data, f, indent=4, ensure_ascii=False)
                    
                logger.info(f"Created new node with tool {new_node.tool_name} and score {new_node.score}.")
            else:
                if os.getenv('WITH_INTER_TRANS', 'false').lower() == 'true' or self.is_terminal(new_node):
                    if self.translate_style == 'base':
                        new_target_lines = translate(
                            client=self.inference_client,
                            source_lines=new_node.source_lines,
                            info_dict=new_node.existed_info,
                            src_language=self.info_pool.src_language,
                            tgt_language=self.info_pool.tgt_language,
                            temperature=self.infer_temperature,
                            ensure_alignment=True
                        )
                        new_node.score = None
                        new_node.target_lines = new_target_lines
                    elif self.translate_style == 'cot':
                        new_target_lines = translate_cot(
                            client=self.inference_client,
                            source_lines=new_node.source_lines,
                            info_dict=new_node.existed_info,
                            src_language=self.info_pool.src_language,
                            tgt_language=self.info_pool.tgt_language,
                            temperature=self.infer_temperature,
                            ensure_alignment=True
                        )
                        new_node.score = None
                        new_node.target_lines = new_target_lines
                    else:
                        raise ValueError(f"Unknown translate style: {self.translate_style}")
                else:
                    new_node.score = None
                    new_node.target_lines = [''] * len(new_node.source_lines)
            new_nodes.append(new_node)
            
        return new_nodes
        
    def update_info_pool(self, info_pool: InformationPool, target_lines: list[str], score: float, history: list[dict]=None):
        assert len(info_pool.target_pages) == len(info_pool.comet_scores) == info_pool.current_page_id, f"Target length: {len(info_pool.target_pages)}, Comet scores length: {len(info_pool.comet_scores)}, Current page ID: {info_pool.current_page_id}"
        info_pool.target_pages.append(target_lines)
        info_pool.comet_scores.append(score)

        page_summaries = info_pool.page_summaries
        page_summaries.update_summary(
            src_page_text=sep_map[info_pool.src_lang].join(info_pool.source_pages[info_pool.current_page_id]),
            tgt_page_text=sep_map[info_pool.tgt_lang].join(info_pool.target_pages[info_pool.current_page_id]),
            page_id=info_pool.current_page_id
        )

        entity_records = info_pool.entity_records
        entity_records.update_record(
            src_lines=info_pool.source_pages[info_pool.current_page_id],
            src_page_text=sep_map[info_pool.src_lang].join(info_pool.source_pages[info_pool.current_page_id]),
            tgt_page_text=sep_map[info_pool.tgt_lang].join(info_pool.target_pages[info_pool.current_page_id]),
        )
        
        sentence_pairs = info_pool.sentence_pairs
        sentence_pairs.update_pairs(
            src_sentences=info_pool.source_pages[info_pool.current_page_id],
            tgt_sentences=target_lines
        )

    def is_terminal(self, node: Node):
        return node.tool_name == 'END'

    def build_tree(self, root: Node):
        cur_node: Node = root
        while not self.is_terminal(cur_node):
            cur_node.children = self.expand(node=cur_node)
            cur_node = cur_node.select_descendant()
        return cur_node

    def save_data(self, root_node: Node, leaf_node: Node, save_path: str, page_id: int, hyp_file: str=None):
        end_idx_map = {'view_pages': 3, 'look_up_entities': 5, 'END': 7}
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if self.stage == 'train':
            image_file_name = os.path.join(self.save_path, f"structure_{page_id}")
            dot = self.visualize(root_node)
            dot.render(image_file_name, format='png', cleanup=True)
            
            cur_node = leaf_node
            
            selection_data = []
            multi_chosen_data = []
            multi_rejected_data = []
            
            while cur_node.parent:
                end_idx = end_idx_map[cur_node.tool_name]
                
                if os.path.exists(f'{save_path}/chosen_{page_id}.json'):
                    with open(f'{save_path}/chosen_{page_id}.json', 'r') as f:
                        written_data = json.load(f)
                else:
                    written_data = []
                written_data = [cur_node.history[:end_idx]] + written_data
                with open(f'{save_path}/chosen_{page_id}.json', 'w') as f:
                    json.dump(written_data, f, indent=4, ensure_ascii=False)

                worst_brother_node = min(cur_node.parent.children, key=lambda child: child.score)
                if os.path.exists(f'{save_path}/rejected_{page_id}.json'):
                    with open(f'{save_path}/rejected_{page_id}.json', 'r') as f:
                        written_data = json.load(f)
                else:
                    written_data = []
                written_data = [worst_brother_node.history[:end_idx]] + written_data
                with open(f'{save_path}/rejected_{page_id}.json', 'w') as f:
                    json.dump(written_data, f, indent=4, ensure_ascii=False)
                    
                selection_data.insert(0, [node.selection for node in cur_node.parent.children])
                
                for node in cur_node.parent.children:
                    if node == cur_node:
                        continue
                    if node.score >= cur_node.score:
                        continue
                    multi_chosen_data.append(cur_node.history[:end_idx])
                    multi_rejected_data.append(node.history[:end_idx])
                
                cur_node = cur_node.parent
        
            with open(f'{save_path}/trajectory_{page_id}.json', 'w') as f:
                json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
                
            
            with open(f'{save_path}/sampled_selections_{page_id}.json', 'w') as f:
                json.dump(selection_data, f, indent=4, ensure_ascii=False)
                
            with open(f'{save_path}/multi_chosen_{page_id}.json', 'w') as f:
                json.dump(multi_chosen_data, f, indent=4, ensure_ascii=False)
            with open(f'{save_path}/multi_rejected_{page_id}.json', 'w') as f:
                json.dump(multi_rejected_data, f, indent=4, ensure_ascii=False)
            
        if self.stage == 'test':

            messages_list = []
            for root in self.root_list:
                if root is None:
                    continue
                cur_node = root
                while len(cur_node.children) > 0:
                    assert len(cur_node.children) == 1
                    cur_node = cur_node.select_descendant()
                messages_list.append(cur_node.history)
            with open(f'{save_path}/{hyp_file}_messages.json', 'w') as f:
                json.dump(messages_list, f, indent=4, ensure_ascii=False)
            
            hyp_lines = [line for page_lines in self.info_pool.target_pages for line in page_lines]
            with open(f'{save_path}/{hyp_file}', 'w') as f:
                f.write('\n'.join(hyp_lines) + '\n')

    def run(self):
        start_time = datetime.now()
        print(f"Start Time: {start_time}")

        self.info_pool.current_page_id = 0
        target_lines = translate(
            client=self.inference_client,
            source_lines=self.info_pool.source_pages[0],
            info_dict={},
            src_language=self.info_pool.src_language,
            tgt_language=self.info_pool.tgt_language,
            temperature=self.infer_temperature,
            ensure_alignment=True
        )
        
        score = None
        if self.stage == 'train':
            score = get_comet_score(instances=[{'src': src, 'mt': tgt, 'ref': ref} for src, tgt, ref in zip(self.info_pool.source_pages[0], target_lines, self.info_pool.reference_pages[0])], comet_api=self.comet_api, system_level=True)
        
        logger.info(f"Begin updating information pool with page 1 translation.")
        self.update_info_pool(info_pool=self.info_pool, target_lines=target_lines, score=score)
        logger.info(f"Finished updating information pool with page 1 translation.")
        
        self.root_list.append(None)
        
        with open('./prompts/system_prompt.txt', 'r') as f:
            system_prompt = f.read()
        initial_messages = [
            {'role': 'system', 'content': system_prompt.format_map({'src_lang': self.info_pool.src_language, 'tgt_lang': self.info_pool.tgt_language})}
        ]
        
        for page_id in range(1, len(self.info_pool.source_pages)):
            print(f'##################### Constructing trajectory for page {page_id + 1} / {len(self.info_pool.source_pages)} #####################')
            logger.info(f"Begin constructing trajectory for page {page_id + 1}")
            self.info_pool.current_page_id = page_id

            root: Node = Node(
                tool_name='view_summaries',
                source_lines=self.info_pool.source_pages[page_id],
                reference_lines=self.info_pool.reference_pages[page_id],
                history=deepcopy(initial_messages)
            )

            if self.stage == 'train' or (self.stage == 'test' and os.getenv('WITH_INTER_TRANS', 'false').lower() == 'true'):
                base_target_lines = translate(
                    self.inference_client,
                    root.source_lines,
                    root.existed_info,
                    self.info_pool.src_language,
                    self.info_pool.tgt_language,
                    temperature=self.infer_temperature,
                    ensure_alignment=True
                )
            else:
                base_target_lines = [''] * len(root.source_lines)
            self.base_target_lines.append(base_target_lines)

            base_score = None
            if self.stage == 'train':
                base_score = get_comet_score([{'src': i, 'mt': j, 'ref': k} for i, j, k in zip(root.source_lines, base_target_lines, root.reference_lines)], comet_api=self.comet_api, system_level=True)

            root.target_lines = base_target_lines
            root.score = base_score

            best_leaf_node = self.build_tree(root)
            logger.info(f"End constructing trajectory for page {page_id + 1}")
            self.root_list.append(root)
            
            logger.info(f"Begin updating information pool with page {page_id + 1} translation.")
            self.update_info_pool(info_pool=self.info_pool, target_lines=best_leaf_node.target_lines, score=best_leaf_node.score)
            logger.info(f"Updated information pool with page {page_id + 1} translation.")

            if self.stage == 'train':
                logger.info(f"Begin saving training data for page {page_id + 1}.")
                self.save_data(root_node=root, leaf_node=best_leaf_node, save_path=self.save_path, page_id=page_id)            
                logger.info(f"Finished saving training data for page {page_id + 1}.")
            
            logger.info(f"Finished processing page {page_id + 1}.")

        if self.stage == 'test':
            self.save_data(root_node=root, leaf_node=best_leaf_node, save_path=self.save_path, page_id=page_id, hyp_file=self.hyp_file)

        end_time = datetime.now()
        print(f"End Time: {end_time}")
        print(f"Total runtime: {end_time - start_time}")


    async def async_get_possible_actions(self, node: Node, candidate_info, sample_times):
        """Async version of get_possible_actions – concurrent LLM calls per strategy."""
        tool_name = node.tool_name
        candidate_num = len(candidate_info)

        if self.stage == 'train':
            if self.sample_strategy == 'random':
                candidate_ids = [i + 1 for i in range(candidate_num)]
                possible_selections = [list(c) for r in range(1, len(candidate_ids) + 1) for c in itertools.combinations(candidate_ids, r)]
                possible_selections = [[]] + random.sample(possible_selections, min(len(possible_selections), sample_times))
                if candidate_ids not in possible_selections:
                    possible_selections.append(candidate_ids)

                for i, sel in enumerate(possible_selections):
                    print(f'Sample {i + 1}: {sel} / {candidate_num}')

                tasks = [
                    async_force_generate(
                        messages=deepcopy(node.history),
                        selection=sel,
                        tool_name=tool_name,
                        tokenizer=self.tokenizer,
                        async_client=self.async_schedule_client,
                        temperature=self.schedule_temperature,
                        candidate_num=candidate_num
                    )
                    for sel in possible_selections
                ]
                actions = list(await asyncio.gather(*tasks))
                assert len(actions) == len(possible_selections)
                return actions, possible_selections

            elif self.sample_strategy == 'generate':
                tasks = [
                    async_llm_invoke(self.async_schedule_client, messages=node.history, temperature=self.schedule_temperature)
                    for _ in range(sample_times)
                ]
                responses = await asyncio.gather(*tasks)

                actions = []
                sampled_selections = []
                for i, response_dict in enumerate(responses):
                    response = response_dict['content']
                    selection_match = re.search(r'\[Selection\]\s*([\s\S]*?)(?:\n\[|$)', response)
                    if selection_match is None:
                        continue
                    selection = selection_match.group(1).strip()
                    if selection == "N/A":
                        chosen_ids = []
                    else:
                        chosen_ids = re.findall(r'\d+', selection)
                        chosen_ids = [int(i) if i.isdigit() else i for i in chosen_ids]
                    print(f'Sample {i + 1}: {chosen_ids} / {candidate_num}')
                    if chosen_ids not in sampled_selections:
                        sampled_selections.append(chosen_ids)
                        actions.append(response)

                if [] not in sampled_selections:
                    force_action = await async_force_generate(
                        messages=deepcopy(node.history),
                        selection=[],
                        tool_name=tool_name,
                        tokenizer=self.tokenizer,
                        async_client=self.async_schedule_client,
                        temperature=self.schedule_temperature,
                        candidate_num=candidate_num
                    )
                    actions.append(force_action)
                    sampled_selections.append([])

                return actions, sampled_selections
        else:
            response_dict = await async_llm_invoke(self.async_schedule_client, messages=node.history, temperature=self.schedule_temperature)
            action = response_dict['content']

            selection_match = re.search(r'\[Selection\]\s*([\s\S]*?)(?:\n\[|$)', action)
            if selection_match is None:
                print(f"Selection match not found in action: {action}")
                return [action], [[]]

            selection = selection_match.group(1).strip()
            if selection == "N/A":
                chosen_ids = []
                print(f"{node.tool_name}: N/A / {candidate_num}")
            else:
                chosen_ids = re.findall(r'\d+', selection)
                chosen_ids = [int(i) if i.isdigit() else i for i in chosen_ids]
                print(f"{node.tool_name}: {chosen_ids} / {candidate_num}")

            return [action], [chosen_ids]

    async def async_expand(self, node: Node, sample_times=6, mid_trans_sample_times=5, final_trans_sample_times=5):
        """Async version of expand – force_generate calls and translate sampling run concurrently."""
        tool_function = get_function_by_name(node.tool_name, 'tools')
        print(f"##### {node.tool_name} #####")
        candidate_info = tool_function(self.info_pool, node)

        prepare_tools_list = list(PrepareTools.__args__)
        next_tool_name = prepare_tools_list[prepare_tools_list.index(node.tool_name) + 1]

        observation = build_observation_prompt(target_lines=node.target_lines, candidate_info=candidate_info, tool_name=node.tool_name)
        if len(node.history) == 1:
            source_content = '\n'.join(node.source_lines)
            node.history.append({'role': 'user', 'content': f'<Source Content>\n{source_content}\n\n{observation}'})
        else:
            node.history.append({'role': 'user', 'content': observation})

        possible_actions, possible_selections = await self.async_get_possible_actions(node=node, candidate_info=candidate_info, sample_times=sample_times)
        logger.info(f"Finished getting possible actions for tool {node.tool_name}.")

        async def process_one_selection(action, selection):
            new_node = Node(
                tool_name=next_tool_name,
                existed_info=deepcopy(node.existed_info),
                parent=node,
                source_lines=deepcopy(node.source_lines),
                target_lines=deepcopy(node.target_lines),
                reference_lines=deepcopy(node.reference_lines),
                history=deepcopy(node.history),
                selection=selection
            )
            action_msg = {'role': 'assistant', 'content': action}
            new_node.history.append(action_msg)
            new_info = get_new_info(candidate_info=candidate_info, action=action_msg, tool_name=node.tool_name)
            new_node.existed_info.update(new_info)

            if self.stage == 'train':
                translate_tasks = [
                    async_translate(
                        async_client=self.async_inference_client,
                        source_lines=new_node.source_lines,
                        info_dict=new_node.existed_info,
                        src_language=self.info_pool.src_language,
                        tgt_language=self.info_pool.tgt_language,
                        temperature=self.infer_temperature,
                        ensure_alignment=False,
                    )
                    for _ in range(mid_trans_sample_times)
                ]
                translate_results = await asyncio.gather(*translate_tasks, return_exceptions=True)

                parsed_samples = []
                for result in translate_results:
                    if isinstance(result, Exception):
                        logger.warning(f"Translation failed: {result}")
                        parsed_samples.append(([], [], None))
                    else:
                        sample_target_lines, sample_messages = result
                        if len(sample_target_lines) == len(node.source_lines):
                            instances = [{'src': i, 'mt': j, 'ref': k} for i, j, k in zip(new_node.source_lines, sample_target_lines, new_node.reference_lines)]
                            parsed_samples.append((sample_target_lines, sample_messages, instances))
                        else:
                            parsed_samples.append((sample_target_lines, sample_messages, None))

                valid_indices = [(idx, data[2]) for idx, data in enumerate(parsed_samples) if data[2] is not None]
                if valid_indices:
                    tasks = [async_get_comet_score(instances, 200, 10, self.comet_api, False) for _, instances in valid_indices]
                    results = await asyncio.gather(*tasks)
                    sample_scores_map = {idx: sum(scores) / len(scores) for (idx, _), scores in zip(valid_indices, results)}
                else:
                    sample_scores_map = {}

                for idx, (sample_target_lines, sample_messages, instances) in enumerate(parsed_samples):
                    sample_score = sample_scores_map.get(idx, float('-inf'))
                    new_node.base_trans_samples.append({'score': sample_score, 'target_lines': sample_target_lines, 'messages': sample_messages})
                    logger.info(f"Finished sampling base translation with score {sample_score}.")
                
                score_debug_file = os.getenv('SCORE_DEBUG_FILE', '')
                if score_debug_file != '':
                    sample_results = []
                    for idx, (sample_target_lines, sample_messages, instances) in enumerate(parsed_samples):
                        sample_score = sample_scores_map.get(idx, float('-inf'))
                        sample_results.append({
                            'score': sample_score,
                            'source_lines': new_node.source_lines,
                            'target_lines': sample_target_lines,
                            'reference_lines': new_node.reference_lines,
                            'messages': sample_messages
                        })
                    with FileLock(score_debug_file + '.lock'):
                        if os.path.exists(score_debug_file):
                            with open(score_debug_file, 'r') as f:
                                existing_data = json.load(f)
                        else:
                            existing_data = []
                        existing_data.append(sample_results)
                        with open(score_debug_file, 'w') as f:
                            json.dump(existing_data, f, indent=4, ensure_ascii=False)

                tmp_scores = [s['score'] for s in new_node.base_trans_samples if s['score'] != float('-inf')]
                print(f"{selection}: {tmp_scores}.")
                new_node.score = sum(tmp_scores) / len(tmp_scores) if tmp_scores else float('-inf')
                    new_node.target_lines = max(new_node.base_trans_samples, key=lambda x: x['score'])['target_lines']
                else:
                    print(f"All samples misaligned for selection {selection}; inheriting target lines from parent node ({len(node.target_lines)}/{len(new_node.source_lines)}).")
                    new_node.target_lines = deepcopy(node.target_lines)
                logger.info(f"Created new node with tool {new_node.tool_name} and score {new_node.score}.")
            else:
                if os.getenv('WITH_INTER_TRANS', 'false').lower() == 'true' or self.is_terminal(new_node):
                    if self.translate_style == 'base':
                        new_target_lines = await async_translate(
                            async_client=self.async_inference_client,
                            source_lines=new_node.source_lines,
                            info_dict=new_node.existed_info,
                            src_language=self.info_pool.src_language,
                            tgt_language=self.info_pool.tgt_language,
                            temperature=self.infer_temperature,
                            ensure_alignment=True,
                        )
                        new_node.score = None
                        new_node.target_lines = new_target_lines
                    else:
                        raise ValueError(f"Async expand only supports translate_style='base'; got '{self.translate_style}'")
                else:
                    new_node.score = None
                    new_node.target_lines = [''] * len(new_node.source_lines)

            return new_node

        new_nodes = list(await asyncio.gather(*[
            process_one_selection(action, sel)
            for action, sel in zip(possible_actions, possible_selections)
        ]))

        if self.stage == 'train':
            write_file = os.path.join(self.save_path, f"base_trans_{self.info_pool.current_page_id}.json")
            written_data = []
            if os.path.exists(write_file):
                with open(write_file, 'r') as f:
                    written_data = json.load(f)
            for new_node in new_nodes:
                written_data.append({
                    'source_lines': new_node.source_lines,
                    'reference_lines': new_node.reference_lines,
                    'base_target_lines': self.base_target_lines[-1],
                    'trans_samples': new_node.base_trans_samples
                })
            with open(write_file, 'w') as f:
                json.dump(written_data, f, indent=4, ensure_ascii=False)

        return new_nodes

    async def async_build_tree(self, root: Node):
        cur_node: Node = root
        while not self.is_terminal(cur_node):
            cur_node.children = await self.async_expand(node=cur_node)
            cur_node = cur_node.select_descendant()
        return cur_node

    async def run_async(self):
        """Async entry point – pages are still sequential, but within each expansion
        all force_generate and translate calls run concurrently via asyncio.gather."""
        start_time = datetime.now()

        self.info_pool.current_page_id = 0
        target_lines = await async_translate(
            async_client=self.async_inference_client,
            source_lines=self.info_pool.source_pages[0],
            info_dict={},
            src_language=self.info_pool.src_language,
            tgt_language=self.info_pool.tgt_language,
            temperature=self.infer_temperature,
            ensure_alignment=True,
        )

        score = None
        if self.stage == 'train':
            score = await async_get_comet_score(
                [{'src': src, 'mt': tgt, 'ref': ref}
                 for src, tgt, ref in zip(self.info_pool.source_pages[0], target_lines, self.info_pool.reference_pages[0])],
                200, 10, self.comet_api, True
            )

        logger.info(f"Begin updating information pool with page 1 translation.")
        await asyncio.to_thread(self.update_info_pool, self.info_pool, target_lines, score)
        logger.info(f"Finished updating information pool with page 1 translation.")

        self.root_list.append(None)

        with open('./prompts/system_prompt.txt', 'r') as f:
            system_prompt = f.read()
        initial_messages = [
            {'role': 'system', 'content': system_prompt.format_map({'src_lang': self.info_pool.src_language, 'tgt_lang': self.info_pool.tgt_language})}
        ]

        for page_id in range(1, len(self.info_pool.source_pages)):
            print(f'##################### Constructing trajectory for page {page_id + 1} / {len(self.info_pool.source_pages)} #####################')
            logger.info(f"Begin constructing trajectory for page {page_id + 1}")
            self.info_pool.current_page_id = page_id

            root: Node = Node(
                tool_name='view_summaries',
                source_lines=self.info_pool.source_pages[page_id],
                reference_lines=self.info_pool.reference_pages[page_id],
                history=deepcopy(initial_messages)
            )

            if self.stage == 'train' or (self.stage == 'test' and os.getenv('WITH_INTER_TRANS', 'false').lower() == 'true'):
                base_target_lines = await async_translate(
                    self.async_inference_client,
                    root.source_lines,
                    root.existed_info,
                    self.info_pool.src_language,
                    self.info_pool.tgt_language,
                    temperature=self.infer_temperature,
                    ensure_alignment=True,
                )
            else:
                base_target_lines = [''] * len(root.source_lines)
            self.base_target_lines.append(base_target_lines)

            base_score = None
            if self.stage == 'train':
                base_score = await async_get_comet_score(
                    [{'src': i, 'mt': j, 'ref': k} for i, j, k in zip(root.source_lines, base_target_lines, root.reference_lines)],
                    200, 10, self.comet_api, True
                )

            root.target_lines = base_target_lines
            root.score = base_score

            best_leaf_node = await self.async_build_tree(root)
            logger.info(f"End constructing trajectory for page {page_id + 1}")
            self.root_list.append(root)

            logger.info(f"Begin updating information pool with page {page_id + 1} translation.")
            await asyncio.to_thread(self.update_info_pool, self.info_pool, best_leaf_node.target_lines, best_leaf_node.score)
            logger.info(f"Updated information pool with page {page_id + 1} translation.")

            if self.stage == 'train':
                logger.info(f"Begin saving training data for page {page_id + 1}.")
                self.save_data(root_node=root, leaf_node=best_leaf_node, save_path=self.save_path, page_id=page_id)
                logger.info(f"Finished saving training data for page {page_id + 1}.")

            logger.info(f"Finished processing page {page_id + 1}.")

        if self.stage == 'test':
            self.save_data(root_node=root, leaf_node=best_leaf_node, save_path=self.save_path, page_id=page_id, hyp_file=self.hyp_file)

        end_time = datetime.now()
        print(f"Total runtime: {end_time - start_time}")

    def visualize(self, root: Node):
        tool_name_short_map = {
            "view_summaries": "summary",
            "view_pages": "page",
            "look_up_entities": "entitity"
        }
        tool_content_map = {
            "view_pages": 2,
            "look_up_entities": 4,
            "END": 6,
        }
        dot = graphviz.Digraph()
        dot.attr(rankdir="LR")

        def add_node_to_graph(node: Node, parent=None):
            node_id = str(id(node))
            cur_tool = tool_name_short_map[node.tool_name] if node.tool_name in tool_name_short_map else node.tool_name
            
            if cur_tool == 'summary':
                action = '' 
            else:
                llm_response = node.history[tool_content_map[node.tool_name]]['content']
                selection_match = re.search(r'\[Selection\]\s*([\s\S]*?)(?:\n\[|$)', llm_response)
                if selection_match is None:
                    action = 'Parsing error'
                else:
                    selection = selection_match.group(1).strip()
                    if selection == "N/A":
                        action = 'N/A'
                    else:
                        action = re.findall(r'\d+', selection)
                        action = [int(i) if i.isdigit() else i for i in action]
            
            if self.stage == 'train':
                label = f"Action={node.selection}\nScore={node.score*100:.3f}\nNext tool={cur_tool}"
            else:
                label = f"Action={node.selection}\nScore=None\nNext tool={cur_tool}"
            dot.node(name=node_id, label=label)
            if parent:
                parent_id = str(id(parent))
                dot.edge(parent_id, node_id)
            for child in node.children:
                add_node_to_graph(child, parent=node)

        add_node_to_graph(root)
        return dot
