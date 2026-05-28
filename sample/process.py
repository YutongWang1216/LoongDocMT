import argparse
import os
import json
from tqdm import tqdm
import glob
import random
import re
from transformers import AutoTokenizer
from copy import deepcopy


def get_available_output_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    idx = 1
    while True:
        new_path = f"{base_path}_{idx}"
        if not os.path.exists(new_path):
            return new_path
        idx += 1


def generate_reference_label(messages, referce_lines):
    label = ''
    for idx, line in enumerate(referce_lines):
        label += f'#{idx} <s>{line}</s>\n'
    new_messages = deepcopy(messages)
    new_messages[-1]['content'] = label
    return new_messages


def write_data_info(args, file_name):
    
    dataset_info = {}
    if os.path.exists(os.path.join(args.output_path, "dataset_info.json")):
        with open(os.path.join(args.output_path, "dataset_info.json"), "r") as f:    
            dataset_info = json.load(f)
        
    if args.stage == 'sft':
        dataset_info[file_name] = {
            "file_name": f"{file_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                # "system_tag": "system"
            }
        }
        if 'tool' in file_name:
            dataset_info[file_name]["tags"]["system_tag"] = "system"
    elif args.stage == 'dpo':
        dataset_info[file_name] = {
            "file_name": f"{file_name}.json",
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
                # "system": "system"
            }
        }
        if 'tool' in file_name:
            dataset_info[file_name]["columns"]["system"] = "system"
    
    with open(os.path.join(args.output_path, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)


def main():
    random.seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str)
    parser.add_argument("--output_path", "-o", type=str, required=True)
    parser.add_argument("--max_length", "-l", type=int, required=True)
    parser.add_argument("--balanced", "-b", type=str)
    parser.add_argument("--na_ratio", "-nr", nargs='+', type=float)
    # parser.add_argument("--trans_tool_ratio", "-tr", type=float, default=None)
    parser.add_argument("--stage", "-s", choices=['sft', 'dpo'], type=str)
    parser.add_argument("--format", "-f", choices=['sharegpt', 'openai'], type=str)
    parser.add_argument("--multi_pairs", "-m", choices=['True', 'False'], type=str)
    parser.add_argument("--trans_style", "-ts", choices=['none', 'base', 'cot'], type=str)
    parser.add_argument("--trans_label", "-tl", choices=['sample', 'reference'], type=str)
    parser.add_argument("--merge_data", "-md", choices=['True', 'False'], type=str)
    parser.add_argument("--language", type=str)
    # parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--tool_data_size", type=int, default=None)
    parser.add_argument("--trans_data_size", type=int, default=None)
    args = parser.parse_args()

    if args.stage == 'sft' and args.multi_pairs == 'True':
        raise ValueError("For SFT stage, multi_pairs should be False.")
    
    if args.stage == 'sft' and args.format == 'sharegpt':
        raise ValueError("For SFT stage, format should be openai.")
    
    if args.balanced == 'True':
        if args.na_ratio is None:
            raise ValueError("na_ratio should be provided when balanced is True.")
        if len(args.na_ratio) < 3:
            raise ValueError("na_ratio should have three values for three tools (view_summaries, view_pages, look_up_entities).")

    tool_map = {3: "view_summaries", 5: "view_pages", 7: "look_up_entities"}
    whole_data_na = {'view_summaries': [], 'view_pages': [], 'look_up_entities': []}
    whole_data_others = {'view_summaries': [], 'view_pages': [], 'look_up_entities': []}

    chapter_list = os.listdir(args.input_path)
        
    for chapter in tqdm(chapter_list):
        chapter_path = os.path.join(args.input_path, chapter)
        if args.multi_pairs == 'False':
            chosen_file_list = glob.glob(os.path.join(chapter_path, "chosen_*.json"))
        else:
            chosen_file_list = glob.glob(os.path.join(chapter_path, "multi_chosen_*.json"))
        chosen_file_list = sorted(chosen_file_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))
        rejected_file_list = [file.replace("chosen", "rejected") for file in chosen_file_list]
        for chosen_file_path, rejected_file_path in zip(chosen_file_list, rejected_file_list):
            if not os.path.isfile(chosen_file_path) or not os.path.isfile(rejected_file_path):
                continue
            with open(chosen_file_path, "r") as cf, open(rejected_file_path, "r") as rf:
                chosen_data = json.load(cf)
                rejected_data = json.load(rf)
                
            assert len(chosen_data[-1]) == 7
            assert len(rejected_data[-1]) == 7
                
            for chosen_messages, rejected_messages in zip(chosen_data, rejected_data):

                for messages in [chosen_messages, rejected_messages]:
                    for message in messages:
                        if message['role'] != 'assistant':
                            continue
                        message["content"] = re.sub(r"\s*\(selected\)|\s*\(not selected\)|\s*\(unselected\)", "", message["content"])
                        message["content"] = message["content"].replace("[Selected Items]", "[Selection]")
                        message["content"] = re.sub(r"\[Rejection\].*", "", message["content"], flags=re.DOTALL)

                selection_match = re.search(r'\[Selection\]\s*([\s\S]*?)(?:\n\[|$)', chosen_messages[-1]['content'])
                selection = selection_match.group(1).strip()
                chosen_selection = selection
            
                if chosen_selection == 'N/A':
                    whole_data_na[tool_map[len(chosen_messages)]].append((chosen_messages, rejected_messages))
                else:
                    whole_data_others[tool_map[len(chosen_messages)]].append((chosen_messages, rejected_messages))
    
    for data in whole_data_na:
        print(f"{data} Others: {len(whole_data_others[data])}")
        print(f"{data} N/A: {len(whole_data_na[data])}")
    
    tokenizer = AutoTokenizer.from_pretrained('/path/to/llm/tokenizer')
    
    balanced = eval(args.balanced)
    final_data = []
    if balanced:
        other_size = min(len(whole_data_others['view_summaries']), len(whole_data_others['view_pages']), len(whole_data_others['look_up_entities']))
        na_size = int(other_size * args.na_ratio)
        
        for tool in whole_data_others:
            final_data.extend(random.sample(whole_data_others[tool], min(other_size, len(whole_data_others[tool]))))
            final_data.extend(random.sample(whole_data_na[tool], min(na_size, len(whole_data_na[tool]))))
            print(f"{tool} na size: {min(na_size, len(whole_data_na[tool]))}, other size: {min(other_size, len(whole_data_others[tool]))}")
    else:
        for tool in whole_data_others:
            final_data.extend(whole_data_others[tool])
            final_data.extend(whole_data_na[tool])
            
    skip_num = {'length': 0, 'score': 0, 'message': 0, 'degraded_cot': 0}
    tool_data = []
    for chosen_messages, rejected_messages in tqdm(final_data):
        assert len(chosen_messages) == len(rejected_messages)
        assert chosen_messages[:-1] == rejected_messages[:-1], f"Chosen and rejected messages do not match:\n{json.dumps(chosen_messages[:-1], indent=2, ensure_ascii=False)}\n\n{json.dumps(rejected_messages[:-1], indent=2, ensure_ascii=False)}"
        if len(tokenizer.apply_chat_template(chosen_messages, add_generation_prompt=False)) > args.max_length or len(tokenizer.apply_chat_template(rejected_messages, add_generation_prompt=False)) > args.max_length:
            skip_num['length'] += 1
            continue
        if args.stage == 'sft':
            tool_data.append({'messages': chosen_messages})
        elif args.stage == 'dpo':
            if args.format == 'sharegpt':
                role_map = {'user': 'human', 'assistant': 'gpt'}
                assert chosen_messages[0]['role'] == 'system'
                system_message = chosen_messages[0]['content']
                reformat_chosen_messages = [{'from': role_map[msg['role']], 'value': msg['content']} for msg in chosen_messages[1:]]
                reformat_rejected_messages = [{'from': role_map[msg['role']], 'value': msg['content']} for msg in rejected_messages[1:]]
                tool_data.append({
                    'conversations': reformat_chosen_messages[:-1],
                    'chosen': reformat_chosen_messages[-1],
                    'rejected': reformat_rejected_messages[-1],
                    'system': system_message
                })
            elif args.format == 'openai':
                tool_data.append({
                    'prompt': chosen_messages[:-1],
                    'chosen': [chosen_messages[-1]],
                    'rejected': [rejected_messages[-1]]
                })
        else:
            raise ValueError(f"Unknown stage: {args.stage}")
    print(f"##### {args.stage} Tool data size: {len(tool_data)} #####")
    
    if args.tool_data_size is not None:
        if len(tool_data) > args.tool_data_size:
            print(f"Sampled tool data size: {args.tool_data_size}, original size: {len(tool_data)}")
            tool_data = random.sample(tool_data, args.tool_data_size)
        else:
            print(f"Tool data size is less than or equal to {args.tool_data_size}, original size: {len(tool_data)}")

    trans_data = []
    if args.trans_style != 'none':
        for chapter in tqdm(chapter_list):
            chapter_path = os.path.join(args.input_path, chapter)
            if args.trans_style == 'base':
                trans_file_list = glob.glob(os.path.join(chapter_path, "base_trans_*.json"))
            elif args.trans_style == 'cot':
                trans_file_list = glob.glob(os.path.join(chapter_path, "cot_trans_*.json"))
            trans_file_list = sorted(trans_file_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

            for trans_file_path in trans_file_list:
                if not os.path.isfile(trans_file_path):
                    continue
                with open(trans_file_path, "r") as f:
                    page_trans_data_list = json.load(f)

                for page_trans_data in page_trans_data_list:
                    target_samples = page_trans_data['trans_samples']
                    
                    sample_ids = list(range(len(target_samples)))
                    chosen_id = max(sample_ids, key=lambda x: target_samples[x]['score'])
                    reject_id = min(sample_ids, key=lambda x: target_samples[x]['score'])
                    
                    chosen_score = target_samples[chosen_id]['score']
                    reject_score = target_samples[reject_id]['score']
                    
                    if chosen_score == float('-inf') or chosen_score == reject_score:
                        skip_num['score'] += 1
                        continue
                    chosen_messages = target_samples[chosen_id]['messages']
                    rejected_messages = target_samples[reject_id]['messages']
                    chosen_messages = [{'role': msg['role'], 'content': msg['content'] if isinstance(msg['content'], str) else msg['content']['content']} for msg in chosen_messages]
                    rejected_messages = [{'role': msg['role'], 'content': msg['content'] if isinstance(msg['content'], str) else msg['content']['content']} for msg in rejected_messages]

                    if args.trans_label == 'reference':
                        chosen_messages = generate_reference_label(chosen_messages, page_trans_data['reference_lines'])

                    if args.trans_style == 'cot' and len(chosen_messages) == 2:
                        skip_num['degraded_cot'] += 1
                        
                    if args.trans_style == 'base' or (args.trans_style == 'cot' and len(chosen_messages) == 2):
                        slice_ids = [2]
                    else:
                        slice_ids = [2, 4]
                    start_id = 0
                    # system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                    if chosen_messages[0]['role'] == 'system':
                        slice_ids = [i + 1 for i in slice_ids]
                        start_id = 1
                        system_message = chosen_messages[0]['content']
                        
                    assert len(chosen_messages) == slice_ids[-1], f"{len(chosen_messages)} != {slice_ids[-1]}"
                    assert len(rejected_messages) == slice_ids[-1]
                    
                    role_map = {'user': 'human', 'assistant': 'gpt'}
                    
                    for slice_id in slice_ids:
                        if args.stage == 'sft':
                            trans_data.append({'messages': chosen_messages[:slice_id]})
                        else:
                            reformat_chosen_messages = [{'from': role_map[msg['role']], 'value': msg['content']} for msg in chosen_messages[start_id:slice_id]]
                            reformat_rejected_messages = [{'from': role_map[msg['role']], 'value': msg['content']} for msg in rejected_messages[start_id:slice_id]]
                            trans_data.append({
                                'conversations': reformat_chosen_messages[:-1],
                                'chosen': reformat_chosen_messages[-1],
                                'rejected': reformat_rejected_messages[-1],
                            })
                    
    print(f"##### {args.stage} Trans data size: {len(trans_data)} #####")

    if args.trans_data_size is not None:
        if len(trans_data) > args.trans_data_size:
            print(f"Sampled trans data size: {args.trans_data_size}, original size: {len(trans_data)}")
            trans_data = random.sample(trans_data, args.trans_data_size)
        else:
            print(f"Trans data size is less than or equal to {args.trans_data_size}, original size: {len(trans_data)}")

    print("Skipped:", skip_num)
    print(f"Tool data size: {len(tool_data)}")
    print(f"Translate data size: {len(trans_data)}")

    if args.merge_data == 'True':
        write_data = tool_data + trans_data
        random.shuffle(write_data)
        
        with open(os.path.join(args.output_path, f"{args.stage}_{args.language}.json"), "w") as f:
            json.dump(write_data, f, ensure_ascii=False, indent=4)
            
        write_data_info(args, f"{args.stage}_{args.language}")
            
    else:
        with open(os.path.join(args.output_path, f"{args.stage}_{args.language}_tool.json"), "w") as f:
            json.dump(tool_data, f, ensure_ascii=False, indent=4)
            
        write_data_info(args, f"{args.stage}_{args.language}_tool")
        
        with open(os.path.join(args.output_path, f"{args.stage}_{args.language}_trans_{args.trans_style}_{args.trans_label}.json"), "w") as f:
            json.dump(trans_data, f, ensure_ascii=False, indent=4)
            
        write_data_info(args, f"{args.stage}_{args.language}_trans_{args.trans_style}_{args.trans_label}")

    with open(os.path.join(args.output_path, "data_card.txt"), "a") as f:
        f.write(f"{args.stage}_{args.language}_tool: {len(tool_data)}\n{args.stage}_{args.language}_trans_{args.trans_style}_{args.trans_label}: {len(trans_data)}\n")
        

if __name__ == "__main__":
    main()
    