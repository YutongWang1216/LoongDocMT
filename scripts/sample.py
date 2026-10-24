import concurrent.futures
import argparse
import os
import random
import time
import traceback
import asyncio
from sentence_transformers import SentenceTransformer
import torch
from trajectory import Trajectory
from openai import OpenAI, AsyncOpenAI
from information_pool import InformationPool
import glob


random.seed(42)

def run_trajectory(src_file:str, ref_file:str, out_path:str, url:str, language:str, window_size:int, timeout:int, comet_api: str, tokenizer_path: str, sentence_encoder):


    client = OpenAI(
        api_key="EMPTY",
        base_url=f"{url}/v1",
        timeout=timeout
    )
    os.makedirs(f"{out_path}", exist_ok=True)

    with open(src_file, "r") as sf, open(ref_file, "r") as rf:
        src_list = [line.strip() for line in sf]
        ref_list = [line.strip() for line in rf]

    src_pages = [src_list[i:i + window_size] for i in range(0, len(src_list), window_size)]
    ref_pages = [ref_list[i:i + window_size] for i in range(0, len(ref_list), window_size)]

    src_lang = language[:2]
    tgt_lang = language[-2:]

    init_info_pool = InformationPool(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        llm_client=client,
        source_pages=src_pages,
        reference_pages=ref_pages,
        encoder=sentence_encoder
    )

    async def _run():
        async with AsyncOpenAI(
            api_key="EMPTY",
            base_url=f"{url}/v1",
            timeout=timeout
        ) as async_client:
            trajectory = Trajectory(
                info_pool=init_info_pool,
                schedule_client=client,
                inference_client=client,
                async_schedule_client=async_client,
                async_inference_client=async_client,
                save_path=out_path,
                comet_api=comet_api,
                stage='train',
                sample_strategy='random',
                tokenizer_path=tokenizer_path
            )
            await trajectory.run_async()

    asyncio.run(_run())


def run_sample(model_id, src_file, ref_file, out_path, language, url, comet_api, tokenizer_path, encoder):

    run_trajectory(
        src_file=src_file,
        ref_file=ref_file,
        out_path=out_path,
        url=url,
        language=language,
        window_size=args.window_size,
        timeout=50,
        comet_api=comet_api,
        tokenizer_path=tokenizer_path,
        sentence_encoder=encoder
    )


def get_tasks(in_dir, out_dir, languages):
    
    all_tasks = []
    labels = []
    
    for lang in languages:
        src_lang, tgt_lang = lang.split('-')
        
        src_doc_list = glob.glob(f'{in_dir}/{lang}/{src_lang}.*')
        for src_doc in src_doc_list:
            
            tgt_doc = src_doc.replace(f'{src_lang}.', f'{tgt_lang}.')

            doc_id = src_doc.split('.')[-1]
            result_path = f'{out_dir}/{src_lang}-{tgt_lang}/{doc_id}'
            src_file = src_doc
            ref_file = tgt_doc
            
            if not os.path.exists(src_file) or not os.path.exists(ref_file):
                continue
            
            all_tasks.append((src_file, ref_file, result_path, lang))
            if not os.path.exists(result_path):
                labels.append(False)
            else:
                labels.append(True)
        
    assert len(all_tasks) == len(labels)
    ids = list(range(len(all_tasks)))
    random.shuffle(ids)
    unfinished_tasks = [all_tasks[i] for i in ids if not labels[i]]
    
    return unfinished_tasks


def init_encoders(max_workers, ckpt_path):
    encoders = []
    if torch.cuda.is_available():
        use_gpu = True
        gpu_num = torch.cuda.device_count()
    else:
        use_gpu = False
        gpu_num = 0
    for i in range(max_workers):
        devide = f'cuda:{i % gpu_num}' if use_gpu else 'cpu'
        print(f'Loading encoder #{i} on device {devide}...')
        encoder = SentenceTransformer(ckpt_path, device=devide)
        encoders.append(encoder)
        print(f'Loaded encoder #{i} on device {devide}.')
        time.sleep(2)
    return encoders


def main():

    tasks = get_tasks(args.in_dir, args.out_dir, args.languages)
    max_workers = len(args.urls)

    sentence_encoders = init_encoders(max_workers, ckpt_path=args.encoder_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {}
        for i in range(max_workers):
            if i < len(tasks):
                task = tasks[i]
                future = executor.submit(
                    run_sample,
                    model_id=i,
                    src_file=task[0],
                    ref_file=task[1],
                    out_path=task[2],
                    language=task[3],
                    url=args.urls[i],
                    comet_api=args.comet_api_list[int(i / max_workers * len(args.comet_api_list))],
                    tokenizer_path=args.tokenizer_path,
                    encoder=sentence_encoders[i]
                )
                future_to_model[future] = i

        next_task_index = max_workers

        while future_to_model:
            done, _ = concurrent.futures.wait(
                future_to_model.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in done:
                model_id = future_to_model.pop(future)
                try:
                    result = future.result()
                except Exception as e:
                    traceback.print_exc()
                    time.sleep(random.randint(5, 20))

                if next_task_index < len(tasks):
                    new_task = tasks[next_task_index]
                    new_future = executor.submit(
                        run_sample,
                        model_id=model_id,
                        src_file=new_task[0],
                        ref_file=new_task[1],
                        out_path=new_task[2],
                        language=new_task[3],
                        url=args.urls[model_id],
                        comet_api=args.comet_api_list[int(model_id / max_workers * len(args.comet_api_list))],
                        tokenizer_path=args.tokenizer_path,
                        encoder=sentence_encoders[model_id]
                    )
                    future_to_model[new_future] = model_id
                    next_task_index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--languages', type=str, nargs='+', required=True)
    parser.add_argument('--urls', type=str, nargs="+", required=True)
    parser.add_argument('--comet_api_list', type=str, nargs='+', required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--encoder_path', type=str, required=True)
    parser.add_argument('--window_size', type=int, required=True)
    args = parser.parse_args()
    main()
