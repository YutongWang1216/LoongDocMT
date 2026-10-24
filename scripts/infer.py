from trajectory import Trajectory
import argparse
from openai import AsyncOpenAI
from openai import OpenAI
import os
import asyncio
from sentence_transformers import SentenceTransformer
from information_pool import InformationPool


async def process_file(src_file, output_dir, args, infer_client, schedule_client, sentence_encoder, src_lang, tgt_lang, url):
    with open(src_file, "r") as sf:
        src_list = [line.strip() for line in sf]

    src_pages = [src_list[i:i + args.window_size] for i in range(0, len(src_list), args.window_size)]

    client = OpenAI(
        api_key="EMPTY",
        base_url=f"{url}/v1",
        timeout=200
    )

    init_info_pool = InformationPool(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        llm_client=client,
        source_pages=src_pages,
        reference_pages=[None] * len(src_pages),
        encoder=sentence_encoder,
        temperature=args.infer_temperature
    )

    trajectory = Trajectory(
        info_pool=init_info_pool,
        async_schedule_client=schedule_client,
        async_inference_client=infer_client,
        save_path=output_dir,
        hyp_file=f'{os.path.basename(src_file)}'.replace(f'{src_lang}.', f'{tgt_lang}.'),
        stage='test',
        schedule_temperature=args.schedule_temperature,
        infer_temperature=args.infer_temperature,
        translate_style=args.translate_style,
    )

    await trajectory.run_async()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--window_size", "-w", type=int, required=True)
    parser.add_argument("--infer_address", "-ia", type=str, required=True)
    parser.add_argument("--schedule_address", "-sa", type=str, required=True)
    parser.add_argument("--language", "-l", type=str, required=True)
    parser.add_argument("--schedule_temperature", "-st", type=float, required=True)
    parser.add_argument("--infer_temperature", "-it", type=float, required=True)
    parser.add_argument("--translate_style", "-ts", type=str, choices=["base", "cot"], required=True)
    parser.add_argument("--encoder_path", "-e", type=str, required=True)
    parser.add_argument("--parallel", "-p", action="store_true", help="Process source files in parallel (default: serial)")
    args = parser.parse_args()

    src_lang = args.language[:2]
    tgt_lang = args.language[-2:]

    if not args.infer_address.startswith("http://"):
        args.infer_address = "http://" + args.infer_address
    if not args.schedule_address.startswith("http://"):
        args.schedule_address = "http://" + args.schedule_address

    infer_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"{args.infer_address}/v1",
        timeout=50
    )

    schedule_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"{args.schedule_address}/v1",
        timeout=50
    )

    os.makedirs(args.output_dir, exist_ok=True)

    sentence_encoder = SentenceTransformer(args.encoder_path)

    src_files = []
    for f in os.listdir(args.input_dir):
        if not (f.startswith(src_lang + '.') and os.path.isfile(os.path.join(args.input_dir, f))):
            continue
        hyp_file = os.path.join(args.output_dir, f.replace(f'{src_lang}.', f'{tgt_lang}.'))
        if not os.path.exists(hyp_file):
            src_files.append(os.path.join(args.input_dir, f))
    src_files.sort(key=lambda x: int(os.path.basename(x).split('.')[-1]))
    print(f"Found {len(src_files)} files to process:\n{src_files}")

    if args.parallel:
        semaphore = asyncio.Semaphore(30)

        async def process_file_with_semaphore(src_file):
            async with semaphore:
                await process_file(src_file, args.output_dir, args, infer_client, schedule_client, sentence_encoder, src_lang, tgt_lang, args.schedule_address)

        tasks = [process_file_with_semaphore(src_file) for src_file in src_files]
        await asyncio.gather(*tasks)
    else:
        for src_file in src_files:
            await process_file(src_file, args.output_dir, args, infer_client, schedule_client, sentence_encoder, src_lang, tgt_lang, args.schedule_address)


if __name__ == '__main__':
    asyncio.run(main())
