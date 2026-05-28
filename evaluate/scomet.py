import asyncio
import aiohttp
import argparse
import os
import time
import glob
from pathlib import Path


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
                            return sum(sentence_level_scores) / len(sentence_level_scores) if len(sentence_level_scores) > 0 else 0.0
                        else:
                            end_time = time.time()
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


async def score_file(src_file, ref_file, hyp_file, comet_api):
    with open(src_file) as f:
        src_lines = [l.rstrip("\n") for l in f]
    with open(ref_file) as f:
        ref_lines = [l.rstrip("\n") for l in f]
    with open(hyp_file) as f:
        hyp_lines = [l.rstrip("\n") for l in f]

    assert len(src_lines) == len(ref_lines) == len(hyp_lines), \
        f"Line count mismatch: {src_file}({len(src_lines)}), {ref_file}({len(ref_lines)}), {hyp_file}({len(hyp_lines)})"

    instances = [
        {"src": s, "ref": r, "mt": h}
        for s, r, h in zip(src_lines, ref_lines, hyp_lines)
    ]
    scores = await async_get_comet_score(instances, comet_api=comet_api)
    return scores


def main():
    src_lang, tgt_lang = args.language.split("-")
    data_dir = args.data_dir
    result_dir = args.result_dir
    comet_api = args.url

    src_files = sorted(glob.glob(os.path.join(data_dir, f"{src_lang}.*")))

    async def score_one(src_file):
        ref_file = src_file.replace(f"{src_lang}.", f"{tgt_lang}.")
        hyp_file = os.path.join(result_dir, os.path.basename(ref_file))

        if not os.path.exists(ref_file):
            print(f"Reference file not found, skipping: {ref_file}")
            return None
        if not os.path.exists(hyp_file):
            print(f"Hypothesis file not found, skipping: {hyp_file}")
            return None

        scores = await score_file(src_file, ref_file, hyp_file, comet_api)
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"{hyp_file}: {avg*100:.2f}")
        return (hyp_file, len(scores), avg)

    async def run():
        tasks = [asyncio.create_task(score_one(src_file)) for src_file in src_files]
        results = await asyncio.gather(*tasks)

        valid = sorted([r for r in results if r is not None], key=lambda x: int(x[0].split(".")[-1]))  # Sort by file number
        all_scores_flat = [avg for _, _, avg in valid]
        overall = sum(all_scores_flat) / len(all_scores_flat) if all_scores_flat else 0.0

        out_path = os.path.join(result_dir, "comet.txt")
        with open(out_path, "w") as f:
            for hyp_file, n, avg in valid:
                f.write(f"{hyp_file}: {avg*100:.2f}\n")
            f.write(f"Average: {overall*100:.2f}\n")
        print(f"\nResults saved to {out_path}")
        print(f"Average: {overall*100:.2f}")

    asyncio.run(run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, required=True)
    parser.add_argument("--result_dir", "-r", type=str, required=True)
    parser.add_argument("--language", "-l", type=str, required=True)
    parser.add_argument("--url", "-u", type=str, required=True)
    args = parser.parse_args()
    
    main()
    