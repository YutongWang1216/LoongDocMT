import asyncio
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from comet import load_from_checkpoint
import argparse
import uvicorn


app = FastAPI(title="COMET Evaluation API")

MODEL_PATH = "/path/to/wmt22-comet-da/model.ckpt"
GPUS = int(os.environ.get("COMET_GPUS", "0"))

model = load_from_checkpoint(MODEL_PATH)
if GPUS > 0:
    model = model.cuda()
print(f"[deploy] model loaded on {'GPU' if GPUS > 0 else 'CPU'}", flush=True)

request_queue: asyncio.Queue = None
MAX_BATCH_INSTANCES = 256
MAX_WAIT_MS = 500


async def batch_worker():
    while True:
        first_item = await request_queue.get()
        batch = [first_item]
        total = len(first_item["instances"])

        deadline = asyncio.get_event_loop().time() + MAX_WAIT_MS / 1000
        while total < MAX_BATCH_INSTANCES:
            timeout = deadline - asyncio.get_event_loop().time()
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(request_queue.get(), timeout=timeout)
                batch.append(item)
                total += len(item["instances"])
            except asyncio.TimeoutError:
                break

        all_instances = []
        for item in batch:
            all_instances.extend(item["instances"])

        print(f"[batch_worker] requests={len(batch)}, instances={len(all_instances)}", flush=True)

        try:
            output = model.predict(all_instances, batch_size=64, gpus=GPUS)
            print(output)
            all_scores = output["scores"]

            offset = 0
            for item in batch:
                n = len(item["instances"])
                item["future"].set_result(all_scores[offset:offset + n])
                offset += n
        except Exception as e:
            for item in batch:
                if not item["future"].done():
                    item["future"].set_exception(e)


@app.on_event("startup")
async def startup():
    global request_queue
    request_queue = asyncio.Queue()
    asyncio.create_task(batch_worker())


class InputData(BaseModel):
    instances: List[Dict]


@app.post("/evaluate")
async def evaluate(data: InputData):
    future = asyncio.get_event_loop().create_future()
    await request_queue.put({"instances": data.instances, "future": future})
    scores = await future
    return {"scores": scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy COMET Evaluation API")
    parser.add_argument("--port", type=int, default=8090, help="Port to run the API on")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)