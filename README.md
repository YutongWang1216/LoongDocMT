# LoongDocMT
This repository anonymously releases the codes and data for the paper Loong: A Human-Like Long Document Translation Agent with Observe-and-Act Adaptive Context Selection

## **🔗 Quick Links**

- **[About Loong](#about)**
- **[File Structure](#structure)**
- **[Requirements](#requirements)**
- **[Quick Start](#start)**

## **🐉 About Loong**<a name="about"></a>
Loong is a human-like long document translation agent that employs reasoning-driven adaptive context selection optimized via reinforcement learning to resolve context limitation and noise issues in DocMT-LLMs, achieving significant gains in document translation quality and ultra-long document stability.

## **📜 File Structure**<a name="structure"></a>
| Directory           | Contents                                                       |
| ------------------- | -------------------------------------------------------------- |
| `./data`            | Testing Data                                                   |
| `./scripts`         | Shared Python modules used by sampling and inference           |
| `./scripts/prompts` | Prompts for the agent                                          |
| `./sample`          | Training-data sampling launchers (`run_sample.sh`, `run_process.sh`, `process.py`) |
| `./train`           | LLaMA-Factory recipes and launcher for SFT + DPO fine-tuning   |
| `./inference`       | Inference launcher (`run_infer.sh`)                            |
| `./evaluate`        | Scripts for sCOMET, dCOMET and LLM-as-a-Judge                  |
| `./results`         | Testing outputs                                                |


## **🛠️ Requirements**<a name="requirements"></a>
Loong conduct LLM inference via vLLM deployments.
- Python 3.11.11
- Pytorch 2.9.1+cu121
- transformers==4.57.0
- openai==1.90.0
- vllm==0.11.2
- llamafactory==0.9.4.dev0
- sentence-transformers==4.1.0

## **🚀 Quick Start**<a name="start"></a>

### **Installation**

```bash
pip install -r requirements.txt
```

### **Training Data Sampling**

Training data is produced in two steps: first sample raw trajectories with
`run_sample.sh`, then convert them into LLaMA-Factory SFT/DPO datasets with
`run_process.sh`. A COMET scoring service must be deployed beforehand, since the
sampling pipeline calls it on every trajectory.

#### Step 0 — Deploy COMET service

- evaluate/deploy_comet.sh

Launches a COMET model server in the background (logs to `evaluate/deploy.log`).
The endpoint exposed here must match the `comet_apis` entries used by Step 1
(and the `comet_api` argument used later by sCOMET evaluation).

Set the following inside the script before running:

- `CUDA_VISIBLE_DEVICES` — GPU id(s) to bind to.
- `COMET_GPUS`           — number of GPUs to use for the service.
- `--port`               — listening port (default `8090`).

Also set the COMET checkpoint path inside `deploy.py` (the `wmt22-comet-da` model
to serve).

```bash
bash evaluate/deploy_comet.sh
```

#### Step 1 — Raw trajectory sampling

- sample/run_sample.sh

Runs the observe-and-act sampling pipeline over the News Commentary v18.1 source files
to collect training data.
The input directory is expected to hold per-language sub-directories whose files are
named `${src_lang}.${doc_id}` and `${tgt_lang}.${doc_id}` (e.g., `en-zh/en.0`,
`en-zh/zh.0`). Outputs are written under `${out_dir}/${language}/${doc_id}`.

Set the following inside the script before running:

- `in_dir`         — parent directory containing per-language sub-dirs of News Commentary v18.1 raw training files.
- `out_dir`        — output directory for sampled trajectories (default `./results`).
- `languages`      — bash array of one or more translation directions, choices=[en-zh,en-de,en-fr,zh-en,de-en,fr-en].
- `urls`           — bash array of one or more deployed vLLM model APIs (e.g., `127.0.0.1:8000`); the worker pool size equals the number of URLs.
- `comet_apis`     — bash array of one or more deployed COMET model APIs (e.g., `127.0.0.1:8088`); workers are sharded across them.
- `tokenizer_path` — path to the LLM's tokenizer.
- `encoder_path`   — path to the `all-distilroberta-v1` checkpoint.
- `window_size`    — number of sentences per page within a document.

```bash
bash sample/run_sample.sh
```

#### Step 2 — Dataset construction

- sample/run_process.sh

Iterates over `en-zh`, `en-de`, `en-fr` and invokes `process.py` twice per language —
once for the SFT split (`openai` format) and once for the DPO split (`sharegpt` format) —
producing one `*_tool.json` and one `*_trans_*.json` file per stage plus a
`dataset_info.json` registry consumable by LLaMA-Factory.

Set the following inside the script before running:

- `--input_path`        — directory holding the per-chapter trajectories emitted by Step 1.
- `--output_path`       — directory where the SFT/DPO JSON files and `dataset_info.json` are written.
- `--language`          — translation direction; iterated by the loop.

Also set the tokenizer path inside `process.py` (used for length filtering):

- `tokenizer = AutoTokenizer.from_pretrained(...)` — replace with the path to the LLM's tokenizer.

```bash
bash sample/run_process.sh
```

### **Model Tuning**

- train/run_train.sh

Fine-tunes the base LLM in two stages: full-parameter SFT followed by LoRA-based DPO,
driven by LLaMA-Factory recipes.

Set the following inside the recipe files before running:

- `full_sft.yaml`
  - `model_name_or_path` — path to the pre-trained LLM checkpoint.
  - `deepspeed`          — path to `ds_z3_config.json`.
  - `dataset_dir`        — path to the SFT training data.
  - `template`           — `qwen` for Qwen2.5, `qwen3` for Qwen3, `llama3` for Llama3.1.
- `lora_dpo.yaml`
  - `model_name_or_path` — path to the SFT checkpoint.
  - `dataset_dir`        — path to the DPO training data.
  - `template`           — `qwen` for Qwen2.5, `qwen3_nothink` for Qwen3, `llama3` for Llama3.1.

```bash
bash train/run_train.sh
```

### **Inference**

- inference/run_infer.sh

Translates each document under the given source test file with the trained Loong agent
and writes hypotheses to the result directory.

Set the following inside the script before running:

- `address`  — deployed vLLM model API (e.g., `127.0.0.1:8000`).
- `language` — translation direction, choices=[en-zh,en-de,en-fr,zh-en,de-en,fr-en].
- `src_file` — source test file.

```bash
bash inference/run_infer.sh
```

### **Evaluation**

We provide three evaluators under `./evaluate` to assess translation quality from complementary perspectives:
sentence-level COMET (sCOMET), document-level COMET (dCOMET), and an LLM-as-a-Judge protocol.

All three scripts share the same I/O convention:

- `data_dir`   — directory containing the source and reference files, named `${src_lang}.${doc_id}` and `${tgt_lang}.${doc_id}` (e.g., `en.0`, `zh.0`).
- `result_dir` — directory containing the hypothesis files produced by inference, named `${tgt_lang}.${doc_id}`.
- `language`   — translation direction, choices=[en-zh,en-de,en-fr,zh-en,de-en,fr-en].

#### sCOMET (sentence-level COMET)

- evaluate/eval_scomet.sh

Posts source–hypothesis–reference triples to a deployed COMET service and reports the
per-document average and the overall average. Output is written to `${result_dir}/comet.txt`.

```bash
bash eval_scomet.sh <data_dir> <result_dir> <language> [comet_api]
# comet_api defaults to 127.0.0.1:8090
```

#### dCOMET (document-level COMET)

- evaluate/eval_dcomet_total.sh

Concatenates all documents and uses `comet-score` with document-id boundaries to compute
document-level COMET in a single pass. Output is appended to `${result_dir}/doccomet_total.txt`.

The path to the `wmt22-comet-da` checkpoint can be provided either as a 4th positional
argument or through the `COMET_MODEL_PATH` environment variable.

```bash
bash eval_dcomet_total.sh <data_dir> <result_dir> <language> <comet_model_path>
# or:
export COMET_MODEL_PATH=/path/to/wmt22-comet-da/model.ckpt
bash eval_dcomet_total.sh <data_dir> <result_dir> <language>
```

#### LLM-as-a-Judge

- evaluate/eval_llm.sh

Prompts a judge LLM to score each hypothesis document holistically on five dimensions —
General Quality, Cohesion, Coherence, Style Consistency, and Terminology Consistency
(0–100 each, plus a Meta average). Output is written to `${result_dir}/llm_${model}.txt`.

Set the judge model inside `eval_llm.sh`, and provide OpenAI-compatible credentials
either via environment variables or by editing the script:

```bash
# eval_llm.sh
model=      # judge model name (e.g., gpt-4.1)

# environment variables (read by default)
export OPENAI_API_KEY=...     # OpenAI-compatible API key
export OPENAI_BASE_URL=...    # OpenAI-compatible endpoint
```

```bash
bash eval_llm.sh <data_dir> <result_dir> <language>
```

