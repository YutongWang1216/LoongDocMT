from langcodes import Language
from openai import AsyncOpenAI
import argparse
import asyncio
import os
import re


PROMPT_TEMPLATE = """You are an expert linguist and translation quality evaluator. Your task is to evaluate the quality of a document-level translation from {src_language} to {tgt_language} based solely on the Source Document, the Reference Document (Gold Standard), and the Hypothesis Document (Model Output).

Please assess the [Hypothesis] text as a whole against the [Source] and [Reference]. Provide a holistic score from 0 to 100 for the following three specific dimensions, where 0 represents a complete failure and 100 represents a perfect, native-level professional translation.

[Source]:
{src_doc}

[Reference]:
{ref_doc}

[Hypothesis]:
{hyp_doc}

[Evaluation Dimensions]:

1. **General Quality**:
   - Focuses on accuracy (faithfulness to the source meaning) and fluency (grammatical correctness and natural flow).
   - A high score means the translation is precise, preserves the original meaning without omission or hallucination, and reads naturally in the target language.

2. **Cohesion**:
   - Focuses on the explicit linking words and grammatical connections between sentences and clauses (e.g., correct use of pronouns, conjunctions, substitution, and ellipsis).
   - A high score means the text is syntactically well-connected, and references (anaphora/cataphora) are clear and unambiguous throughout the document.

3. **Coherence**:
   - Focuses on the logical arrangement and semantic relationships of ideas. It assesses whether the text "makes sense" as a whole narrative or argument.
   - A high score means the discourse flows logically, follows the thought patterns/conventions of the target culture, and is easy for a reader to understand without referring to the source.

4. **Style Consistency**:
   - Focuses on the maintenance of tone, register (formal/informal), and voice throughout the document.
   - A high score means the translation maintains a unified style that matches the source text's intent (e.g., not switching between academic and slang phrasing).

5. **Terminology Consistency**:
   - Focuses on the consistent translation of specific terms, entities, and keywords across the entire document.
   - A high score means the same concept is translated using the same term throughout, avoiding confusion caused by using multiple synonyms for the same specific entity.

[Output Requirement]:
For each dimension, provide a score (0-100) and a brief justification based on the whole document.

Your response must strictly follow this format:

### Evaluation Report

**1. General Quality**
Score: [0-100]
Rationale: ...

**2. Cohesion**
Score: [0-100]
Rationale: ...

**3. Coherence**
Score: [0-100]
Rationale: ...

**4. Style Consistency**
Score: [0-100]
Rationale: ...

**5. Terminology Consistency**
Score: [0-100]
Rationale: ..."""


async def invoke_llm(prompt):

    response = await client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.,
        # top_p=1.0,
    )
    return response.choices[0].message.content


async def process_single_file(src_file, tgt_file, ref_file, src_lang, tgt_lang, metrics, semaphore):
    async with semaphore:
        with open(src_file, 'r') as sf, open(tgt_file, 'r') as tf, open(ref_file, 'r') as rf:
            source_doc = sf.read()
            target_doc = tf.read()
            reference_doc = rf.read()

        lang_map = {
            'en': 'English',
            'zh': 'Chinese',
            'fr': 'French',
            'de': 'German'
        }
        src_language_name = Language.make(language=src_lang).display_name()
        tgt_language_name = Language.make(language=tgt_lang).display_name()

        prompt = PROMPT_TEMPLATE.format(
            src_language=src_language_name,
            tgt_language=tgt_language_name,
            src_doc=source_doc,
            ref_doc=reference_doc,
            hyp_doc=target_doc,
        )
        evaluation_report = await invoke_llm(prompt)

        items = re.findall(r'\*\*\d+\.\s*([^\*]+)\*\*\s*Score:\s*([0-9]+)', evaluation_report)
        scores = {k: None for k in metrics}
        for name, score in items:
            scores[name.strip()] = int(score.strip())
        valid_scores = [v for v in scores.values() if v is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        info = '\t'.join([f'{k}: {v if v is not None else "N/A"}' for k, v in scores.items()])
        info += f'\tAverage: {avg_score:.2f}'
        print(f"{tgt_file}\t{info}")
        return tgt_file, scores, info


async def main():
    metrics = [
        "General Quality",
        "Cohesion",
        "Coherence",
        "Style Consistency",
        "Terminology Consistency",
    ]
    src_lang, tgt_lang = args.language.split('-')

    semaphore = asyncio.Semaphore(30)

    tasks = [
        process_single_file(src_file, tgt_file, ref_file, src_lang, tgt_lang, metrics, semaphore)
        for src_file, tgt_file, ref_file in zip(args.source_files, args.target_files, args.reference_files)
    ]
    results = await asyncio.gather(*tasks)

    whole_scores = {k: [] for k in metrics}
    write_txt = ""
    for tgt_file, scores, info in results:
        for k, v in scores.items():
            if v is not None:
                whole_scores[k].append(v)
        write_txt += f"{tgt_file}\t{info}\n"
    info = "Average Scores\t"
    avg_list = []
    for metric in metrics:
        avg = sum(whole_scores[metric]) / len(whole_scores[metric])
        info += f"{metric}:\t{avg:.2f}\t"
        avg_list.append(avg)
    info = f"{info.strip()}\nMeta\t{sum(avg_list) / len(avg_list):.2f}\n"
    print(info)
    write_txt += info
    with open(args.output_file, 'w') as out_f:
        out_f.write(write_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_files', '-s', type=str, nargs='+', required=True)
    parser.add_argument('--target_files', '-t', type=str, nargs='+', required=True)
    parser.add_argument('--reference_files', '-r', type=str, nargs='+', required=True)
    parser.add_argument('--language', '-l', type=str, required=True)
    parser.add_argument('--output_file', '-o', type=str, required=True)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--api_key', type=str, default=os.getenv('OPENAI_API_KEY'))
    parser.add_argument('--base_url', type=str, default=os.getenv('OPENAI_BASE_URL'))
    args = parser.parse_args()

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    asyncio.run(main())
