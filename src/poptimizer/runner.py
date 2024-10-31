import os
import argparse
import dataclasses

import orjson

import logging
import pathlib
import random
import sys
from copy import deepcopy

import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from xopen import xopen
import numpy as np

from poptimizer.prompting import Document
from poptimizer.util import (
    get_logprobs,
    shuffle_one_instance,
    set_seed
)

logging.basicConfig()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def compile_all_gt_example(input_path):
    """
    Read the input file and compile all the ground truth examples in a dict:
        {
            "question": question,
            "doc": gold_doc,
            "answers": answers
        }
    """
    gt_data = []
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = orjson.loads(line)
            question = input_example["question"]
            gt_doc = None
            for ctx in input_example["ctxs"]:
                one_document = Document.from_dict(ctx)
                if one_document.isgold:
                    gt_doc = ctx
                    break
            gt_data.append(
                {
                    "question": question,
                    "doc": gt_doc,
                    "answers": input_example["answers"]
                }
            )

    return gt_data



def get_input_documents(
    input_path,
    model_name,
    prompt_mention_random_ordering,
    use_random_ordering,
    max_prompt_length,
    restart_from_ind=0,
    num_shuffles_per_instance=1,
    num_docs=20,
    total_num_instances=100,
    seed=1000,
    append_answer=True,
    gold_index_position="lost-in-the-middle",
    prompt_type="ICQ",
    verbose=False
):
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    num_instances = restart_from_ind
    # Fetch all of the prompts
    if gold_index_position in ["lost-in-the-middle", "rotate", "every"]:
        logger.info(f"Reading input file {input_path}")
        with xopen(input_path) as fin:
            examples, prompts, documents = [], [], []

            fin = list(fin)
            np.random.shuffle(fin)

            num_instances = restart_from_ind
            for i, line in tqdm(enumerate(fin)):
                if i < restart_from_ind:
                    continue
                input_example = orjson.loads(line)
                all_documents, all_prompts = shuffle_one_instance(
                    input_example,
                    tokenizer,
                    num_docs=num_docs,
                    num_shuffles=num_shuffles_per_instance,
                    max_prompt_length=max_prompt_length,
                    prompt_mention_random_ordering=prompt_mention_random_ordering,
                    model_name=model_name,
                    did_format_warn=False,
                    use_random_ordering=use_random_ordering,
                    append_answer=append_answer,
                    gold_index_position=gold_index_position,
                    verbose=verbose,
                    prompt_type=prompt_type
                )
                examples.extend([deepcopy(input_example) for _ in all_prompts])
                prompts.extend(all_prompts)
                documents.extend(all_documents)

                if len(all_documents) > 0:
                    num_instances += 1
                if num_instances == total_num_instances:
                    break

    return examples, prompts, documents


def runner(
    input_path,
    model_name,
    temperature,
    top_p,
    prompt_mention_random_ordering,
    use_random_ordering,
    num_gpus,
    max_new_tokens,
    max_prompt_length,
    hf_cache_path,
    output_path,
    num_shuffles_per_instance=1,
    num_docs=20,
    total_num_instances=100,
    gpu_memory_utilization=0.6,
    seed=1000,
    append_answer=True,
    gold_index_position="every",
    prompt_type="ICQ",  
    verbose=False
):
    """
    Args: 
        input_path: str, path to the input file
        model_name: str, name of the model to use
        prompt_mention_random_ordering: bool, whether to shuffle the prompt and mention order
        use_random_ordering: bool, whether to shuffle the documents

        VLLM parameters:
            temperature: float, temperature for sampling
            top_p: float, top_p for sampling
            num_gpus: int, number of GPUs to use
            max_new_tokens: int, maximum number of tokens to generate
            max_prompt_length: int, maximum length of the prompt
            hf_cache_path: str, path to the huggingface cache
            gpu_memory_utilization: float, GPU memory utilization

        output_path: str, path to the output file
        num_shuffles_per_instance: int, number of shuffles per instance
        num_docs: int, number of documents to use
        total_num_instances: int, total number of instances to generate
        seed: int, random seed
        append_answer: bool, whether to append the answer to the prompt
        gold_index_position: str, where to place the gold index
            options: [every, lost-in-the-middle, rotate]
        prompt_type: str, type of prompt
            options: [ICQ, IQC, IQCQ]
        verbose: bool, whether to print verbose output
    """

    set_seed(seed)
    logger.info(f"Running with seed {seed}")
    logger.info(
        f"All hyperparameters:\n input_path={input_path}\nmodel_name={model_name}\ntemperature={temperature}\n"
        f"top_p={top_p}\nprompt_mention_random_ordering={prompt_mention_random_ordering}\n"
        f"use_random_ordering={use_random_ordering}\n"
        f"num_gpus={num_gpus}\nmax_new_tokens={max_new_tokens}\nmax_prompt_length={max_prompt_length}\n"
        f"hf_cache_path={hf_cache_path}\noutput_path={output_path}\nnum_shuffles_per_instance={num_shuffles_per_instance}\n"
        f"num_docs={num_docs}\ntotal_num_instances={total_num_instances}\ngpu_memory_utilization={gpu_memory_utilization}\n"
        f"seed={seed}\nappend_answer={append_answer}\ngold_index_position={gold_index_position}"
    )

    # Get responses for all of the prompts
    if not torch.cuda.is_available():
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")

    logger.info(f"Loading model {model_name}")

    extra_kw = {}
    if model_name.startswith("meta-llama/Meta-Llama-3.1"):
        # Currently, Llama-3.1 needs extra parameters to work properly with VLLM
        # as of VLLM 0.5.1
        extra_kw["rope_scaling"] = {
            "type": "yarn", 
            "factor": 8.0, 
            'original_max_position_embeddings': 8192
        }

    # Initialize vllm model
    model = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,
        distributed_executor_backend="mp",
        enforce_eager=True,
        trust_remote_code=True,
        max_num_batched_tokens=max_prompt_length,
        max_model_len=max_prompt_length,
        seed=seed,
        disable_custom_all_reduce=True,
        **extra_kw
    )
    logger.info(f"Loaded model {model_name}")

    # Set output file
    output_file = output_path + \
        f"/{gold_index_position}_{total_num_instances}instance_{num_docs}docs_{seed}seed_{model_name}.{prompt_type}.jsonl"
    restart_from_ind = 0
    # check if output_file exists
    if os.path.exists(output_file):
        # if output_file exists, try restarting from where it left off
        logger.warning(f"Output path {output_file} already exists, restarting from where it left off")
        # check the number of lines in the output file
        import subprocess
        num_lines = int(subprocess.check_output(f"wc -l {output_file}", shell=True).split()[0])
        logger.warning(f"Number of lines in the output file: {num_lines}")
        # decide where to start from
        # each instance has (num_docs * num_shuffles_per_instance) lines
        restart_from_ind = num_lines // (num_docs * num_shuffles_per_instance)
        logger.warning(f"Restarting from index: {restart_from_ind}")
        output_file = output_path + \
            f"/{gold_index_position}_{total_num_instances}instance_{num_docs}docs_{seed}seed_{model_name}.{prompt_type}.continue.jsonl"
        if os.path.exists(output_file):
            logger.warning(f"Output path {output_file} already exists, exiting")
            sys.exit(1)

    # Load input data
    examples, prompts, all_model_documents = get_input_documents(
        input_path,
        model_name,
        prompt_mention_random_ordering,
        use_random_ordering,
        max_prompt_length,
        restart_from_ind=restart_from_ind,
        num_shuffles_per_instance=num_shuffles_per_instance,
        num_docs=num_docs,
        total_num_instances=total_num_instances,
        seed=seed,
        append_answer=append_answer,
        gold_index_position=gold_index_position,
        prompt_type=prompt_type,
        verbose=verbose
    )

    logger.info(f"Loaded {len(prompts)} prompts to process")

    logger.info(f"Writing results to {output_file}")
    logger.info("Generating responses")

    # prompt_queue_size: number of prompts sent to vllm
    # to generate responses for in one batch
    if gold_index_position == "lost-in-the-middle":
        prompt_queue_size = (num_docs // 5) + 1
    else:
        prompt_queue_size = num_docs

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        prompt_logprobs=0
    )
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with xopen(output_file, "wb") as f:
        # read input data
        for i in trange(0, len(prompts), prompt_queue_size):
            seg_prompts = prompts[i:i + prompt_queue_size]
            try:
                # adding prompt_queue_size prompts to the model
                seg_raw_responses = model.generate([x["prompt"] for x in seg_prompts], sampling_params, use_tqdm=False)
            except Exception as e:
                print(f"Error generating responses: {e}, skipping batch {i}:{i + prompt_queue_size}")
                continue
            seg_responses = [output.outputs[0].text.strip() for output in seg_raw_responses]
            seg_logprobs = [get_logprobs(response, **prompt)
                            for response, prompt in zip(seg_raw_responses, seg_prompts)]

            # Write responses to output file
            for example, model_documents, prompt, response, logprob in zip(
                examples[i:i + prompt_queue_size],
                all_model_documents[i:i + prompt_queue_size],
                seg_prompts, seg_responses, seg_logprobs
            ):
                example["model_prompt"] = prompt["prompt"]
                example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
                example["model_answer"] = response
                example["model"] = model_name
                example["model_temperature"] = temperature
                example["model_top_p"] = top_p
                example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
                example["model_use_random_ordering"] = use_random_ordering

                example.update(logprob)
                example.update(prompt)
                f.write(orjson.dumps(example) + b"\n")

            del seg_raw_responses, seg_responses, seg_logprobs
