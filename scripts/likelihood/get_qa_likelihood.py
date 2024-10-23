#!/usr/bin/env python3
"""
Given a data file with questions and retrieval results to use, run VLLM to get the likelihood of 
each part in the input prompt.
"""
import argparse
import dataclasses
import json
import logging
import pathlib
import random
import sys
from copy import deepcopy

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from xopen import xopen

import numpy as np

from poptimizer.runner import (
    runner
)
from poptimizer.consts import (
    SUPPORTED_LM_LIST
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        help="Path to data with questions and documents to use.",
        required=True,
        choices=[
            "qa_data/10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl.gz",
            "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz",
            "qa_data/30_total_documents/nq-open-30_total_documents_gold_at_0.jsonl.gz"
        ]
    )
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
        choices=SUPPORTED_LM_LIST,
    )
    parser.add_argument(
        "--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument(
        "--top-p", help="Top-p to use in generation", type=float, default=1.0)

    parser.add_argument(
        "--prompt-mention-random-ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    parser.add_argument(
        "--use-random-ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument("--hf-cache-path",
                        help="Path to huggingface cache to use.")
    parser.add_argument(
        "--output-path", help="Path to write output file of generated responses", required=True)
    parser.add_argument("--num-shuffles-per-instance",
                        help="Number of shuffles to do per instance", type=int, default=1)
    parser.add_argument("--num-docs",
                        help="Total number of docs in an RAG input", type=int, default=20)
    parser.add_argument("--total-num-instances",
                        help="Total number of instances to process", type=int, default=100)
    parser.add_argument("--seed", help="Random seed", type=int, default=1000)

    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-prompt-length",
        help="Maximum number of tokens in the prompt. Longer prompts will be skipped.",
        type=int,
        default=3800,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        help="GPU memory utilization for vllm, a float ranging from 0. to 1.0.",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--gold-index-position",
        help="Gold index position in the prompt. lost-in-the-middle: [0, 4, 9, 14, ...]. every: [0, 1, 2, 3, ...]",
        type=str,
        default="lost-in-the-middle",
        choices=["lost-in-the-middle", "every", "rotate"]
    )
    parser.add_argument(
        "--prompt-type",
        help="Prompt type to use in generating responses. ICQ: instruction, context, question. IQC: instruction, question, context. IQCQ: instruction, question, context, question.",
        type=str,
        default="ICQ",
        choices=["ICQ", "IQC", "IQCQ"]
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))

    runner(
        input_path=args.input_path,
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_mention_random_ordering=args.prompt_mention_random_ordering,
        use_random_ordering=args.use_random_ordering,
        num_gpus=args.num_gpus,
        max_new_tokens=1,
        max_prompt_length=args.max_prompt_length,
        hf_cache_path=args.hf_cache_path,
        output_path=args.output_path,
        num_shuffles_per_instance=args.num_shuffles_per_instance,
        num_docs=args.num_docs,
        total_num_instances=args.total_num_instances,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        append_answer=True,
        gold_index_position=args.gold_index_position,
        prompt_type=args.prompt_type
    )
    logger.info("finished running %s", sys.argv[0])
