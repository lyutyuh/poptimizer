import os
import argparse
import dataclasses

from typing import Union, List, Dict, Optional, Any, Tuple
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
from fastchat.model import get_conversation_template
import numpy as np

from poptimizer.prompting import (
    Document,
    verbalize_document,
    get_qa_prompt
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe. "
    "Please ensure that your responses are socially unbiased and positive in nature. "
    "Give the answer directly without any additional or unnecessary information. "
    "If you don't know the answer "
    "to a question, please don't share false information."
)


def is_chat_model(model_name):
    return "chat" in model_name.lower() or\
           "instruct" in model_name.lower()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dict_to_np_array(d):
    return {
        k: np.array(v) for k, v in d.items()
    }


def format_chat_prompt(
    message: str,
    answer: Optional[str] = None,
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    append_answer: bool = False,
    answer_prompt: str = "According to the provided documents, the answer is ",
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
):
    """
    Format the chat prompt for the model.
    Using the default system prompt for the model if the model is not in the list.
    """
    if model_name in ["lmsys/longchat-13b-16k"]:
        # Longchat models use vicuna template
        conv = get_conversation_template("vicuna")
    elif "llama-3.1" in model_name.lower():
        # Llama-3.1 models use llama-3 template
        conv = get_conversation_template("llama-3")
    elif is_chat_model(model_name):
        try:
            conv = get_conversation_template(model_name)
            conv.system_message = default_system_prompt
        except Exception as e:
            logger.error(f"Error in formatting chat prompt: {e}")
            conv = None
    else:
        conv = None
    if conv is not None:
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], answer_prompt + answer if append_answer else None)
        prompt = conv.get_prompt() + ("" if append_answer else (" " + answer_prompt[:-1]))
    else:
        prompt = message + "\n" + (answer_prompt + answer if append_answer else answer_prompt[:-1])

    return prompt


def get_logprobs(
    output,
    question_start,
    question_end,
    doc_token_starts,
    doc_token_ends,
    prompt_length,
    gold_doc_start=None,
    gold_doc_end=None,
    answer_start=None,
    answer_end=None,
    **kwargs
):
    """
    get_logprobs takes the vllm output, and the dictionary of prompt as input.

    Args:
        output: vllm response or vllm response-like object,
                must have a prompt_logprobs attribute, which is a list of 
                objects x, where x is a dict {next_token_id_1: y1, next_token_id_2: y2, ...},
                where yi is a vllm Logprob object, 
                which has attributes: logprob, decoded_token_id, decoded_token, etc.

        prompt_length: length of the prompt
        question_start: start of the question in the prompt
        question_end: end of the question in the prompt
        gold_doc_start: start of the gold document in the prompt
        gold_doc_end: end of the gold document in the prompt
        doc_token_starts: list of start tokens of each document in the prompt

    Optional Args:
        (Not present when called in completion, i.e., append_answer=False)
        answer_start: start of the ground truth answer in the prompt
        answer_end: end of the ground truth answer in the prompt 
    """
    prompt_logprob = 0.
    question_mid_prompt_logprob = 0.
    answer_mid_prompt_question_logprob = 0.

    token_logprobs, doc_logprobs = [], []

    # Loop through the logprobs of all tokens in the prompt (and generated tokens)
    for ix, x in enumerate(output.prompt_logprobs):
        if x is None:
            continue

        assert len(x.values()) == 1
        token_logprob = list(x.values())[0].logprob
        logprob_dict = list(x.values())[0].__dict__
        if len(logprob_dict["decoded_token"]) > 10:
            logprob_dict["decoded_token"] = logprob_dict["decoded_token"][-10:]
        logprob_dict["decoded_token_id"] = list(x.keys())[0]
        token_logprobs.append(logprob_dict)

        # question_start <= ix < question_end, add to question_mid_prompt_logprob
        if question_start <= ix < question_end:
            question_mid_prompt_logprob += token_logprob
        # answer_start <= ix < answer_end, add to answer_mid_prompt_question_logprob
        elif (answer_start is not None) and (answer_start <= ix < answer_end):
            answer_mid_prompt_question_logprob += token_logprob

        if ix < prompt_length:
            prompt_logprob += token_logprob

    for ix, (doc_start, doc_end) in enumerate(zip(doc_token_starts, doc_token_ends)):
        doc_logprobs.append(
            sum(
                [x["logprob"] for x in token_logprobs[doc_start:doc_end]]
            )
        )

    return {
        "prompt_logprob": prompt_logprob,
        "question_mid_prompt_logprob": question_mid_prompt_logprob,
        "answer_mid_prompt_question_logprob": answer_mid_prompt_question_logprob,
        "doc_logprobs": doc_logprobs,
        "token_logprobs": token_logprobs
    }


def get_char_to_token(encoding, raw_string, model_name=None):
    """
    We need this function because Llama-3's char_to_token method does not work
    as expected (as of transformers 4.43.1, tokenizers 0.19.1). 
    It's offset_mapping only contains the char_start of each token and 
    looks like [(0, 0), (0, 0), (1, 1), (6, 6)]. 
    So we construct a char_to_token dict manually only using char_start from
    offset_mapping for all models.
    """

    last_token_index = len(encoding["input_ids"]) - 1
    last_char_index = len(raw_string) - 1

    char_to_token = {}
    prev_start_char = 0
    for i, (start, end) in enumerate(encoding["offset_mapping"][1:]):
        for j in range(prev_start_char, start):
            char_to_token[j] = i
        prev_start_char = start
    for j in range(prev_start_char, len(raw_string)):
        char_to_token[j] = i+1
    char_to_token[len(raw_string)] = i+2
    return char_to_token


def shuffle_one_instance(
    input_example,
    tokenizer,
    num_docs=20,
    num_shuffles=1,
    max_prompt_length=4096,
    prompt_mention_random_ordering=False,
    model_name=None,
    did_format_warn=False,
    use_random_ordering=True,
    append_answer=True,
    gold_index_position="lost-in-the-middle",
    verbose=False,
    prompt_type="ICQ"
):
    """
    We call a question and a set of documents associated with it an "instance".
    This function shuffles the documents in an input instance and
    generates a list of prompts.
    """
    question = input_example["question"]
    original_documents = []

    for ctx in input_example["ctxs"]:
        one_document = Document.from_dict(ctx)
        original_documents.append(one_document)
    if not original_documents:
        raise ValueError(
            f"Did not find any documents for example: {input_example}")

    all_documents, all_prompts = [], []

    (original_gold_index,) = [idx for idx, doc in enumerate(
        original_documents) if doc.isgold is True]
    original_gold_document = original_documents[original_gold_index]
    original_distractors = [
        doc for doc in original_documents if doc.isgold is False]

    if gold_index_position == "lost-in-the-middle":
        # gold_index in [0, 4, 4+5, 4+2*5, ...]
        gold_indices = [0,] + [4 + i*5 for i in range(0, num_docs//5)]

        def get_documents(shuffle_id):
            distractors = deepcopy(original_distractors)
            if shuffle_id > 0:
                np.random.shuffle(distractors)
            for gold_index in gold_indices:
                documents = deepcopy(distractors)
                documents.insert(gold_index, original_gold_document)

                yield documents, gold_index

    elif gold_index_position == "every":
        gold_indices = list(range(num_docs))

        def get_documents(shuffle_id):
            distractors = deepcopy(original_distractors)
            if shuffle_id > 0:
                np.random.shuffle(distractors)
            for gold_index in gold_indices:
                documents = deepcopy(distractors)
                documents.insert(gold_index, original_gold_document)

                yield documents, gold_index

    elif gold_index_position == "rotate":
        gold_indices = list(range(num_docs))

        def get_documents(shuffle_id):
            distractors = deepcopy(original_distractors)
            if shuffle_id > 0:
                np.random.shuffle(distractors)
            for gold_index in gold_indices:
                documents = deepcopy([original_gold_document] + distractors)
                documents = documents[num_docs-gold_index:] + \
                    documents[:num_docs-gold_index]

                yield documents, gold_index
    elif gold_index_position == "one-doc-at-a-time":
        # <Instruction> <Document> <Question>
        def get_documents(shuffle_id):
            distractors = [original_gold_document] + original_distractors
            for gold_index in range(num_docs):
                yield distractors[gold_index:gold_index+1], None

    for i in range(num_shuffles):
        for documents, gold_index in get_documents(i):
            # Concatenate all documents into a single string and
            # compute the position of each component in the prompt
            input_data = get_input_data(
                input_example=input_example,
                gold_index=gold_index,
                original_gold_document=original_gold_document,
                documents=documents,
                tokenizer=tokenizer,
                model_name=model_name,
                max_prompt_length=max_prompt_length,
                prompt_type=prompt_type,
                append_answer=append_answer,
                verbose=verbose
            )
            if input_data is None:
                continue

            all_documents.append(documents)
            all_prompts.append(input_data)

    return all_documents, all_prompts


def get_input_data(
    input_example: Dict,
    documents: List[Document],
    tokenizer: AutoTokenizer,
    model_name: str,
    max_prompt_length: int,
    original_gold_document: Optional[Document] = None,
    gold_index: Optional[int] = None,
    prompt_type: str = "ICQ",
    append_answer: bool = True,
    verbose: bool = False,
    answer_prompt: str = "According to the provided documents, the answer is "
):
    """
    Given an input example, a gold index, and a list of documents,
    this function constructs a prompt and returns a dictionary of
    input data for the model.
    """
    question = input_example["question"]
    answer_string = input_example["answers"][0]

    prompt, concatenated_documents = get_qa_prompt(
        question, documents, prompt_type=prompt_type)

    prompt = format_chat_prompt(
        prompt,
        answer=answer_string,
        model_name=model_name,
        append_answer=append_answer
    )
    prompt_question_answer_length = len(tokenizer(prompt)["input_ids"])

    tokenized_prompt = tokenizer(
        prompt, return_length=True, return_offsets_mapping=True)
    char_to_token = get_char_to_token(
        tokenized_prompt, prompt, model_name=model_name)

    if gold_index is not None:
        gold_doc_char_start = prompt.find(
            verbalize_document(original_gold_document))
        gold_doc_char_end = gold_doc_char_start + \
            len(verbalize_document(original_gold_document))
    else:
        gold_doc_char_start = None
        gold_doc_char_end = None

    doc_char_starts = [prompt.find(verbalize_document(doc))
                       for i, doc in enumerate(documents)]
    doc_char_ends = [len(verbalize_document(doc))-1 +
                     doc_char_starts[i] for i, doc in enumerate(documents)]

    assert len(doc_char_starts) == len(doc_char_ends) == len(
        documents), f"{doc_char_starts}, {doc_char_ends}, {gold_index}, {gold_doc_char_start}, {gold_doc_char_end}"
    assert gold_doc_char_start == doc_char_starts[
        gold_index], f"{doc_char_starts}, {doc_char_ends}, {gold_index}, {gold_doc_char_start}, {gold_doc_char_end}"
    assert gold_doc_char_end == doc_char_ends[
        gold_index], f"{doc_char_starts}, {doc_char_ends}, {gold_index}, {gold_doc_char_start}, {gold_doc_char_end}"

    doc_token_starts = [char_to_token[start] for start in doc_char_starts]
    doc_token_ends = [char_to_token[end] for end in doc_char_ends]

    first_doc_start = doc_token_starts[0]
    last_doc_end = doc_token_ends[-1]

    if gold_index is not None:
        gold_doc_start = char_to_token[gold_doc_char_start]
        gold_doc_end = char_to_token[gold_doc_char_end]
    else:
        gold_doc_start = None
        gold_doc_end = None

    docs_end = char_to_token[prompt.rfind(concatenated_documents)+len(concatenated_documents)]
    prompt_length = char_to_token[prompt.rfind("Question: ")+len("Question: ")]
    question_start = char_to_token[prompt.rfind(question)]
    question_end = char_to_token[prompt.rfind(question)+len(question)]
    prompt_question_length = char_to_token[prompt.rfind(
        answer_prompt) + len(answer_prompt)]

    if verbose:
        print(question, [tokenizer.decode(tokenized_prompt["input_ids"][i])
              for i in range(question_start, question_end)])
        for i, doc in enumerate(documents):
            print(f"Doc{i+1}")
            print([tokenizer.decode(tokenized_prompt["input_ids"][j])
                  for j in range(doc_token_starts[i], doc_token_ends[i]+2)])
        print([tokenizer.decode(tokenized_prompt["input_ids"][i])
              for i in range(first_doc_start, first_doc_start+10)])
        print([tokenizer.decode(tokenized_prompt["input_ids"][i])
              for i in range(last_doc_end-10, last_doc_end)])
        print(prompt)

    if max_prompt_length < prompt_question_answer_length:
        logger.info(
            f"Skipping prompt {prompt[:100]}... with length {prompt_question_answer_length}, which "
            f"is greater than maximum prompt length {max_prompt_length}"
        )
        return None

    input_data = {
        "prompt": prompt,
        "question_start": question_start,
        "question_end": question_end,
        "docs_end": docs_end,
        "prompt_length": prompt_length,
        "prompt_question_length": prompt_question_length,
        "prompt_question_answer_length": prompt_question_answer_length,
        "first_doc_start": first_doc_start,
        "doc_token_starts": doc_token_starts,
        "doc_token_ends": doc_token_ends,
        "gold_doc_start": gold_doc_start,
        "gold_doc_end": gold_doc_end,
        "answer_location": int(gold_index)
    }
    if append_answer:
        input_data["answer_start"] = char_to_token[prompt.rfind(answer_string)]
        input_data["answer_end"] = char_to_token[prompt.rfind(
            answer_string)+len(answer_string)]

        if verbose:
            logger.info([tokenizer.decode(tokenized_prompt["input_ids"][i]) for i in range(
                input_data["answer_start"], input_data["answer_end"])])

    return input_data


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]
