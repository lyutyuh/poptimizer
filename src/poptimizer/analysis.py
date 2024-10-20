import argparse
import orjson as json
import logging
import statistics
import sys
from copy import deepcopy

from tqdm import tqdm
from transformers import AutoTokenizer
from xopen import xopen
import re
import os
import multiprocessing as mp
from functools import partial

from lost_in_the_middle.metrics import best_subspan_em
from lost_in_the_middle.util import get_char_to_token

import scipy
import numpy as np
import pandas as pd

METRICS = [
    # Exact substring match
    (best_subspan_em, "best_subspan_em"),
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_metrics_for_example(example):
    gold_answers = example["answers"]
    model_answer = example["model_answer"]

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)


def data_to_np_arrays(all_examples):

    questions = [example["question"] for example in all_examples]
    docs = [example["model_documents"] for example in all_examples]

    prompt_logprobs = np.array([example["prompt_logprob"] for example in all_examples])
    question_mid_prompt_logprob = np.array([example["question_mid_prompt_logprob"] for example in all_examples])
    question_lengths = np.array([example["question_end"]-example["question_start"] for example in all_examples])
    mean_question_logprob = question_mid_prompt_logprob / question_lengths

    answer_mid_prompt_question_logprob = np.array(
        [example["answer_mid_prompt_question_logprob"] for example in all_examples])
    if "answer_end" in all_examples[0]:
        answer_lengths = np.array([example["answer_end"]-example["answer_start"] for example in all_examples])
    else:
        answer_lengths = None

    answer_location = np.array([example["answer_location"] for example in all_examples])

    all_example_metrics = [get_metrics_for_example(example) for example in all_examples]
    accuracy = np.array([x[0]["best_subspan_em"] for x in all_example_metrics])

    gold_logprobs, doc_logprobs = [], []
    for example in all_examples:
        eg_doc_logprobs = []

        for ix, (doc_start, doc_end) in enumerate(zip(example["doc_token_starts"], example["doc_token_ends"])):
            eg_doc_logprobs.append(
                sum(
                    [x["logprob"] for x in example["token_logprobs"][doc_start:doc_end]]
                )
            )

        doc_logprobs.append(eg_doc_logprobs)
        gold_logprobs.append(eg_doc_logprobs[example["answer_location"]])

    doc_logprobs = np.array(doc_logprobs)
    gold_logprobs = np.array(gold_logprobs)

    token_logprobs = np.array([0 for example in all_examples])

    np_data = {
        "questions": questions,
        "docs": docs,
        "Prompt logprob": prompt_logprobs,
        "Ground Truth Doc logprob": np.squeeze(np.take_along_axis(doc_logprobs, np.expand_dims(answer_location, 1), axis=1)),
        "Question logprob": question_mid_prompt_logprob,
        "Question length": question_lengths,
        "Mean Question logprob": mean_question_logprob,
        "Answer logprob": answer_mid_prompt_question_logprob,
        "Answer length": answer_lengths,
        "Answer Location": answer_location,
        "Doc logprobs": doc_logprobs,
        "Token logprobs": token_logprobs,
        "All Doc logprob": doc_logprobs[:,:].sum(axis=1),
        "Gold Doc logprob": gold_logprobs,
        "Accuracy": accuracy
    }

    return np_data


def logprob_analysis(all_examples, distance=100, make_plot=False):
    import seaborn as sns
    import matplotlib.pyplot as plt

    np_data = data_to_np_arrays(all_examples)
    sorted_values = []
    # change color for every <distance> points
    n_examples = (len(all_examples) // distance)
    for i in range(n_examples):
        resamples = np.array(range(i*distance, (i+1)*distance))
        data = pd.DataFrame({
            "Prompt logprob": np_data["Prompt logprob"][resamples],
            "Ground Truth Doc logprob": np_data["Ground Truth Doc logprob"][resamples],
            "Question logprob": np_data["Question logprob"][resamples],
            "Answer logprob": np_data["Answer logprob"][resamples],
            "Answer Location": [int(x) for x in np_data["Answer Location"][resamples]],
            "Mean Doc logprob": np_data["Mean Doc logprob"][resamples],
            "Gold Doc logprob": np_data["Gold Doc logprob"][resamples],
            "Token logprobs": np_data["Token logprobs"][resamples]
        })

        color_palette = sns.color_palette("magma", as_cmap=True)
        this_color = color_palette(i / (n_examples))

        x_key = "Answer logprob"
        y_key = "Question logprob"

        # sort data[y_key] by data[x_key]
        x_value = np_data[y_key][resamples]
        sorted_x_value = x_value[np.argsort(np_data[x_key][resamples])]
        sorted_values.append(sorted_x_value)

        if make_plot:
            mask_1 = (data["Answer Location"] != -1)
            sns.regplot(x=data[x_key][mask_1], y=data[y_key]
                        [mask_1], scatter_kws={'s': 2}, marker="v", color=this_color)

            mask_2 = (data["Answer Location"] == -1)
            sns.regplot(x=data[x_key][mask_2], y=data[y_key]
                        [mask_2], scatter_kws={'s': 2}, marker="x", color=this_color)

    if make_plot:
        plt.xlabel(x_key)
        plt.ylabel(y_key)
        # save plot as pdf
        plt.savefig("logprob_analysis.pdf", dpi=1000)

    return np_data


def process_line(line, tokenizer, model_name):
    input_example = json.loads(line)

    if "doc_token_starts" not in input_example or "doc_token_ends" not in input_example:
        prompt = input_example["model_prompt"]
        documents = input_example["model_documents"]
        tokenized_prompt = tokenizer(prompt, return_length=True, return_offsets_mapping=True)
        char_to_token = get_char_to_token(tokenized_prompt, prompt, model_name=model_name)
    else:
        return input_example

    if "doc_token_starts" not in input_example:
        doc_char_starts = [prompt.find(f"Document [{i+1}]") for i, doc in enumerate(documents)]
        doc_token_starts = [char_to_token[start] for start in doc_char_starts]
        input_example["doc_token_starts"] = doc_token_starts

    if "doc_token_ends" not in input_example:
        doc_char_ends = [len(f"Document [{i+1}](Title: {doc['title']}) {doc['text']}")-1 + doc_char_starts[i] for i, doc in enumerate(documents)]
        doc_token_ends = [char_to_token[end] for end in doc_char_ends]
        input_example["doc_token_ends"] = doc_token_ends

    return input_example


def compute_by_answer_location(
    input_path, 
    model_name="",
    max_doc_num=30
):
    model_name_to_identifier = {
        "meta-llama": "meta-llama",
        "llama": "meta-llama",
        "mpt": "mosaicml",
        "mistral": "mistralai",
        "longchat": "lmsys"
    }

    identifier = model_name
    for k in model_name_to_identifier:
        if model_name.lower().startswith(k):
            identifier = model_name_to_identifier[k] + "/" + model_name
            break

    tokenizer = AutoTokenizer.from_pretrained(identifier)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    all_examples = []

    if os.path.exists(input_path+".npy"):
        np_data = np.load(input_path+".npy", allow_pickle=True)[()]
    else:
        with xopen(input_path) as fin:
            all_examples = [process_line(line, tokenizer, identifier) for line in tqdm(fin.readlines())]
            np_data = data_to_np_arrays(all_examples)
            np.save(open(input_path+".npy", "wb"), np_data)

    mean_by_answer_location = {}
    for i in range(max_doc_num+1):
        mask = (np_data["Answer Location"] == i)
        if mask.sum() != 0:
            mean_by_answer_location[i] = {
                "Model name": model_name,
                "Answer logprob": np.mean(np_data["Answer logprob"][mask]),
                "Question logprob": np.mean(np_data["Question logprob"][mask]),
                "Mean Question logprob": np.mean(np_data["Question logprob"][mask] / np_data["Question length"][mask]),
                "Prompt logprob": np.mean(np_data["Prompt logprob"][mask]),
                "All Doc logprob": np.mean(np_data["All Doc logprob"][mask]),
                "Mean Doc logprob": np.mean(np_data["Doc logprobs"][:, i]),
                "Gold Doc logprob": np.mean(np_data["Gold Doc logprob"][mask]),
                "Accuracy": np.mean(np_data["Accuracy"][mask]),
                "High Acc Question logprob": np.mean(np_data["Question logprob"][mask][np_data["Accuracy"][mask] >= 0.5]),
                "Low Acc Question logprob": np.mean(np_data["Question logprob"][mask][np_data["Accuracy"][mask] < 0.5]),
            }
            if np_data["Answer length"] is not None:
                mean_by_answer_location[i]["Answer length"] = np.mean(np_data["Answer length"][mask])

    return mean_by_answer_location



if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        help="Path to data with questions and documents to use.",
        required=True,
    )
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
    )
    args = parser.parse_args()

    input_path = args.input_path
    model_name = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # avoiding error

    with xopen(input_path) as fin:
        all_examples = [process_line(line, tokenizer, model_name) for line in tqdm(fin.readlines())]
        np_data = data_to_np_arrays(all_examples)
        np.save(open(input_path+".npy", "wb"), np_data)
