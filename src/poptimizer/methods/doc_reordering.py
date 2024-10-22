from typing import List, Dict
from vllm import LLM, SamplingParams
from collections import defaultdict

import torch
from poptimizer.methods.base import BaseGauge, RAGInstance
from poptimizer.util import get_qa_prompt

import logging
from copy import deepcopy
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def init_vllm(
    model_name,
    num_gpus=1,
    gpu_memory_utilization=0.6,
    max_prompt_length=4096,
    torch_dtype=torch.bfloat16,
    seed=42
):
    extra_kw = {}
    if model_name.startswith("meta-llama/Meta-Llama-3.1"):
        # Currently, Llama-3.1 needs extra parameters to work properly with VLLM
        # as of VLLM 0.5.1
        extra_kw["rope_scaling"] = {
            "type": "yarn",
            "factor": 8.0,
            'original_max_position_embeddings': 8192
        }

    logger.info(
        f"Initializing model {model_name} with {num_gpus} GPUs with vllm"
    )
    # Initialize vllm model
    model = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch_dtype,
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
    return model


class DocReorderingGauge(BaseGauge):

    def __init__(
        self, model_name,
        top_p=1,
        num_gpus=1,
        gpu_memory_utilization=0.6,
        max_prompt_length=4096,
        max_new_tokens=100,
        torch_dtype=torch.bfloat16,
        seed=42
    ):
        """
        Args:
            model: The pretrained language model that will be used to score prompts.

        """
        super().__init__(
            model_name=model_name,
            top_p=top_p,
            max_new_tokens=max_new_tokens
        )

        self.num_gpus = num_gpus
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_prompt_length = max_prompt_length
        self.torch_dtype = torch_dtype
        self.seed = seed

        self.model = init_vllm(
            model_name,
            num_gpus=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_prompt_length=max_prompt_length,
            torch_dtype=torch_dtype,
            seed=seed
        )

    def shuffle_documents(self, shuffle_id, instance):
        documents = deepcopy(instance.documents)
        if shuffle_id > 0:
            np.random.shuffle(documents)
        num_docs = len(documents)
        for start in range(num_docs):
            documents = instance.documents[num_docs-start:] + \
                instance.documents[:num_docs-start]

            yield documents

    def optimize_instance(
        self,
        instance: RAGInstance
    ):
        """
        Args:
            instance: The instance to optimize.
                with two fields: question and documents.
                question: The question to ask.
                documents: A list of documents to use.
        Returns:
            The optimized prompt with the documents reordered.
        """
        num_docs = len(instance.documents)
        num_shuffles = num_docs

        instances: List[RAGInstance] = []
        for shuffle_id in range(num_shuffles):
            for documents in self.shuffle_documents(shuffle_id, instance):
                new_instance = RAGInstance(
                    question=instance.question,
                    documents=documents
                )
                instances.append(new_instance)
        scores = self.logp_question_instances(instances)

        doc_gold_scores: Dict[str, float] = defaultdict(float)
        doc_counts: Dict[str, int] = defaultdict(int)
        for i_instance, instance in enumerate(instances):
            for i_doc, doc in enumerate(instance.documents):
                if i_doc == 0 or i_doc == num_docs - 1:
                    doc_gold_scores[doc] += scores[i_instance]
                    doc_counts[doc] += 1
                else:
                    doc_gold_scores[doc] -= scores[i_instance]
                    doc_counts[doc] += 1

        doc_scores = {doc: doc_gold_scores[doc] /
                      doc_counts[doc] for doc in doc_gold_scores}
        # documents with the higher scores are more likely 
        # to be the gold document, # and we reorder 
        # the documents accordingly, putting 
        # the most likely gold document first
        sorted_docs = sorted(instance.documents,
                             key=lambda x: doc_scores[x], reverse=True)

        return RAGInstance(
            question=instance.question,
            documents=sorted_docs
        )
