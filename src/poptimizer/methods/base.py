from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List, Tuple

from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
from fastchat.model import get_conversation_template, load_model
from poptimizer.prompting import (
    Document,
    get_qa_prompt,
    verbalize_document,
)
from poptimizer.util import (
    format_chat_prompt,
    get_logprobs,
    get_char_to_token
)


class RAGInstance(ABC):

    def __init__(
        self,
        question: str,
        documents: List[Document]
    ):
        self.question = question
        self.documents = documents

    def __str__(self):
        return f"Question: {self.question}\nDocuments: {self.documents}"

    def get_prompt(self) -> Tuple[str, Tuple[int, int]]:
        """
        Return a prompt constructed from the question and documents.
        """
        prompt, concatenated_documents = get_qa_prompt(
            self.question, self.documents)
        return prompt


def get_instance_metadata(
    instance: RAGInstance,
    tokenizer: AutoTokenizer,
    model_name: str,
    prompt_type: str = "ICQ"
):
    """
    Given an input example, a gold index, and a list of documents,
    this function constructs a prompt and returns a dictionary of
    input data for the model.
    """
    question = instance.question
    documents = instance.documents

    prompt, concatenated_documents = get_qa_prompt(
        question, documents, prompt_type=prompt_type)

    prompt = format_chat_prompt(
        prompt,
        model_name=model_name
    )
    total_length = len(tokenizer(prompt)["input_ids"])

    tokenized_prompt = tokenizer(
        prompt, return_length=True, return_offsets_mapping=True)
    char_to_token = get_char_to_token(
        tokenized_prompt, prompt, model_name=model_name)

    doc_char_starts = [prompt.find(verbalize_document(i, doc))
                       for i, doc in enumerate(documents)]
    doc_char_ends = [len(verbalize_document(i, doc))-1 +
                     doc_char_starts[i] for i, doc in enumerate(documents)]

    doc_token_starts = [char_to_token[start] for start in doc_char_starts]
    doc_token_ends = [char_to_token[end] for end in doc_char_ends]

    first_doc_start = doc_token_starts[0]
    last_doc_end = doc_token_ends[-1]

    docs_end = char_to_token[prompt.rfind(
        concatenated_documents)+len(concatenated_documents)]
    question_start = char_to_token[prompt.rfind(question)]
    question_end = char_to_token[prompt.rfind(question)+len(question)]

    instance_metadata = {
        "prompt": prompt,
        "question_start": question_start,
        "question_end": question_end,
        "docs_end": docs_end,
        "total_length": total_length,
        "first_doc_start": first_doc_start,
        "doc_token_starts": doc_token_starts,
        "doc_token_ends": doc_token_ends,
        "prompt_length": total_length,
    }

    return instance_metadata


class BaseGauge(ABC):
    def __init__(self, model_name, top_p, max_new_tokens):
        """
        Args:
            model: The pretrained language model that will be used to score prompts.

        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            prompt_logprobs=0
        )

        self.generation_params = SamplingParams(
            temperature=1.0,
            top_p=top_p,
            max_tokens=max_new_tokens,
            prompt_logprobs=0
        )


    def logp_instances(self, instances):
        """
        Compute the log probabilities given the documents.
        Args:
            instances: A list of instances to score.
        Returns:
            A list of scores for each prompt.
        """
        prompts = [
            get_instance_metadata(instance, self.tokenizer, self.model_name)
            for instance in instances
        ]

        raw_responses = self.model.generate(
            [prompt["prompt"] for prompt in prompts], self.sampling_params, use_tqdm=False
        )
        responses = [output.outputs[0].text.strip()
                     for output in raw_responses]
        logprobs: List[Dict[str, float]] = [get_logprobs(response, **prompt)
                                            for response, prompt in zip(raw_responses, prompts)]

        # logprobs contains
        # prompt_logprob, question_mid_prompt_logprob
        # answer_mid_prompt_question_logprob, doc_logprobs, token_logprobs

        return logprobs

    def logp_question_instances(self, instances):
        """
        Compute the log probability of the question given the documents.
        Args:
            instances: A list of instances to score.
        Returns:
            A list of scores for each prompt.
        """
        logprobs = self.logp_instances(instances)
        logp_questions = [logprob["answer_mid_prompt_question_logprob"]
                          for logprob in logprobs]

        return logp_questions

    def generation_with_instance(
        self,
        instance,
        prompt_type="ICQ"
    ):
        prompt, concatenated_documents = get_qa_prompt(
            instance.question, instance.documents, prompt_type=prompt_type)
        prompt = format_chat_prompt(
            prompt, answer=None, model_name=self.model_name, append_answer=False
        )
        print(prompt)
        raw_responses = self.model.generate(
            [prompt], self.generation_params, use_tqdm=False
        )

        return raw_responses

    @abstractmethod
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
        pass
