from vllm import LLM, SamplingParams

from poptimizer.methods.base import BaseGauge


class LikelihoodSelectionGauge(BaseGauge):

    def __init__(
        self, model_name,
        top_p=0.9,
        num_gpus=1,
        gpu_memory_utilization=0.6,
        max_prompt_length=4096,
        torch_dtype=torch.bfloat16,
        seed=42
    ):
        """
        Args:
            model: The pretrained language model that will be used to score prompts.

        """
        self.model_name = model_name

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
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=top_p,
            max_tokens=1,
            prompt_logprobs=0
        )


    def shuffle_documents(shuffle_id, instance):
        documents = deepcopy(instance.documents)
        if shuffle_id > 0:
            np.random.shuffle(documents)
        num_docs = len(documents)
        yield documents


    def optimize_instance(
        self,
        instance: RAGInstance
    ):
        """
        Args:
            prompt: The prompt to optimize.
            documents: A list of documents to use in the prompt.
        Returns:
            The optimized prompt.
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
        best_instance = instances[np.argmax(scores)]        

        return best_instance