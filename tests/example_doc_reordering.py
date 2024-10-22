import torch

from poptimizer.methods.doc_reordering import DocReorderingGauge
from poptimizer.methods.likelihood_selection import LikelihoodSelectionGauge
from poptimizer.methods.base import RAGInstance
from poptimizer.prompting import (
    Document
)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Initialize the gauge
doc_reorder_gauge = DocReorderingGauge(
    model_name,
    top_p=1,
    num_gpus=1,
    gpu_memory_utilization=0.6,
    max_prompt_length=4096,
    torch_dtype=torch.bfloat16,
    seed=42
)
instance = RAGInstance(
    question="What is the capital of France?",
    documents=[
        Document(text="Paris is the capital of France."), 
        Document(text="The capital of France is Paris."),
        Document(text="France's capital is Paris.")
    ]
)

optimized_instance = doc_reorder_gauge.optimize_instance(instance)

# print(optimized_instance)

response = doc_reorder_gauge.generation_with_instance(optimized_instance)

print(response)