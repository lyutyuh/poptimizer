"""
List of supported language models
"""

# LMs that are tested
SUPPORTED_LM_LIST = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "mosaicml/mpt-7b-8k",
    "mosaicml/mpt-7b-8k-chat",
    "mosaicml/mpt-7b-8k-instruct",
    "mosaicml/mpt-30b-instruct",
    "mosaicml/mpt-30b",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "lmsys/longchat-13b-16k"
]

# LMs that require the BOS token to be added to the prompt
ADD_BOS_LM_LIST = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "lmsys/longchat-13b-16k"
]

# LMs that do not require the BOS token to be added to the prompt
NO_BOS_LM_LIST = [
    "mosaicml/mpt-7b-8k",
    "mosaicml/mpt-7b-8k-instruct",
    "mosaicml/mpt-30b-instruct",
    "mosaicml/mpt-30b",
]
