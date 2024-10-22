"""
Adapted from https://github.com/nelson-liu/lost-in-the-middle
"""
import pathlib
from copy import deepcopy
from typing import List, Optional, Tuple, Type, TypeVar

from pydantic.dataclasses import dataclass

PROMPTS_ROOT = (pathlib.Path(__file__).parent / "prompts").resolve()

T = TypeVar("T")

@dataclass(frozen=True)
class Document:
    text: str
    title: Optional[str] = None
    id: Optional[str] = None
    score: Optional[float] = None
    hasanswer: Optional[bool] = None
    isgold: Optional[bool] = None
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        return cls(**dict(data, id=id, score=score))


def verbalize_document(document_index:int, document: Document):
    """
    Verbalize the document into a string.
    """
    if document.title is None or document.title == "":
        return f"Document [{document_index+1}] {document.text}"
    else:
        return f"Document [{document_index+1}](Title: {document.title}) {document.text}"
        


def get_qa_prompt(
    question: str, 
    documents: List[Document],
    prompt_type: str = "ICQ"
):
    """
    Get the prompt to be feed into the model for a question answering task.
    The prompt will then be formatted with model-specific templates.
    """

    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not documents:
        raise ValueError(f"Provided `documents` must be truthy, got: {documents}")

    prompt_filenames = {
        "ICQ": "qa_ICQ.prompt",
        "IQC": "qa_IQC.prompt",
        "IQCQ": "qa_IQCQ.prompt"
    }
    if prompt_type in prompt_filenames:
        prompt_filename = prompt_filenames[prompt_type]
    else:
        raise ValueError(f"Invalid `prompt_type`: {prompt_type}")

    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")

    # Format the documents into strings
    formatted_documents = []
    for document_index, document in enumerate(documents):
        formatted_documents.append(verbalize_document(document_index, document))
    concatenated_documents = "\n".join(formatted_documents)
    prompt = prompt_template.format(question=question, search_results=concatenated_documents)
    return (prompt, concatenated_documents)
