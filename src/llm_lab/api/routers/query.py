from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from llm_lab.api.dependencies import get_llm_client, get_retriever_client
from llm_lab.api.exceptions import CustomException
from llm_lab.core.rag_service import RagService
from llm_lab.llm.types import LlmClient
from llm_lab.observability.context import dataset_context_var, top_k_context_var
from llm_lab.retrieval.retriever import Retriever
from llm_lab.vector_store.types import ScoredChunk


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str
    embedding_model: str = Field(description="Embedding model name")
    top_k: int = Field(default=3)
    dataset: str = Field(description="Dataset name")


class SourceChunk(BaseModel):
    source: str
    chunk_id: int


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)
    answer: str
    sources: list[SourceChunk]


router = APIRouter(prefix="", tags=["Query"])


def validate_query_request(request: QueryRequest) -> None:
    if not request.query:
        raise CustomException(
            status_code=400, message="Question must be a non-empty string"
        )
    if request.top_k < 1 or request.top_k > 10:
        raise CustomException(status_code=400, message="top_k must be between 1 and 10")
    if not request.dataset:
        raise CustomException(
            status_code=400, message="Dataset name must be a non-empty string"
        )


def build_response(
    top_chunks: list[ScoredChunk],
    response: str,
) -> QueryResponse:
    return QueryResponse(
        answer=response,
        sources=[
            SourceChunk(
                source=sc.indexed_chunk.source, chunk_id=sc.indexed_chunk.chunk_id
            )
            for sc in top_chunks
        ],
    )


@router.post("/query")
async def query(
    body: QueryRequest,
    llm_client: LlmClient = Depends(get_llm_client),
    retriever: Retriever = Depends(get_retriever_client),
) -> QueryResponse:
    validate_query_request(body)
    dataset_context_var.set(body.dataset)
    top_k_context_var.set(body.top_k)
    rag = RagService(llm_client, retriever)
    try:
        query_result = rag.answer_question(
            dataset=body.dataset,
            embedding_model=body.embedding_model,
            query=body.query,
            top_k=body.top_k,
        )
    except (ValueError, FileNotFoundError) as err:
        raise CustomException(status_code=500, message=str(err)) from err
    return build_response(query_result.chunks, query_result.answer)
