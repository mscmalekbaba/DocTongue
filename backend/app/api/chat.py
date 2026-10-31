"""Chat API router: POST /api/chat.

Validates the incoming question against guardrails, delegates to the
retrieval pipeline, and returns a grounded answer with citations.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import Settings
from app.core.dependencies import (
    get_answer_generator,
    get_chat_memory_store,
    get_vector_store,
    settings_dependency,
)
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_memory import ChatMemoryStore
from app.services.guardrails import validate_chat_question
from app.services.llm import AnswerGenerator
from app.services.retrieval import answer_question
from app.services.vector_store import VectorStore


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    settings: Settings = Depends(settings_dependency),
    vector_store: VectorStore = Depends(get_vector_store),
    answer_generator: AnswerGenerator = Depends(get_answer_generator),
    chat_memory_store: ChatMemoryStore = Depends(get_chat_memory_store),
) -> ChatResponse:
    """Answer a user question grounded in the indexed document collection.

    Rejects questions that trigger safety guardrails with HTTP 400.
    Returns HTTP 500 if the retrieval pipeline raises an unexpected error.
    """
    violation = validate_chat_question(payload.question)
    if violation:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=violation)

    try:
        return answer_question(
            question=payload.question,
            session_id=payload.session_id,
            settings=settings,
            vector_store=vector_store,
            answer_generator=answer_generator,
            chat_memory_store=chat_memory_store,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to answer question.") from exc