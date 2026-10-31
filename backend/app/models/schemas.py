"""Pydantic request/response schemas shared across the API.

These models are used both for HTTP serialisation (FastAPI route I/O)
and for the internal document manifest (via :class:`DocumentSummary`).
"""

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness check response."""

    status: str = Field(description="Always 'ok' when the service is healthy.")


class DocumentSummary(BaseModel):
    """Metadata record for a single indexed document."""

    id: str = Field(description="UUID assigned to the document at upload time.")
    filename: str = Field(description="Original filename provided by the client.")
    content_type: str = Field(description="MIME type derived from the file extension (e.g. 'application/pdf').")
    page_count: int = Field(description="Number of pages detected by the PDF parser.")
    chunk_count: int = Field(description="Number of text chunks indexed in the vector store.")
    created_at: datetime = Field(description="UTC timestamp of when the document was indexed.")


class DocumentListResponse(BaseModel):
    """Response body for GET /api/documents."""

    documents: list[DocumentSummary] = Field(description="All currently indexed documents, ordered by insertion.")


class DocumentUploadResponse(BaseModel):
    """Response body for POST /api/documents."""

    document: DocumentSummary = Field(description="Metadata of the newly indexed document.")


class DeleteDocumentResponse(BaseModel):
    """Response body for DELETE /api/documents/{document_id}."""

    deleted: bool = Field(description="True when the document was found and removed.")
    document_id: str = Field(description="ID of the document that was deleted.")


class Citation(BaseModel):
    """A single source excerpt used to ground an answer."""

    document_id: str = Field(description="UUID of the source document.")
    filename: str = Field(description="Original filename of the source document.")
    excerpt: str = Field(description="Raw text chunk from the document used as evidence.")
    page_number: int | None = Field(default=None, description="Page the excerpt was taken from, or null if unavailable.")
    score: float = Field(ge=0.0, le=1.0, description="Cosine similarity score between the query and this excerpt (0–1).")


class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""

    question: str = Field(
        min_length=3,
        max_length=4000,
        description="The user's question. Must be between 3 and 4000 characters.",
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        description="Stable identifier for the chat session. Used to maintain multi-turn memory. Defaults to 'default'.",
    )


class ChatResponse(BaseModel):
    """Response body for POST /api/chat."""

    answer: str = Field(description="Generated answer text, which may contain [N] citation markers.")
    citations: list[Citation] = Field(description="Document excerpts used to ground the answer.")
    grounded: bool = Field(description="True when the answer is supported by at least one retrieved excerpt.")
