"""Document ingestion pipeline: upload validation, storage, PDF parsing, and chunking."""

from dataclasses import dataclass
from pathlib import Path
import uuid

from fastapi import UploadFile
from pypdf import PdfReader

from app.core.config import Settings
from app.services.chunking import PageContent, TextChunk, split_pages_into_chunks


SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class DocumentProcessingError(Exception):
    """Raised when an uploaded document cannot be validated, stored, or parsed."""


@dataclass(slots=True)
class ProcessedDocument:
    """Result of a successful document upload and processing pass.

    Attributes:
        document_id: UUID assigned to this document.
        filename: Original filename provided by the client.
        content_type: MIME type derived from the file extension.
        stored_path: Absolute path where the raw file was written on disk.
        page_count: Number of pages detected by the PDF parser.
        chunks: Text chunks ready to be indexed in the vector store.
    """

    document_id: str
    filename: str
    content_type: str
    stored_path: Path
    page_count: int
    chunks: list[TextChunk]


async def persist_and_process_upload(
    file: UploadFile,
    settings: Settings,
) -> ProcessedDocument:
    """Validate, save, and chunk an uploaded file.

    Validates the file extension and size, writes the raw bytes to
    ``settings.uploads_dir``, extracts text with PyPDF, and runs the
    chunking pass.  Cleans up the stored file if any step fails.

    Args:
        file: The multipart file received from the HTTP request.
        settings: Application settings used for storage paths and chunking params.

    Returns:
        A :class:`ProcessedDocument` ready to register in the document store
        and vector store.

    Raises:
        DocumentProcessingError: If the file is missing, empty, too large,
            an unsupported type, unparseable, or yields no extractable text.
    """
    filename = (file.filename or "").strip()
    if not filename:
        raise DocumentProcessingError("Uploaded files must include a filename.")

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentProcessingError("Unsupported file type. Upload a PDF file.")

    payload = await file.read()
    if not payload:
        raise DocumentProcessingError("The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise DocumentProcessingError("File too large. Maximum size is 50 MB.")

    document_id = str(uuid.uuid4())
    stored_name = f"{document_id}{extension}"
    stored_path = settings.uploads_dir / stored_name
    stored_path.write_bytes(payload)

    try:
        pages = _extract_pages(stored_path, extension)
    except Exception as exc:
        if stored_path.exists():
            stored_path.unlink()
        raise DocumentProcessingError(f"Failed to parse document: {exc}") from exc

    chunks = split_pages_into_chunks(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        if stored_path.exists():
            stored_path.unlink()
        raise DocumentProcessingError("No readable text was found in the uploaded document.")

    return ProcessedDocument(
        document_id=document_id,
        filename=filename,
        content_type=SUPPORTED_EXTENSIONS[extension],
        stored_path=stored_path,
        page_count=max(len(pages), 1),
        chunks=chunks,
    )


def delete_stored_file(path: Path) -> None:
    """Delete *path* from disk if it exists. Safe to call on missing paths."""
    if path.exists():
        path.unlink()


def extract_pages_from_pdf(path: Path) -> list[PageContent]:
    """Extract page text from the PDF at *path*.

    Separate from :func:`persist_and_process_upload` so callers that already
    have a file on disk — the evaluation runner ingesting its fixture corpus,
    for instance — can reuse the same extraction without constructing an
    :class:`~fastapi.UploadFile`.
    """
    reader = PdfReader(str(path))
    pages: list[PageContent] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(PageContent(text=page.extract_text() or "", page_number=index))
    return pages


def _extract_pages(path: Path, extension: str) -> list[PageContent]:
    """Extract page text from a supported document file.

    Args:
        path: Path to the file on disk.
        extension: Lowercase file extension (e.g. ``".pdf"``).

    Returns:
        Ordered list of :class:`PageContent` objects, one per page.

    Raises:
        DocumentProcessingError: If the extension is not supported.
    """
    if extension != ".pdf":
        raise DocumentProcessingError("Unsupported file type. Upload a PDF file.")

    return extract_pages_from_pdf(path)