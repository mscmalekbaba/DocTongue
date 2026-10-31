from app.core.config import Settings
from app.services.llm import AnswerGenerator
from app.services.retrieval import answer_question
from app.services.vector_store import SearchResult


class StubVectorStore:
    def __init__(self, hits: list[SearchResult]) -> None:
        self._hits = hits
        self.last_query: str | None = None

    def search(self, question: str, limit: int) -> list[SearchResult]:
        self.last_query = question
        return self._hits[:limit]


class StubMemoryStore:
    def __init__(self, recent_turns: list[dict[str, str]] | None = None) -> None:
        self.recent_turns = recent_turns or []
        self.appended_turns: list[tuple[str, str, str]] = []

    def get_recent_turns(self, session_id: str, limit: int | None = None) -> list[dict[str, str]]:
        if limit is None:
            return list(self.recent_turns)
        return self.recent_turns[:limit]

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        self.appended_turns.append((session_id, question, answer))


class FailingAnswerGenerator:
    def reformulate_query(self, question: str, recent_turns: list[dict[str, str]]) -> str:
        return question

    def generate_answer(self, question: str, contexts: list[str]) -> str:
        raise RuntimeError("provider failure")


class MarkerAnswerGenerator:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.last_recent_turns: list[dict[str, str]] = []

    def reformulate_query(self, question: str, recent_turns: list[dict[str, str]]) -> str:
        self.last_recent_turns = recent_turns
        return question

    def generate_answer(self, question: str, contexts: list[str]) -> str:
        return self._answer


def test_answer_question_returns_not_found_when_no_hits() -> None:
    settings = Settings()
    answer_generator = AnswerGenerator(settings)
    vector_store = StubVectorStore([])
    memory_store = StubMemoryStore()

    response = answer_question(
        question="What is the retention policy?",
        session_id="test-session",
        settings=settings,
        vector_store=vector_store,
        answer_generator=answer_generator,
        chat_memory_store=memory_store,
    )

    assert response.grounded is False
    assert response.citations == []
    assert response.answer == "I could not find an answer in the indexed documents."
    assert memory_store.appended_turns == [
        (
            "test-session",
            "What is the retention policy?",
            "I could not find an answer in the indexed documents.",
        )
    ]


def test_answer_question_appends_audit_log(tmp_path) -> None:
    settings = Settings(storage_root=tmp_path / "data")
    settings.ensure_storage()
    answer_generator = AnswerGenerator(settings)
    vector_store = StubVectorStore([])
    memory_store = StubMemoryStore()

    response = answer_question(
        question="What is the retention policy?",
        session_id="test-session",
        settings=settings,
        vector_store=vector_store,
        answer_generator=answer_generator,
        chat_memory_store=memory_store,
    )

    audit_text = settings.chat_audit_log_path.read_text(encoding="utf-8")

    assert response.answer in audit_text
    assert "Question: What is the retention policy?" in audit_text
    assert "Answer: I could not find an answer in the indexed documents." in audit_text
    assert "Timestamp: " in audit_text
    assert audit_text.endswith("---\n")


def test_answer_question_rejects_weak_support() -> None:
    settings = Settings()
    answer_generator = AnswerGenerator(settings)
    memory_store = StubMemoryStore()
    vector_store = StubVectorStore(
        [
            SearchResult(
                document_id="doc-1",
                filename="guide.txt",
                content="Orange orchard harvest timing and irrigation details.",
                page_number=1,
                score=0.2,
            )
        ]
    )

    response = answer_question(
        question="What is the employee expense reimbursement deadline?",
        session_id="test-session",
        settings=settings,
        vector_store=vector_store,
        answer_generator=answer_generator,
        chat_memory_store=memory_store,
    )

    assert response.grounded is False
    assert response.citations == []
    assert (
        response.answer
        == "I could not find enough supporting evidence in the indexed documents to answer that."
    )
    assert memory_store.appended_turns == [
        (
            "test-session",
            "What is the employee expense reimbursement deadline?",
            "I could not find enough supporting evidence in the indexed documents to answer that.",
        )
    ]


def test_answer_question_returns_evidence_fallback_when_llm_fails() -> None:
    settings = Settings()
    memory_store = StubMemoryStore()
    vector_store = StubVectorStore(
        [
            SearchResult(
                document_id="doc-1",
                filename="guide.txt",
                content="The retention policy keeps records for seven years.",
                page_number=2,
                score=0.8,
            )
        ]
    )

    response = answer_question(
        question="What is the retention policy?",
        session_id="test-session",
        settings=settings,
        vector_store=vector_store,
        answer_generator=FailingAnswerGenerator(),
        chat_memory_store=memory_store,
    )

    assert response.grounded is False
    assert response.citations
    assert response.citations[0].filename == "guide.txt"
    assert "could not generate" in response.answer
    assert memory_store.appended_turns[-1] == (
        "test-session",
        "What is the retention policy?",
        response.answer,
    )


def test_answer_question_binds_citations_to_referenced_sources() -> None:
    settings = Settings(max_answer_citations=3)
    memory_store = StubMemoryStore()
    vector_store = StubVectorStore(
        [
            SearchResult(
                document_id="doc-1",
                filename="policy-a.txt",
                content="Expense reports are due by Friday.",
                page_number=1,
                score=0.92,
            ),
            SearchResult(
                document_id="doc-2",
                filename="policy-b.txt",
                content="Retention policy keeps records for seven years.",
                page_number=2,
                score=0.89,
            ),
        ]
    )

    response = answer_question(
        question="What is the retention policy?",
        session_id="test-session",
        settings=settings,
        vector_store=vector_store,
        answer_generator=MarkerAnswerGenerator(
            "The retention policy keeps records for seven years. [2]"
        ),
        chat_memory_store=memory_store,
    )

    assert response.grounded is True
    assert len(response.citations) == 1
    assert response.citations[0].document_id == "doc-2"
    assert response.citations[0].filename == "policy-b.txt"


def test_answer_question_deduplicates_citations_by_document() -> None:
    settings = Settings(max_answer_citations=3)
    memory_store = StubMemoryStore()
    vector_store = StubVectorStore(
        [
            SearchResult(
                document_id="doc-1",
                filename="interview-options.txt",
                content="Option A: Chat with your docs.",
                page_number=1,
                score=0.95,
            ),
            SearchResult(
                document_id="doc-1",
                filename="interview-options.txt",
                content="Option B: Code documentation assistant.",
                page_number=1,
                score=0.92,
            ),
            SearchResult(
                document_id="doc-2",
                filename="guidelines.txt",
                content="Choose the option that excites you most.",
                page_number=2,
                score=0.9,
            ),
        ]
    )

    response = answer_question(
        question="What are the interview project options?",
        session_id="test-session",
        settings=settings,
        vector_store=vector_store,
        answer_generator=MarkerAnswerGenerator(
            "Available options include chat with docs [1], code documentation assistant [2], and guidance [3]."
        ),
        chat_memory_store=memory_store,
    )

    assert response.grounded is True
    assert len(response.citations) == 2
    assert response.citations[0].document_id == "doc-1"
    assert response.citations[1].document_id == "doc-2"
    assert "[2]" in response.answer


def test_answer_question_normalizes_markers_for_same_document() -> None:
    settings = Settings(max_answer_citations=3)
    memory_store = StubMemoryStore()
    vector_store = StubVectorStore(
        [
            SearchResult(
                document_id="doc-1",
                filename="interview-options.txt",
                content="Option A: Chat with your docs.",
                page_number=1,
                score=0.95,
            ),
            SearchResult(
                document_id="doc-1",
                filename="interview-options.txt",
                content="Option B: Code documentation assistant.",
                page_number=1,
                score=0.92,
            ),
        ]
    )

    response = answer_question(
        question="What are the interview project options?",
        session_id="test-session",
        settings=settings,
        vector_store=vector_store,
        answer_generator=MarkerAnswerGenerator("Options include A and B [1][2]."),
        chat_memory_store=memory_store,
    )

    assert response.grounded is True
    assert len(response.citations) == 1
    assert response.citations[0].document_id == "doc-1"
    assert "[1][2]" not in response.answer
    assert "[1]." in response.answer or "[1]" in response.answer


def test_answer_question_uses_reformulated_query_and_window_history() -> None:
    settings = Settings(max_answer_citations=3, chat_memory_window=3)
    memory_store = StubMemoryStore(
        [
            {
                "question": "What does the onboarding policy say about contractors?",
                "answer": "Contractors complete onboarding in five days.",
            },
            {
                "question": "And what about interns?",
                "answer": "Intern onboarding uses the same timeline.",
            },
            {
                "question": "Does that include remote workers?",
                "answer": "Yes, remote workers are included.",
            },
            {
                "question": "Old question that should be trimmed by limit",
                "answer": "Old answer that should be trimmed by limit",
            },
        ]
    )
    vector_store = StubVectorStore(
        [
            SearchResult(
                document_id="doc-1",
                filename="onboarding.txt",
                content="Interns and contractors must complete onboarding in five days.",
                page_number=1,
                score=0.94,
            )
        ]
    )

    class ReformulatingGenerator(MarkerAnswerGenerator):
        def reformulate_query(self, question: str, recent_turns: list[dict[str, str]]) -> str:
            self.last_recent_turns = recent_turns
            return "contractor intern remote onboarding requirements"

    answer_generator = ReformulatingGenerator(
        "Interns and contractors complete onboarding in five days. [1]"
    )

    response = answer_question(
        question="What are their onboarding requirements?",
        session_id="session-42",
        settings=settings,
        vector_store=vector_store,
        answer_generator=answer_generator,
        chat_memory_store=memory_store,
    )

    assert response.grounded is True
    assert vector_store.last_query == "contractor intern remote onboarding requirements"
    assert answer_generator.last_recent_turns == [
        {
            "question": "What does the onboarding policy say about contractors?",
            "answer": "Contractors complete onboarding in five days.",
        },
        {
            "question": "And what about interns?",
            "answer": "Intern onboarding uses the same timeline.",
        },
        {
            "question": "Does that include remote workers?",
            "answer": "Yes, remote workers are included.",
        },
    ]
    assert memory_store.appended_turns[-1] == (
        "session-42",
        "What are their onboarding requirements?",
        "Interns and contractors complete onboarding in five days. [1]",
    )


def test_answer_question_remaps_single_marker_line_to_better_supported_source() -> None:
    settings = Settings(max_answer_citations=3)
    memory_store = StubMemoryStore()
    vector_store = StubVectorStore(
        [
            SearchResult(
                document_id="doc-1",
                filename="resource-1.txt",
                content="The exercise asks for submission via an LMS portal with no repository requirement.",
                page_number=1,
                score=0.93,
            ),
            SearchResult(
                document_id="doc-2",
                filename="resource-2.txt",
                content="Submission requires a GitHub repository and a README.md with quick setup instructions.",
                page_number=3,
                score=0.88,
            ),
        ]
    )

    response = answer_question(
        question="How should I submit the VIBE exercise?",
        session_id="test-session",
        settings=settings,
        vector_store=vector_store,
        answer_generator=MarkerAnswerGenerator(
            "Submission requires a GitHub repository with README quick setup instructions. [1]"
        ),
        chat_memory_store=memory_store,
    )

    assert response.grounded is True
    assert response.answer.endswith("[1]")
    assert len(response.citations) == 1
    assert response.citations[0].document_id == "doc-2"
    assert response.citations[0].filename == "resource-2.txt"


def test_answer_question_keeps_multi_marker_line_unchanged() -> None:
    settings = Settings(max_answer_citations=3)
    memory_store = StubMemoryStore()
    vector_store = StubVectorStore(
        [
            SearchResult(
                document_id="doc-1",
                filename="resource-1.txt",
                content="Requirement A details.",
                page_number=1,
                score=0.93,
            ),
            SearchResult(
                document_id="doc-2",
                filename="resource-2.txt",
                content="Requirement B details.",
                page_number=2,
                score=0.91,
            ),
        ]
    )

    response = answer_question(
        question="List the requirements.",
        session_id="test-session",
        settings=settings,
        vector_store=vector_store,
        answer_generator=MarkerAnswerGenerator(
            "The requirements include A [1] and B [2]."
        ),
        chat_memory_store=memory_store,
    )

    assert response.grounded is True
    assert "[1] and B [2]" in response.answer
    assert len(response.citations) == 2