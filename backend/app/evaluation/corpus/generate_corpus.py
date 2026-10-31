"""Generate the evaluation corpus PDFs with no third-party dependencies.

The corpus is four fictional ShieldCare insurance documents.  Three of them are
*structurally identical and differ only in their values*, which is the whole
point: a question like "what is the deductible?" matches all three almost
equally on lexical overlap, so answering it correctly requires the retrieval
layer to actually discriminate between documents.  The fourth is a comparison
table laid out with absolute text positioning, which is how real PDFs encode
tables and which reproduces the extraction mangling that fixed-size chunking
has to cope with.

Run from the backend directory to regenerate the committed PDFs::

    python -m app.evaluation.corpus.generate_corpus

The generated files are committed alongside this script so that an evaluation
run never has to regenerate them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN_LEFT = 56
MARGIN_TOP = 780
BODY_SIZE = 10.5
BODY_LEADING = 15.5
# Helvetica averages a little under half the font size per character; 92 keeps
# wrapped body lines inside the right margin without needing real font metrics.
WRAP_COLUMNS = 92

FONT_REGULAR = "F1"
FONT_BOLD = "F2"


@dataclass(slots=True)
class TextRun:
    """A single positioned run of text within a page.

    Attributes:
        x: Horizontal position in PDF user-space points from the left edge.
        y: Vertical position in points from the bottom edge.
        text: The literal string to draw.
        font: Resource name of the font to use (``"F1"`` or ``"F2"``).
        size: Font size in points.
    """

    x: float
    y: float
    text: str
    font: str = FONT_REGULAR
    size: float = BODY_SIZE


@dataclass(slots=True)
class Page:
    """An ordered collection of positioned text runs forming one page."""

    runs: list[TextRun] = field(default_factory=list)


class PageBuilder:
    """Accumulates text runs top-to-bottom, starting a new page when full."""

    def __init__(self) -> None:
        self._pages: list[Page] = [Page()]
        self._y = MARGIN_TOP

    def heading(self, text: str, size: float = 13.0) -> None:
        """Place *text* in bold, with extra space above it."""
        self._advance(BODY_LEADING * 0.8)
        self._place(text, font=FONT_BOLD, size=size)
        self._advance(BODY_LEADING * 0.35)

    def title(self, text: str) -> None:
        """Place *text* as the document title."""
        self._place(text, font=FONT_BOLD, size=16.0)
        self._advance(BODY_LEADING * 0.6)

    def body(self, text: str) -> None:
        """Place *text* as wrapped body copy."""
        for line in _wrap(text, WRAP_COLUMNS):
            self._place(line)

    def gap(self, amount: float = BODY_LEADING) -> None:
        """Leave *amount* points of vertical space."""
        self._advance(amount)

    def row(self, cells: list[str], columns: list[float], bold: bool = False) -> None:
        """Place *cells* at the absolute x positions given by *columns*.

        This is how a table is encoded in a real PDF: independent text runs at
        fixed coordinates, with no structural relationship between a cell and
        its column header.
        """
        page = self._current_page(BODY_LEADING)
        font = FONT_BOLD if bold else FONT_REGULAR
        for cell, x in zip(cells, columns, strict=False):
            page.runs.append(TextRun(x=x, y=self._y, text=cell, font=font))
        self._advance(BODY_LEADING)

    def pages(self) -> list[Page]:
        """Return the accumulated pages, dropping a trailing empty one."""
        if len(self._pages) > 1 and not self._pages[-1].runs:
            return self._pages[:-1]
        return self._pages

    def _place(self, text: str, font: str = FONT_REGULAR, size: float = BODY_SIZE) -> None:
        page = self._current_page(BODY_LEADING)
        page.runs.append(TextRun(x=MARGIN_LEFT, y=self._y, text=text, font=font, size=size))
        self._advance(BODY_LEADING if size <= BODY_SIZE else size * 1.45)

    def _advance(self, amount: float) -> None:
        self._y -= amount

    def _current_page(self, needed: float) -> Page:
        """Return the page with room for *needed* points, starting one if not."""
        if self._y - needed < 70:
            self._pages.append(Page())
            self._y = MARGIN_TOP
        return self._pages[-1]


def _wrap(text: str, columns: int) -> list[str]:
    """Greedily wrap *text* to at most *columns* characters per line."""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= columns:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _escape(text: str) -> bytes:
    """Encode *text* as a PDF literal string body using WinAnsi code points.

    Backslashes and both parentheses are escaped because they delimit literal
    strings.  Characters outside ASCII are emitted as three-digit octal escapes
    of their WinAnsi byte, which is what the fonts declare.
    """
    out = bytearray()
    for raw_byte in text.encode("cp1252", errors="replace"):
        if raw_byte in (0x5C, 0x28, 0x29):  # backslash, ( and )
            out.extend(b"\\")
            out.append(raw_byte)
        elif 32 <= raw_byte < 127:
            out.append(raw_byte)
        else:
            out.extend(f"\\{raw_byte:03o}".encode("ascii"))
    return bytes(out)


def _content_stream(page: Page) -> bytes:
    """Render *page* into a PDF content stream.

    Each run is emitted as its own text object with an absolute text matrix,
    which keeps positioning independent of the preceding run.
    """
    parts: list[bytes] = []
    for run in page.runs:
        parts.append(b"BT")
        parts.append(f"/{run.font} {run.size:g} Tf".encode("ascii"))
        parts.append(f"1 0 0 1 {run.x:g} {run.y:g} Tm".encode("ascii"))
        parts.append(b"(" + _escape(run.text) + b") Tj")
        parts.append(b"ET")
    return b"\n".join(parts) + b"\n"


def build_pdf(pages: list[Page], title: str) -> bytes:
    """Assemble *pages* into a complete single-file PDF document.

    Builds the object table by hand and computes the cross-reference offsets
    from the emitted byte positions, so the result needs no external library to
    produce and is readable by any conforming parser.
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        """Append an object and return its 1-based object number."""
        objects.append(body)
        return len(objects)

    # Reserve object 1 for the catalog and 2 for the page tree so that the
    # page objects can reference the tree before it is written.
    objects.append(b"")
    objects.append(b"")
    font_regular = add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>"
    )
    font_bold = add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
        b"/Encoding /WinAnsiEncoding >>"
    )
    info = add(b"<< /Title (" + _escape(title) + b") /Producer (DocTongue eval corpus) >>")

    page_numbers: list[int] = []
    for page in pages:
        stream = _content_stream(page)
        contents = add(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream"
        )
        page_numbers.append(
            add(
                b"<< /Type /Page /Parent 2 0 R "
                b"/MediaBox [0 0 " + f"{PAGE_WIDTH} {PAGE_HEIGHT}".encode("ascii") + b"] "
                b"/Resources << /Font << "
                b"/" + FONT_REGULAR.encode("ascii") + b" " + f"{font_regular} 0 R".encode("ascii") + b" "
                b"/" + FONT_BOLD.encode("ascii") + b" " + f"{font_bold} 0 R".encode("ascii") + b" "
                b">> >> "
                b"/Contents " + f"{contents} 0 R".encode("ascii") + b" >>"
            )
        )

    kids = b" ".join(f"{number} 0 R".encode("ascii") for number in page_numbers)
    objects[1] = (
        b"<< /Type /Pages /Kids [" + kids + b"] /Count "
        + str(len(page_numbers)).encode("ascii") + b" >>"
    )
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{index} 0 obj\n".encode("ascii"))
        out.extend(body)
        out.extend(b"\nendobj\n")

    xref_offset = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(
        b"trailer\n<< /Size " + str(len(objects) + 1).encode("ascii")
        + b" /Root 1 0 R /Info " + f"{info} 0 R".encode("ascii") + b" >>\n"
    )
    out.extend(f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(out)


@dataclass(frozen=True, slots=True)
class PlanSpec:
    """The varying values of one ShieldCare plan tier."""

    tier: str
    policy_reference: str
    monthly_premium: str
    deductible: str
    ceiling: str
    claim_window: str
    dental: str
    physiotherapy: str
    waiting_period: str
    out_of_network: str
    nurse_line: str
    excess_per_claim: str


PLANS: tuple[PlanSpec, ...] = (
    PlanSpec(
        tier="Basic",
        policy_reference="SC-BAS-2025",
        monthly_premium="£24",
        deductible="£1,500",
        ceiling="£50,000",
        claim_window="30 days",
        dental="is not included",
        physiotherapy="no physiotherapy sessions",
        waiting_period="90 days",
        out_of_network="40%",
        nurse_line="is not available",
        excess_per_claim="£75",
    ),
    PlanSpec(
        tier="Standard",
        policy_reference="SC-STD-2025",
        monthly_premium="£48",
        deductible="£750",
        ceiling="£250,000",
        claim_window="60 days",
        dental="is included",
        physiotherapy="6 physiotherapy sessions per year",
        waiting_period="30 days",
        out_of_network="60%",
        nurse_line="is available on weekdays",
        excess_per_claim="£40",
    ),
    PlanSpec(
        tier="Premium",
        policy_reference="SC-PRM-2025",
        monthly_premium="£95",
        deductible="£250",
        ceiling="£1,000,000",
        claim_window="90 days",
        dental="is included",
        physiotherapy="24 physiotherapy sessions per year",
        waiting_period="14 days",
        out_of_network="80%",
        nurse_line="is available 24 hours a day",
        excess_per_claim="£10",
    ),
)


def plan_document(spec: PlanSpec) -> list[Page]:
    """Build the pages for one plan's policy summary.

    Section headings and sentence structure are identical across tiers so that
    only the values distinguish the documents.
    """
    builder = PageBuilder()
    builder.title(f"ShieldCare Health Insurance - {spec.tier} Plan")
    builder.body(
        f"Policy summary document. Policy reference {spec.policy_reference}. "
        "Effective from 1 January 2025 until 31 December 2025. "
        "This summary is issued by ShieldCare Insurance Services Limited, a fictional "
        "entity used for evaluation purposes only."
    )

    builder.heading("1. Monthly premium")
    builder.body(
        f"The monthly premium for the {spec.tier} plan is {spec.monthly_premium} per insured "
        "adult. Premiums are collected by direct debit on the first working day of each month."
    )

    builder.heading("2. Annual deductible")
    builder.body(
        f"The {spec.tier} plan carries an annual deductible of {spec.deductible}. The deductible "
        "applies once per policy year across all eligible claims, and must be met before "
        "ShieldCare begins to reimburse treatment costs."
    )

    builder.heading("3. Annual coverage ceiling")
    builder.body(
        f"The maximum amount payable under the {spec.tier} plan is {spec.ceiling} per policy "
        "year. Costs above the ceiling remain the responsibility of the policyholder."
    )

    builder.heading("4. Claim submission window")
    builder.body(
        f"Claims under the {spec.tier} plan must be submitted within {spec.claim_window} of the "
        "date of treatment. Claims received after this window are declined without review."
    )

    builder.heading("5. Excess per claim")
    builder.body(
        f"An excess of {spec.excess_per_claim} applies to each separate claim made under the "
        f"{spec.tier} plan, in addition to the annual deductible."
    )

    builder.heading("6. Dental cover")
    builder.body(
        f"Routine and restorative dental treatment {spec.dental} under the {spec.tier} plan."
    )

    builder.heading("7. Physiotherapy")
    builder.body(
        f"The {spec.tier} plan covers {spec.physiotherapy}, subject to referral by a "
        "registered general practitioner."
    )

    builder.heading("8. Waiting period")
    builder.body(
        f"A waiting period of {spec.waiting_period} applies from the policy start date before "
        "non-emergency treatment becomes eligible under the {tier} plan.".format(tier=spec.tier)
    )

    builder.heading("9. Out-of-network reimbursement")
    builder.body(
        f"Treatment obtained outside the ShieldCare provider network is reimbursed at "
        f"{spec.out_of_network} of the eligible cost under the {spec.tier} plan."
    )

    builder.heading("10. Nurse advice line")
    builder.body(
        f"The ShieldCare nurse advice line {spec.nurse_line} to {spec.tier} plan members."
    )

    builder.heading("11. Exclusions")
    builder.body(
        "The following are excluded from all ShieldCare plans: experimental treatment not "
        "approved by a national regulator, injuries sustained during professional sport, and "
        "treatment arising from a declared act of war."
    )

    builder.heading("12. Cancellation")
    builder.body(
        "Policyholders may cancel within 14 days of the policy start date for a full refund. "
        "After 14 days, a pro-rata refund is issued less any claims already paid."
    )

    return builder.pages()


def comparison_document() -> list[Page]:
    """Build the cross-tier comparison table document."""
    builder = PageBuilder()
    builder.title("ShieldCare Plan Comparison 2025")
    builder.body(
        "The table below compares the three ShieldCare plan tiers. Where this table and a plan's "
        "own policy summary disagree, the policy summary takes precedence."
    )
    builder.gap()

    columns = [MARGIN_LEFT, 250.0, 350.0, 455.0]
    builder.row(["Feature", "Basic", "Standard", "Premium"], columns, bold=True)
    builder.gap(4)

    rows = [
        ("Monthly premium", "£24", "£48", "£95"),
        ("Annual deductible", "£1,500", "£750", "£250"),
        ("Annual coverage ceiling", "£50,000", "£250,000", "£1,000,000"),
        ("Claim submission window", "30 days", "60 days", "90 days"),
        ("Excess per claim", "£75", "£40", "£10"),
        ("Dental cover", "No", "Yes", "Yes"),
        ("Physiotherapy sessions", "0", "6", "24"),
        ("Waiting period", "90 days", "30 days", "14 days"),
        ("Out-of-network reimbursement", "40%", "60%", "80%"),
        ("Nurse advice line", "No", "Weekdays", "24 hours"),
        ("Overseas emergency cover", "No", "Yes", "Yes"),
        ("Annual health assessment", "No", "No", "Yes"),
    ]
    for feature, basic, standard, premium in rows:
        builder.row([feature, basic, standard, premium], columns)

    builder.gap()
    builder.heading("Switching between plans")
    builder.body(
        "Members may upgrade to a higher tier at any policy anniversary. A downgrade takes effect "
        "at the start of the next policy year. Waiting periods are not reapplied when upgrading, "
        "provided the policy has been continuously active."
    )

    return builder.pages()


# Deterministic ids, so `expected_sources` in the test set stays stable and a
# re-ingestion upserts over the previous run instead of duplicating chunks.
CORPUS: tuple[tuple[str, str], ...] = (
    ("eval-shield-basic", "shield_basic.pdf"),
    ("eval-shield-standard", "shield_standard.pdf"),
    ("eval-shield-premium", "shield_premium.pdf"),
    ("eval-shield-comparison", "shield_plan_comparison.pdf"),
)


def generate(output_dir: Path | None = None) -> list[Path]:
    """Write every corpus PDF into *output_dir* and return the written paths."""
    target = output_dir or Path(__file__).resolve().parent
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for spec in PLANS:
        pages = plan_document(spec)
        title = f"ShieldCare {spec.tier} Plan"
        path = target / f"shield_{spec.tier.lower()}.pdf"
        path.write_bytes(build_pdf(pages, title))
        written.append(path)

    comparison_path = target / "shield_plan_comparison.pdf"
    comparison_path.write_bytes(build_pdf(comparison_document(), "ShieldCare Plan Comparison"))
    written.append(comparison_path)
    return written


if __name__ == "__main__":
    for written_path in generate():
        print(f"wrote {written_path}")
