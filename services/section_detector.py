from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.rag_pipeline import _extract_section_title


@dataclass
class DetectedSection:
    section_name: str
    page_start: int
    page_end: int
    section_order: int


def detect_sections(pages: list[Any]) -> list[DetectedSection]:
    """
    Iterate through ParsedPage objects.
    For each page, scan lines for section headers using _extract_section_title.
    Track page_start and page_end for each section.
    Return list of DetectedSection sorted by section_order.
    """
    detected_sections: list[DetectedSection] = []

    for page in pages:
        for line in page.text.splitlines():
            section_title = _extract_section_title(line)
            if not section_title:
                continue

            if detected_sections:
                detected_sections[-1].page_end = page.page_number

            detected_sections.append(
                DetectedSection(
                    section_name=section_title,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    section_order=len(detected_sections) + 1,
                )
            )

    if detected_sections and pages:
        detected_sections[-1].page_end = pages[-1].page_number

    return sorted(detected_sections, key=lambda section: section.section_order)
