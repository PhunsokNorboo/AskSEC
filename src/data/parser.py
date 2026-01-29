"""SEC 10-K Filing Parser - Extract key sections from filings."""
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DocumentSection:
    """Represents a section of a 10-K filing."""
    item_number: str
    item_title: str
    content: str
    start_idx: int
    end_idx: int

    def __len__(self) -> int:
        return len(self.content)


class SEC10KParser:
    """Parse and extract sections from SEC 10-K filings."""

    # Key sections to extract (most valuable for RAG)
    SECTIONS = {
        "1": "Business",
        "1A": "Risk Factors",
        "1B": "Unresolved Staff Comments",
        "1C": "Cybersecurity",
        "7": "Management's Discussion and Analysis",
        "7A": "Quantitative and Qualitative Disclosures About Market Risk",
        "8": "Financial Statements and Supplementary Data",
    }

    def __init__(self):
        """Initialize the parser with section patterns."""
        self.section_patterns = self._build_section_patterns()

    def _build_section_patterns(self) -> Dict[str, re.Pattern]:
        """Build regex patterns to match section headers."""
        patterns = {}
        for item_num, item_title in self.SECTIONS.items():
            # Match variations like:
            # "Item 1A", "ITEM 1A.", "Item 1A -", "Item 1A:", "ITEM 1A. Risk Factors"
            # Also handle cases where title might have slight variations
            escaped_num = re.escape(item_num)
            pattern = rf'(?:ITEM|Item)\s*{escaped_num}[\.\s\-:\—]*(?:{re.escape(item_title)})?'
            patterns[item_num] = re.compile(pattern, re.IGNORECASE)
        return patterns

    def parse_filing(self, text: str) -> Dict[str, DocumentSection]:
        """
        Parse a 10-K filing and extract key sections.

        Args:
            text: Full text content of the 10-K filing

        Returns:
            Dictionary mapping item numbers to DocumentSection objects
        """
        # Clean the text first
        text = self._clean_text(text)

        # Find all section boundaries
        section_positions = []
        for item_num, pattern in self.section_patterns.items():
            matches = list(pattern.finditer(text))
            for match in matches:
                section_positions.append({
                    'start': match.start(),
                    'item_num': item_num,
                    'header': match.group(),
                    'item_title': self.SECTIONS[item_num]
                })

        # Sort by position in document
        section_positions.sort(key=lambda x: x['start'])

        # Remove duplicate section matches (keep first substantial one)
        seen_sections = {}
        unique_positions = []
        for pos in section_positions:
            item_num = pos['item_num']
            if item_num not in seen_sections:
                seen_sections[item_num] = pos
                unique_positions.append(pos)

        # Extract section content
        sections = {}
        for i, pos in enumerate(unique_positions):
            start_pos = pos['start']
            item_num = pos['item_num']

            # Find end position (start of next section or end of document)
            if i + 1 < len(unique_positions):
                end_pos = unique_positions[i + 1]['start']
            else:
                end_pos = len(text)

            content = text[start_pos:end_pos].strip()

            # Only keep sections with substantial content (> 500 chars)
            if len(content) > 500:
                sections[item_num] = DocumentSection(
                    item_number=item_num,
                    item_title=pos['item_title'],
                    content=content,
                    start_idx=start_pos,
                    end_idx=end_pos
                )

        return sections

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace (but preserve paragraph structure)
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double

        # Remove page numbers and common headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'Table of Contents', '', text, flags=re.IGNORECASE)

        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('—', '-').replace('–', '-')

        return text.strip()

    def get_section_summary(self, sections: Dict[str, DocumentSection]) -> Dict[str, int]:
        """Get a summary of extracted sections with character counts."""
        return {
            f"Item {num} ({sec.item_title})": len(sec.content)
            for num, sec in sections.items()
        }

    def extract_specific_sections(
        self,
        text: str,
        section_numbers: List[str]
    ) -> Dict[str, DocumentSection]:
        """
        Extract only specific sections from filing.

        Args:
            text: Full filing text
            section_numbers: List of section numbers to extract (e.g., ["1", "1A", "7"])

        Returns:
            Dictionary of requested sections
        """
        all_sections = self.parse_filing(text)
        return {
            num: sec for num, sec in all_sections.items()
            if num in section_numbers
        }
