"""Read a served ``brain_search`` tools/call answer the way BrainBar actually writes it (#1069).

BrainBar's ``TextFormatter.formatSearchResults`` emits one row per hit as a line starting
``- ID: `` -- in compact detail under a ``## "<query>" N/M`` header plus a legend line, in full
detail under ``## Search results for "<query>" - N of M shown``. The hybrid helper's full-detail
text (``mcp/_format.py``) keeps the older ``### N.`` headings under that same long header. Probes
that matched only the long header and ``### N.`` saw zero rows in every compact answer.

Only the FIRST listing header counts, and a row is a line that *starts* with ``- ID: ``. Both
formatters flatten newlines out of previews and summaries, so retrieved text can carry a literal
header or ``- ID:`` inside a row but can never start a line with one. That is also why IDs must
never contain ``|`` or a newline: the compact row delimits its columns with ``|``.
"""

from __future__ import annotations

import re
from typing import Any

LISTING_HEADER = re.compile(r'^## (?:Search results for .* - (\d+) of (\d+) shown|".*" (\d+)/(\d+))$', re.MULTILINE)
ID_ROW = "- ID: "
# Continuation lines of one ``- ID:`` row: compact ``S:``/``P:``, full-detail indented fields.
ROW_CONTINUATION = ("S:", "P:", "  ")
NUMBERED_ROW = re.compile(r"^### \d+\..*?(?=^### \d+\.|\Z)", re.MULTILINE | re.DOTALL)


def structured_results(result: Any) -> list[Any] | None:
    """The hybrid helper's authoritative ``structuredContent.results`` list, when the answer has one."""
    structured = result.get("structuredContent") if isinstance(result, dict) else None
    rows = structured.get("results") if isinstance(structured, dict) else None
    return rows if isinstance(rows, list) else None


def search_result_rows(text: str) -> list[str] | None:
    """The result rows of the first search listing in ``text``.

    ``[]`` is a recognised listing with no rows; ``None`` means ``text`` carries no listing header
    at all (an unknown shape), so a caller can keep its own fallback instead of reading it as empty.
    """
    header = LISTING_HEADER.search(text)
    if header is None:
        return None
    shown = int(header.group(1) or header.group(3))
    if shown == 0:
        return []
    body = text[header.end() :]
    lines = body.splitlines()
    first = next((line for line in lines if line.startswith(ID_ROW) or line.startswith("### ")), None)
    if first is None:
        return []
    if first.startswith("### "):
        return NUMBERED_ROW.findall(body)[:shown]
    rows: list[list[str]] = []
    for line in lines:
        if line.startswith(ID_ROW):
            rows.append([line])
        elif rows and line.startswith(ROW_CONTINUATION):
            rows[-1].append(line)
        elif rows:
            break  # trailing notices (search mode, KG degrade, order) are not part of any row
    return ["\n".join(row) for row in rows]


def search_result_count(text: str, result: Any = None) -> int | None:
    """Hits in a ``brain_search`` answer: structuredContent first, then counted ``- ID:`` rows."""
    structured = structured_results(result)
    if structured is not None:
        return len(structured)
    rows = search_result_rows(text)
    return None if rows is None else len(rows)
