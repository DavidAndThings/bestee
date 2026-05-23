"""Generic xlsx parsing helpers used by ETF-provider scrapers.

The Open XML SpreadsheetML format is a zip-of-XML; these helpers read
just enough of it (sharedStrings + sheet1) to produce a list-of-rows of
cell text values.  They are deliberately tiny and dependency-free so we
don't have to pull in ``openpyxl`` or ``fastexcel`` for the simple
holdings tables every issuer publishes.
"""

import re
import zipfile
from xml.etree import ElementTree as ET

_XLNS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_CELL_REF_RE = re.compile(r"^([A-Z]+)(\d+)$")


def col_letter_to_index(letter: str) -> int:
    """Convert ``A``, ``B``, ..., ``AA`` to a 0-based column index."""
    n = 0
    for ch in letter:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    """Return the workbook's sharedStrings table as a list."""
    try:
        raw = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    strings: list[str] = []
    for si in root.findall(f"{_XLNS}si"):
        # An <si> may contain a single <t> or several <r><t> runs.
        parts = [t.text or "" for t in si.iter(f"{_XLNS}t")]
        strings.append("".join(parts))
    return strings


def parse_sheet(
    zf: zipfile.ZipFile,
    shared: list[str],
    sheet_path: str = "xl/worksheets/sheet1.xml",
) -> list[list[str]]:
    """Parse a worksheet into a list of rows of cell text values.

    Cells referencing sharedStrings (``t="s"``) are resolved to their
    string values; all other cells return their raw ``<v>`` text.  Short
    rows are padded so each returned row has the same length as the row's
    rightmost populated cell.
    """
    root = ET.fromstring(zf.read(sheet_path))
    rows: list[list[str]] = []
    for row in root.iter(f"{_XLNS}row"):
        cells: dict[int, str] = {}
        max_col = -1
        for c in row.findall(f"{_XLNS}c"):
            ref = c.attrib.get("r", "")
            m = _CELL_REF_RE.match(ref)
            if m is None:
                continue
            col_idx = col_letter_to_index(m.group(1))
            max_col = max(max_col, col_idx)
            v = c.find(f"{_XLNS}v")
            if v is None or v.text is None:
                continue
            if c.attrib.get("t") == "s":
                try:
                    cells[col_idx] = shared[int(v.text)]
                except (ValueError, IndexError):
                    cells[col_idx] = ""
            else:
                cells[col_idx] = v.text
        if max_col >= 0:
            rows.append([cells.get(i, "") for i in range(max_col + 1)])
    return rows
