"""One self-contained HTML renderer for experiment reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Mapping, Sequence


def render_report(title: str, sections: Sequence[Mapping[str, object]], output: str | Path) -> Path:
    """Render structured sections to a portable HTML report.

    A section has a ``title`` and one of ``text``, ``table``, or ``data``.
    Layout, escaping, JSON formatting and atomic output are hidden behind this
    interface.
    """

    cards = []
    for section in sections:
        heading = html.escape(str(section.get("title", "Section")))
        if "text" in section:
            body = f"<p>{html.escape(str(section['text']))}</p>"
        elif "table" in section:
            table = section["table"]
            columns = list(table.get("columns", []))
            rows = list(table.get("rows", []))
            headers = "".join(
                f"<th>{html.escape(str(column.get('label', column['key'])))}</th>"
                for column in columns
            )
            cells = []
            for row in rows:
                cells.append(
                    "<tr>" + "".join(
                        f"<td>{html.escape(str(row.get(column['key'], '')))}</td>"
                        for column in columns
                    ) + "</tr>"
                )
            body = (
                '<div class="table-wrap"><table><thead><tr>' + headers
                + "</tr></thead><tbody>" + "".join(cells) + "</tbody></table></div>"
            )
        else:
            payload = json.dumps(section.get("data"), indent=2, ensure_ascii=False, default=str)
            body = f"<pre>{html.escape(payload)}</pre>"
        cards.append(f"<section><h2>{heading}</h2>{body}</section>")
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title><style>
body{{margin:0;background:#f5f6f3;color:#20231f;font:16px/1.55 system-ui,sans-serif}}
main{{max-width:960px;margin:auto;padding:40px 20px}}section{{background:white;margin:18px 0;padding:22px;
border:1px solid #dde1d9;border-radius:12px}}h1,h2{{line-height:1.2}}pre{{overflow:auto;background:#f0f2ed;
padding:16px;border-radius:8px}}p{{white-space:pre-wrap}}.table-wrap{{overflow:auto}}table{{width:100%;
border-collapse:collapse;font-size:14px}}th,td{{padding:9px 10px;border-bottom:1px solid #dde1d9;
text-align:left;vertical-align:top}}th{{background:#f0f2ed;position:sticky;top:0}}</style></head>
<body><main><h1>{title}</h1>{cards}</main></body></html>
""".format(title=html.escape(title), cards="".join(cards))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
    return path

