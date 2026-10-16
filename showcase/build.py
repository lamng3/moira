"""Build showcase/index.html: the site template with the recorded trace inlined, so it opens with no server.

python showcase/build.py   (after showcase/record.py)
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    trace = json.loads((HERE / "trace.json").read_text(encoding="utf-8"))
    data = json.dumps(trace, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    page = (HERE / "template.html").read_text(encoding="utf-8").replace("__TRACE__", data)
    head = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    head += '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
    (HERE / "index.html").write_text(head + page + "\n</body>\n</html>\n", encoding="utf-8")
    print(f"wrote {HERE / 'index.html'} ({(HERE / 'index.html').stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
