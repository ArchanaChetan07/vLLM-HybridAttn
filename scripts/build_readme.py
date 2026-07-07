"""Inject full diagram set into README.md."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    arch = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    diagrams = (ROOT / "docs" / "minicpm_sala_diagrams.md").read_text(encoding="utf-8")
    extra = (ROOT / "docs" / "readme_extra_diagrams.md").read_text(encoding="utf-8")

    arch_body = arch.split("## Overview", 1)[1]
    arch_body = re.split(r"\n## Further reading", arch_body, maxsplit=1)[0]

    diag_body = diagrams.split("## 1.", 1)[1]
    diag_parts = re.split(r"\n## \d+\. ", "\n" + diag_body)
    diag_md = ""
    for sec in diag_parts:
        if not sec.strip():
            continue
        title, _, rest = sec.partition("\n")
        diag_md += f"\n### {title.strip()}\n\n{rest.strip()}\n"

    section = f"""## Architecture

Complete Mermaid diagram set (renders on GitHub). See also [docs/architecture.md](docs/architecture.md) and [docs/minicpm_sala_diagrams.md](docs/minicpm_sala_diagrams.md).

{extra.strip()}

### From architecture.md

{arch_body.strip()}

### From minicpm_sala_diagrams.md

{diag_md.strip()}
"""

    start = readme.index("## Architecture\n\n")
    end_marker = "\n## Repository Layout"
    if end_marker not in readme:
        end_marker = "\n## Repository Structure"
    end = readme.index(end_marker)
    new_readme = readme[:start] + section.rstrip() + "\n\n" + readme[end + 1 :]

    new_readme = new_readme.replace("](#repository-structure)", "](#repository-layout)")
    (ROOT / "README.md").write_text(new_readme, encoding="utf-8", newline="\n")
    count = new_readme.count("```mermaid")
    print(f"README updated: {len(new_readme)} chars, {count} mermaid blocks")


if __name__ == "__main__":
    main()
