"""Build the RAG corpus from two sources:

1. Cantera's official sample scripts from GitHub (samples/python/**)
2. Docstrings from the installed Cantera package via introspection

Examples show correct API usage in realistic combinations (good for showing
CodeGen what a full script looks like). Introspected docstrings carry
authoritative signatures from YOUR INSTALLED VERSION (3.2.0), so they catch
things like the clone=False deprecation note.

Run once: python -m rag.build_corpus
Output:   data/docs/corpus.jsonl (one doc per line)
"""
from __future__ import annotations
import inspect
import json
import shutil
import subprocess
from pathlib import Path

import cantera as ct

from config import DOCS_DIR


CANTERA_GITHUB = "https://github.com/Cantera/cantera.git"
SAMPLES_SUBDIR = "samples/python"


# ---------- Examples from GitHub ----------

def clone_samples(dest: Path) -> Path:
    """Sparse-clone just samples/python/ to avoid pulling the whole repo."""
    if dest.exists():
        print(f"[corpus] samples already at {dest}, skipping clone")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / "_cantera_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    print(f"[corpus] sparse-cloning {SAMPLES_SUBDIR} from Cantera...")
    subprocess.run(
        ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
         CANTERA_GITHUB, str(tmp)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", SAMPLES_SUBDIR],
        cwd=str(tmp), check=True, capture_output=True,
    )
    shutil.move(str(tmp / SAMPLES_SUBDIR), str(dest))
    shutil.rmtree(tmp)
    print(f"[corpus] samples at {dest}")
    return dest


def collect_example_docs(samples_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(samples_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text.strip()) < 100:
            continue
        rel = path.relative_to(samples_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "misc"
        docs.append({
            "source": "example",
            "path": str(rel),
            "category": category,
            "title": path.stem,
            "text": text,
        })
    print(f"[corpus] collected {len(docs)} example scripts")
    return docs


# ---------- Introspected API from installed Cantera ----------

TARGET_CLASSES = [
    "Solution", "IdealGasReactor", "IdealGasConstPressureReactor",
    "ConstPressureReactor", "Reactor", "ReactorNet", "Reservoir",
    "FreeFlame", "BurnerFlame", "CounterflowDiffusionFlame",
    "ThermoPhase", "Kinetics", "Transport",
]

TARGET_FUNCTIONS = ["one_atm", "gas_constant"]


def _member_doc(obj, qualname: str) -> str | None:
    doc = inspect.getdoc(obj)
    if not doc:
        return None
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        sig = ""
    header = f"{qualname}{sig}" if sig else qualname
    return f"{header}\n\n{doc}"


def collect_api_docs() -> list[dict]:
    docs = []

    for fname in TARGET_FUNCTIONS:
        fn = getattr(ct, fname, None)
        if fn is None:
            continue
        text = _member_doc(fn, f"cantera.{fname}")
        if text:
            docs.append({
                "source": "api", "path": f"cantera.{fname}",
                "category": "module", "title": fname, "text": text,
            })

    for cls_name in TARGET_CLASSES:
        cls = getattr(ct, cls_name, None)
        if cls is None:
            print(f"[corpus] WARN: ct.{cls_name} not found in Cantera {ct.__version__}")
            continue

        class_doc = _member_doc(cls, f"cantera.{cls_name}")
        if class_doc:
            docs.append({
                "source": "api", "path": f"cantera.{cls_name}",
                "category": "class", "title": cls_name, "text": class_doc,
            })

        for name, member in inspect.getmembers(cls):
            if name.startswith("_"):
                continue
            qual = f"cantera.{cls_name}.{name}"
            text = _member_doc(member, qual)
            if not text or len(text) < 80:
                continue
            docs.append({
                "source": "api", "path": qual,
                "category": "method" if callable(member) else "property",
                "title": f"{cls_name}.{name}", "text": text,
            })

    print(f"[corpus] collected {len(docs)} API doc entries")
    return docs


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    samples_dir = DOCS_DIR / "cantera_samples"
    clone_samples(samples_dir)

    all_docs = collect_example_docs(samples_dir) + collect_api_docs()

    out = DOCS_DIR / "corpus.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n[corpus] wrote {len(all_docs)} docs → {out}")
    categories: dict[str, int] = {}
    for d in all_docs:
        k = f"{d['source']}/{d['category']}"
        categories[k] = categories.get(k, 0) + 1
    for k in sorted(categories):
        print(f"           {k}: {categories[k]}")


if __name__ == "__main__":
    main()
