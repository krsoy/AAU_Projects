import os
import json
import csv
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Iterable
from multiprocessing import Pool, cpu_count

# ----------------- Config -----------------
FILE_PATH = Path("Webtext/Webtext")
OUT_CSV = Path("plastic_recycling_results.csv")
OUT_TXT = Path("matched_files.txt")
OUT_JSON = Path("matched_files.json")
MIN_HITS = 1  # require at least this many keyword hits to mark as match
MAX_SHARDS = 10  # split into at most this many parallel chunks

# ----------------- Keywords -----------------
EN_KEYWORDS = [
    "plastic recycling", "recycled plastic", "plastic waste", "plastic collection",
    "plastic sorting", "sorting line", "plastic baling", "plastic shredder",
    "plastic granulate", "plastic granulation", "plastic pellets", "pelletizing",
    "plastic regrind", "plastic flakes", "flake washing", "washing line", "hot wash",
    "recycling facility", "recycling plant", "material recovery facility", "mrf",
    "post-consumer plastic", "post industrial plastic", "pir", "pcr",
    "mechanical recycling", "chemical recycling", "pyrolysis", "depolymerization",
    "extrusion", "extruder", "compounding", "regranulation",
    "pet recycling", "pe recycling", "pp recycling", "ps recycling", "pvc recycling",
    "hdpe recycling", "ldpe recycling", "polyethylene terephthalate", "polypropylene",
    "polystyrene", "polyvinyl chloride", "polyethylene", "polyamide", "abs recycling",
    "bottle-to-bottle", "rpet", "rpp", "rhdpe", "recycled resin",
    "recycling technology", "optical sorter", "near-infrared sorter", "nir sorter",
    "float sink", "float-sink", "label remover", "friction washer",
    "dewatering", "dryer", "agglomerator", "pelletizer", "melt filter",
    "screen changer", "granulator", "crusher", "shredder", "conveyor",
    "silo", "big bag station",
]

IT_KEYWORDS = [
    "riciclaggio della plastica", "riciclo della plastica", "plastica riciclata",
    "rifiuti plastici", "raccolta della plastica", "raccolta plastica",
    "selezione plastica", "cernita plastica", "linea di selezione",
    "linea di cernita", "balle di plastica", "pressatura plastica",
    "trituratore", "frantumatore", "mulino", "granulatore",
    "granulazione", "granulo di plastica", "granuli di plastica",
    "rigenerazione plastica", "rigenerati plastici", "rigenerato",
    "scaglie di plastica", "scaglie pet", "fiocchi pet",
    "lavaggio a caldo", "linea di lavaggio", "vasca di lavaggio",
    "lavatrice a frizione", "attrito", "essiccatore", "disidratazione",
    "impianto di riciclaggio", "impianto di recupero",
    "centro di selezione", "piattaforma di selezione",
    "imballaggi in plastica", "plastica post-consumo", "plastica post consumo",
    "plastica post industriale", "pcr", "pir",
    "riciclo meccanico", "riciclo chimico", "pirolisi", "depolimerizzazione",
    "estrusione", "estrusore", "compound", "compounding", "regranulazione",
    "riciclo pet", "riciclo pe", "riciclo pp", "riciclo ps", "riciclo pvc",
    "riciclo hdpe", "riciclo ldpe", "polietilene tereftalato",
    "polipropilene", "polistirene", "cloruro di polivinile",
    "polietilene", "poliammide", "abs riciclato",
    "bottle to bottle", "rpet", "rpp", "rhdpe", "resine riciclate",
    "selezionatrice ottica", "sorter ottico", "selezione nir", "nir",
    "vasca a flottazione", "flottazione", "float sink", "separatore densimetrico",
    "rimuovi etichette", "togli etichette", "friction washer",
    "filtro a massa fusa", "melt filter", "cambio filtro", "screen changer",
    "nastro trasportatore", "trasportatore", "tramoggia", "silo",
    "stazione big bag", "riempitrice big bag", "svuotatore big bag",
]

ALL_KEYWORDS = EN_KEYWORDS + IT_KEYWORDS

# ----------------- Helpers -----------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = " ".join(s.split())
    return s

def detect_keywords(text: str, keywords: List[str], min_hits: int = 1) -> Tuple[bool, List[str]]:
    t = normalize_text(text)
    matched = [kw for kw in keywords if normalize_text(kw) in t]
    return (len(matched) >= min_hits, matched)

def pick_text_fields(payload: Dict) -> str:
    return payload.get("text") or payload.get("description") or ""

def language_hint(matched: List[str]) -> str:
    it_hits = sum(1 for m in matched if m in IT_KEYWORDS)
    en_hits = sum(1 for m in matched if m in EN_KEYWORDS)
    if it_hits and en_hits:
        return "both"
    if it_hits:
        return "it"
    if en_hits:
        return "en"
    return "unknown"

def chunk(lst: List[str], n: int) -> Iterable[List[str]]:
    """Split list into n nearly equal chunks."""
    k = len(lst)
    if n <= 0:
        n = 1
    size = (k + n - 1) // n  # ceil(k/n)
    for i in range(0, k, size):
        yield lst[i:i + size]

# ----------------- Worker -----------------
def process_chunk(filenames: List[str]) -> Tuple[List[Dict], List[str]]:
    """
    Worker function. Processes a list of filenames (basename only),
    returns (rows, matched_files).
    """
    rows: List[Dict] = []
    matched_files: List[str] = []

    for file in filenames:
        file_path = FILE_PATH / file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            rows.append({
                'url':'N/A',
                "filename": file,
                "match": False,
                "hits": 0,
                "language_hint": "unknown",
                "matched_keywords": "",
                "preview": f"READ_ERROR: {e}"
            })
            continue

        text_to_check = pick_text_fields(data)
        if not text_to_check:
            rows.append({
                'url': data['url'] if 'url' in data else 'N/A',

                "filename": file,
                "match": False,
                "hits": 0,
                "language_hint": "unknown",
                "matched_keywords": "",
                "preview": "NO_TEXT_FIELDS"
            })
            continue

        is_match, matched = detect_keywords(text_to_check, ALL_KEYWORDS, MIN_HITS)
        hits = len(matched)
        lang = language_hint(matched)
        preview = normalize_text(text_to_check)[:200]

        if is_match:
            matched_files.append(file)

        rows.append({
            'url': data['url'] if 'url' in data else 'N/A',

            "filename": file,
            "match": bool(is_match),
            "hits": hits,
            "language_hint": lang,
            "matched_keywords": "; ".join(matched),
            "preview": preview
        })

    return rows, matched_files

# ----------------- Main -----------------
def main():
    all_files = sorted([f for f in os.listdir(FILE_PATH) if f.endswith(".json")])

    if not all_files:
        print("No JSON files found.")
        return

    # number of shards = min(10, number of files, cpu_count)
    shards = min(MAX_SHARDS, len(all_files), max(1, cpu_count()))
    file_chunks = list(chunk(all_files, shards))

    print(f"Processing {len(all_files)} files across {shards} process(es)...")

    results_rows: List[Dict] = []
    results_matched: List[str] = []

    # Windows requires if __name__ == '__main__' guard (present at bottom)
    with Pool(processes=shards) as pool:
        async_results = [pool.apply_async(process_chunk, (chunk,)) for chunk in file_chunks]
        # Wait and collect (this is the explicit join/wait via .get)
        for r in async_results:
            rows, matched = r.get()  # blocks until that chunk finishes
            results_rows.extend(rows)
            results_matched.extend(matched)

    # Sort rows by filename for determinism
    results_rows.sort(key=lambda r: r["filename"])
    results_matched = sorted(set(results_matched))

    # --- Save CSV with all rows ---
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["url","filename", "match", "hits", "language_hint", "matched_keywords", "preview"]
        )
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"Saved CSV: {OUT_CSV.resolve()}")

    # --- Save matched lists ---
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for fn in results_matched:
            f.write(fn + "\n")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_matched, f, ensure_ascii=False, indent=2)

    print(f"Saved matched lists:\n- {OUT_TXT.resolve()}\n- {OUT_JSON.resolve()}")

    # Console summary
    print(f"\nMatched {len(results_matched)} / {len(all_files)} files.")
    if results_matched:
        print("Sample matches:", ", ".join(results_matched[:10]))

    # dataframe summary
    try:
        import pandas as pd
        df = pd.DataFrame(results_rows)
        summary = df.groupby("language_hint")["match"].sum()
        df.to_csv(OUT_CSV, index=False)
    except ImportError:
        print("Pandas not installed; skipping dataframe summary.")

if __name__ == "__main__":
    main()
