import argparse, csv, json, hashlib, ast, re
from pathlib import Path
from typing import Dict, Iterable, List

# utils
from src.utils.io import read_jsonl
from src.utils.text import clean

# -----------------------------
# Regex helpers for normalization
# -----------------------------
# units incl. dotted abbrevs and plurals
_UNIT_RE = re.compile(
    r"""
    \b(
      c|cup|cups|
      tsp|tsp\.|teaspoon|teaspoons|
      tbsp|tbsp\.|tablespoon|tablespoons|
      oz|oz\.|ounce|ounces|
      g|gram|grams|kg|kgs|
      ml|l|liter|litre|liters|litres|
      lb|lb\.|pound|pounds|
      pt|pint|pints|
      qt|quart|quarts|
      gal|gallon|gallons|
      pkg|pkgs|package|packages|
      box|boxes
    )\b
    """, re.IGNORECASE | re.VERBOSE,
)


# numbers incl. unicode fractions & simple "1/2", "2 1/2", "2.5"
# examples matched: 1, 1/2, 2 1/2, 2.5, ½, ¼, ¾
_FRACTIONS = "¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞"
_NUM_RE = re.compile(
    rf"""
    (?<!\w)                  # not part of a word
    (?:\d+(?:\.\d+)?         # 2, 2.5
       (?:\s*/\s*\d+)?       # optional /3
     |[{_FRACTIONS}]         # or unicode fraction
     |(?:\d+\s+[{_FRACTIONS}])  # 2 ½
    )
    (?!\w)
    """,
    re.VERBOSE,
)

# parentheticals like "(divided)", "(optional)"
_PARENS_RE  = re.compile(r"\s*\([^)]*\)")
# dashes/bullets/×/*
_DASHES_RE  = re.compile(r"[-–—•×*/]+")

# stray punctuation/periods at edges or after commas
_EDGE_PUNCT_RE = re.compile(r"""(^[^\w]+|[^\w]+$)""")
# multiple spaces/commas
_MULTI_SPACE_RE = re.compile(r"\s+")
_MULTI_COMMA_RE = re.compile(r"\s*,\s*")
_COMMA_SEMI_SPLIT_RE = re.compile(r"\s*[;,]\s*")
_ALT_SPLIT_RE = re.compile(
    r"""
    \s+(?:or|\|)\s+              # 'or' or '|'
    |                            # OR
    (?<!\d)\s*/\s*(?!\d)         # a slash not between digits (so NOT the '1/2' in quantities)
    """,
    re.I | re.VERBOSE
)

_DESCRIPTORS = {
    # --- common prep/state ---
    "blanched","roasted","toasted","shelled","unshelled","cleaned","washed",
    "rinsed","dried","draining","drained","thawed","thaw","peeled","unpeeled",
    "seeded","deseeded","pitted","unpitted","trimmed","stemmed","beaten",
    "separated","divided","broken","halved","quartered","minced","chopped",
    "coarsely","finely","roughly","shredded","sliced","diced","crushed",
    "mashed","grated","ground","powdered","sifted","melted","softened",
    "mixed","stirred","beaten","combined","whipped","frozen","defrosted",
    "prepared","preheated","cooked","uncooked","baked","boiled","fried",
    "sauteed","sautéed","roasted","steamed","sterilized","sterilised","cubed", "cut up",
    "simmered down from", "cut in inch", "deboned cut in size",
    # --- physical/size ---
    "large","larger","small","sm","medium","md","med","tiny","extra","extra-large","pt",
    "very","thick","thin","thinly","thickly","whole","half","piece","pieces","10", "slightly",
    # --- flavor/temperature ---
    "fresh","freshly","ripe","unripe","hot","cold","warm","lukewarm","room",
    "temperature","cool","cooling","chilled","refrigerated",
    # --- packaging/count nouns ---
    "clove","cloves","piece","pieces","package","packages","pkg","can","cans",
    "jar","bottle","bag","packet","pack","box","stick","sticks",
    # --- qualifiers/noise words ---
    "about","approx","approximately","rough","roughly","around","and","or","&",
    "to","taste","plus","more","only","as","needed","desired","optional",
    "carefully","well","lightly","heavily","fully","firm","loose","packed",
    "heaping","level","scant","generous","new","flat","healthy","pinch",
    # --- redundant units or pseudo-units ---
    "dash","pinch","sprig","sprigs","leaf","leaves","bunch","bunches",
    "handful","handfuls","slice","slices","cube","cubes","ring","rings",
    "drop","drops",
    # --- others ---
    "coarse","bite","of","fillet","brand","chunk","container",
    "package","pkg","box","can","jar","stick","cup","qt","gal"
}

_PHRASE_DROP = [
    # serving/purpose
    r"\bto\s+taste\b",
    r"\bplus\s+more\b",
    r"\bfor\s+(serving|garnish|glaze|basting|brushing)\b",
    r"\bto\s+(serve|garnish)\b",
    r"\bfor\s+(the\s+)?(dressing|sauce)\b",
    # cutting / prep
    r"\bcut\s+into\s+(inch\s+)?(pieces?|wedges?|strips?|chunks?|slices?)\b",
    r"\bcored\s+and\s+cut\s+into\b",
    r"\bcut\s+lengthwise\b",
    r"\bcut\s+crosswise\b",
    r"\btrimmed\s+and\s+(cut|chopped)\b",
    r"\bchopped\s+and\s+(drained|rinsed)\b",
    r"\bpeeled\s+and\s+(chopped|sliced|diced)\b",
    r"\bdeseeded\s+and\b",
    r"\bdrained\s+and\s+rinsed\b",
    r"\bthawed\s+and\s+drained\b",
    r"\bcarefully\s+washed,\s*dried\s+and\b",
    r"\bwashed\s+and\s+dried\b",
    r"\bblanched\s+and\s+peeled\b",
    # quantity / flavor intensity
    r"\bheaping\s+tablespoons?\b",
    r"\bheaping\s+teaspoons?\b",
    r"\bscant\s+cup\b",
    r"\bto\s+cover\b",
    # state / form
    r"\bjuice\s+only\b",
    r"\bzest\s+only\b",
    r"\bpitted\s+(and\s+)?chopped\b",
    r"\broasted\s+and\s+shelled\b",
    r"\bground\s+and\s+toasted\b",
    # weird textual leftovers from OCR or prose
    r"\bi\s+like\s+peel\s+mine\b",
    r"\bhealthy\s+pinch\b",
    r"\bdash\s+of\b",
]
_PHRASE_DROP_RE = re.compile("|".join(_PHRASE_DROP), flags=re.I)

# --- Stuck alternatives like "butter margarine", "oil shortening" ---
_STUCK_ALT_PATTERNS = [
    (r"\b(butter)\s+margarine\b", r"\1 or margarine"),
    (r"\b(oil)\s+shortening\b", r"\1 or shortening"),
    (r"\b(sugar)\s+honey\b", r"\1 or honey"),
    (r"\b(chocolate)\s+cocoa\b", r"\1 or cocoa"),
    (r"\b(vinegar)\s+lemon\s+juice\b", r"\1 or lemon juice"),
    (r"\b(cream)\s+milk\b", r"\1 or milk"),
    (r"\b(cashew)\s+peanut\b", r"\1 or peanut"),
    (r"\b(white\s+lily)\s+", ""),   # drop White Lily brand prefix
    (r"\b(flounder)\s+pollack\s+fillet\b", r"\1 or pollack fillet"),
]

_BRAND_DROP = [
    r"\bwhite\s+lily\b",
    r"\beagle\s+brand\b",
    r"\bcampbell[’']?s?\b",
    r"\bsmucker[’']?s?\b",
    r"\bchicken\s+of\s+the\s+sea\b",
    r"\bbetty\s+cro(?:cker)?\b",
    r"\bjell[-\s]?o\b",
    r"\bvelveeta\b",
    r"\bphiladelphia\b",
]
_BRAND_DROP_RE = re.compile("|".join(_BRAND_DROP), re.I)

# --- spelling fixes for common variants / OCR ---
_SPELLFIX = {
    "margerine": "margarine",
    "margerines": "margarine",
    "margerin": "margarine",
    "margerin": "margarine",
    "confectioners sugar": "confectioners' sugar",
}

# --- canonical alias map (collapse noisy variants) ---
# Only put things here you truly want to be the SAME ingredient.
_ALIAS = {
    "soft margarine": "margarine",
    "stick margarine": "margarine",
    "butter margarine": "margarine",
    "block margarine": "margarine",
    "chicken style": "chicken style seasoning",
    "chicken" : "chicken breast",
    "10x sugar": "powdered sugar",
    "10-x sugar": "powdered sugar",
    "10 × sugar": "powdered sugar",
    "lean pork chop": "pork chop",
    "doz okra": "okra",
    "feet": "pig feet",
    "bottled low calorie italian salad dressing": "italian salad dressing",
    "egg": "eggs",
    "egg white": "egg whites",
    "hard egg": "eggs",
    "bottl lemon lime soda" : "lemon lime soda",
    "bottl ginger ale" : "ginger ale",
    "icing sugar": "powdered sugar",
    "confectioners' sugar": "powdered sugar",
    "noodl": "noodles",
    "chinese noodl": "noodles",
    "vegetabl" : "vegetables",
    "qt milk" : "milk",
    "nut": "nuts",
    "potat" : "potatoes",
    "commercial barbecue sauce": "barbecue sauce",
}

def _apply_spellfix(s: str) -> str:
    return _SPELLFIX.get(s, s)

def _apply_alias(s: str) -> str:
    return _ALIAS.get(s, s)






# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Scraps-LLM preprocessing (streaming)")
    p.add_argument("--raw", type=Path, required=True, help="Path to raw file (jsonl or csv)")
    p.add_argument("--format", choices=["jsonl", "csv"], required=True, help="Input format")
    p.add_argument("--outdir", type=Path, default=Path("data/processed"), help="Output directory")

    # split
    p.add_argument("--seed", type=int, default=42, help="Split seed")
    p.add_argument("--train", type=float, default=0.9, help="Train ratio")
    p.add_argument("--val",   type=float, default=0.05, help="Val ratio")
    p.add_argument("--test",  type=float, default=0.05, help="Test ratio")

    # columns (CSV)
    p.add_argument("--ing-col", type=str, default="ingredients", help="CSV column name for ingredients")
    p.add_argument("--rec-col", type=str, default="directions",  help="CSV column name for directions/recipe")
    p.add_argument("--title-col", type=str, default="title",     help="CSV column for title (optional)")
    p.add_argument("--ner-col", type=str, default="NER",         help="CSV column for NER list (optional)")

    # behavior
    p.add_argument("--lower-ingredients", action="store_true", help="Lowercase the ingredients text used for training")
    p.add_argument("--keep-quantities",   action="store_true", default=True,
                   help="KEEP numbers/units in the training 'ingredients' field (default ON)")
    p.add_argument("--no-keep-quantities", dest="keep_quantities", action="store_false",
                   help="Strip numbers/units from the training 'ingredients' field")
    p.add_argument("--ner-mode", choices=["union","intersection","ner_only","raw"], default="union",
                   help="How to combine raw ingredients with NER")

    return p.parse_args()

# -----------------------------
# Small helpers
# -----------------------------
def _expand_items(seq: List[str]) -> List[str]:
    """
    Expand items like 'butter or margarine' into separate entries
    before normalization, so we get 'butter' and 'margarine'.
    """
    out: List[str] = []
    for s in seq:
        if not s:
            continue
        s = _pre_normalize_typos(s)          # rewrite 'butter margarine' -> 'butter or margarine', drop brands, etc.
        s = _PARENS_RE.sub("", s)
        parts = split_alternatives(s) or [s] # split on 'or' / '|' / non-numeric '/'
        out.extend(p.strip() for p in parts if p.strip())
    return out

def _collapse_adjacent_dupes(text: str) -> str:
    """Collapse repeating words or tokens like 'butter butter' or 'soup soup'."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    seen = set()
    result = []
    for p in parts:
        norm = normalize_ingredient_no_qty(p)
        if norm in seen:
            continue
        seen.add(norm)
        result.append(p)
    return ", ".join(result)

def _singularize_word(w: str) -> str:
    # common English rules, conservative
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"        # berries -> berry
    if len(w) > 3 and w.endswith("ves"):
        return w[:-3] + "f"        # leaves -> leaf (will be dropped as descriptor)
    # avoid chopping classes like 'glass', 'boss', 'press'
    if len(w) > 3 and w.endswith("es") and not any(
        w.endswith(suf) for suf in ("ses","xes","zes","ches","shes")
    ):
        return w[:-2]              # tomatoes -> tomato
    if len(w) > 2 and w.endswith("s") and not w.endswith(("ss","us")):
        return w[:-1]              # onions -> onion
    return w

def split_alternatives(s: str) -> list[str]:
    """
    Split a single ingredient phrase into alternatives on 'or', '/', '|'
    outside of parentheses. Simple version (since we strip parens earlier).
    """
    s = s.strip()
    if not s:
        return []
    # after cleaning, parentheses are generally removed; use a simple split
    parts = _ALT_SPLIT_RE.split(s)
    return [p.strip() for p in parts if p.strip()]


def normalized_components_for_entry(raw: str) -> set[str]:
    raw = _pre_normalize_typos(raw)  # rewrite "butter margarine" -> "butter or margarine"
    comps = split_alternatives(_PARENS_RE.sub("", raw)) or [raw]
    out = set()
    for c in comps:
        norm = normalize_ingredient_no_qty(c)
        if norm:
            out.add(norm)
    return out


def parse_listish(s: str) -> List[str]:
    """Accept JSON-like '["a","b"]' or fallback to splitting on commas/semicolons."""
    s = (s or "").strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
    if s:
        return [p for p in _COMMA_SEMI_SPLIT_RE.split(s) if p.strip()]
    return []


def _strip_edge_punct(s: str) -> str:
    s = s.strip()
    s = _EDGE_PUNCT_RE.sub("", s)
    return s.strip()

def _collapse_commas(s: str) -> str:
    # normalize comma spacing and collapse repeats
    s = _MULTI_COMMA_RE.sub(", ", s)
    while ", ," in s:
        s = s.replace(", ,", ",")
    return s.strip(" ,")

def _pre_normalize_typos(s: str) -> str:
    """Fix common OCR/abbreviation errors and stuck alternatives like 'butter margarine'."""
    s = _BRAND_DROP_RE.sub("", s)
    # --- expand 'stuck' alternative pairs ---
    for pat, sub in _STUCK_ALT_PATTERNS:
        s = re.sub(pat, sub, s, flags=re.I)

    repl = [
        (r"\be\s*tract\b", "extract"),
        (r"\bcreme\s+de\s+cocoa\b", "creme de cacao"),
        (r"\bpcz\b", "pcs"),
        (r"\bmi\b", "mix"),
        (r"\blow\s*fat\b", "lowfat"),
    ]
    for pat, sub in repl:
        s = re.sub(pat, sub, s, flags=re.I)
    return s



def normalize_ingredient_no_qty(name: str) -> str:
    """
    Canonicalize an ingredient name WITHOUT quantities/units/descriptors.
    """
    s = (name or "").lower()
    s = _pre_normalize_typos(s)
    s = re.sub(r"\b10\s*[x×\-]?\s*sugar\b", "powdered sugar", s)
    s = _PARENS_RE.sub("", s)           # remove (optional) etc.
    s = _PHRASE_DROP_RE.sub(" ", s)    # remove multi-word descriptors
    s = _NUM_RE.sub("", s)              # remove numbers/fractions
    s = _UNIT_RE.sub("", s)             # remove units (cup, c., tsp., etc.)
    s = _DASHES_RE.sub(" ", s)          # replace dashes/bullets with space
    s = re.sub(r"[.;:]+", " ", s)       # remove lingering punctuation
    s = _MULTI_SPACE_RE.sub(" ", s).strip()

    # token-wise cleanup
    toks = []
    for t in s.split():
        if t in _DESCRIPTORS:
            continue                    # drop descriptors like 'leaves'
        t = _singularize_word(t)        # conservative singularize
        toks.append(t)

    s = " ".join(toks).strip()
    s = _strip_edge_punct(s)
    s = _apply_spellfix(s)
    s = _apply_alias(s)
    # simple repairs for truncated OCR or plural forms
    # if s.endswith("vegetabl"):
    #     s = s.replace("vegetabl", "vegetable")
    # if s.endswith("potat"):
    #     s = s.replace("potat", "potato")

    return s

def _has_qty_or_unit(s: str) -> bool:
    if _NUM_RE.search(s):
        return True
    return bool(_UNIT_RE.search(s))

def _display_score(s: str) -> tuple[int, int]:
    """
    Score display strings so 'richer' ones win:
      1) presence of qty/unit (1 or 0)
      2) length (chars)
    Higher is better.
    """
    return (1 if _has_qty_or_unit(s) else 0, len(s))

def _better_display(old: str | None, new: str) -> str:
    if not old:
        return new
    return new if _display_score(new) > _display_score(old) else old

def dedupe_by_norm_display(csv_like: str) -> str:
    csv_like = _pre_normalize_typos(csv_like)  # <-- NEW
    items = [x.strip() for x in _COMMA_SEMI_SPLIT_RE.split(csv_like) if x.strip()]
    best_by_norm = {}
    order = []
    for disp in items:
        key = normalize_ingredient_no_qty(disp)
        if not key:
            continue
        keep = _better_display(best_by_norm.get(key), disp)
        if key not in best_by_norm:
            best_by_norm[key] = keep
            order.append(key)
        else:
            best_by_norm[key] = keep
    return ", ".join(best_by_norm[k] for k in order)



def merge_ingredients_lists(raw_list, ner_list, mode="union", keep_quantities=False) -> str:
    """
    Build the training 'ingredients' text.
    - keep_quantities=True: Keep quantities/units 
    - keep_quantities=False: Strip numbers/units, deduplicate, normalize, merge per mode.
    """

    # -------------------------
    # KEEP QUANTITIES = TRUE
    # -------------------------
    
    def clean_display(s: str) -> str:
        s = _pre_normalize_typos(s)         # <-- run before clean
        s = clean(s)
        s = _PARENS_RE.sub("", s)
        s = _collapse_commas(s.replace(" .", "."))
        s = _strip_edge_punct(s.replace(".", ""))  # drop stray trailing dots
        return s



    # explode any CSV cell that came in as a single comma/semicolon string
        def _explode(seq: List[str]) -> List[str]:
            out = []
            for s in seq:
                if not s:
                    continue
                parts = _COMMA_SEMI_SPLIT_RE.split(s)
                out.extend(p.strip() for p in parts if p.strip())
            return out

        raw_list = _explode(raw_list)
        ner_list = _explode(ner_list)

    # choose best display per normalized ingredient
        best_by_norm: Dict[str, str] = {}
        order: List[str] = []  # preserve first-seen order of normalized keys

        for s in raw_list:
            comps = normalized_components_for_entry(s)  # {"cucumber"} or {"butter","margarine"}
            if not comps:
                continue
            disp = clean_display(s)
            if not disp:
                continue

        # If it's an OR-line, we treat each component separately but use the same display
            for comp in comps:
                prev = best_by_norm.get(comp)
                cand = disp
                keep = _better_display(prev, cand)
                if prev is None:
                    best_by_norm[comp] = keep
                    order.append(comp)
                else:
                # upgrade if the candidate is better
                    if keep is not prev:
                        best_by_norm[comp] = keep

    # bring in NER-only items if we didn't see them in raw_list
        for s in ner_list:
            comps = normalized_components_for_entry(s)
            for comp in comps:
                if comp in best_by_norm:
                    continue
            # for NER-only, use the normalized name itself as display
                best_by_norm[comp] = comp
                order.append(comp)

   
        display_items = [best_by_norm[c] for c in order if best_by_norm[c]]
        out = _collapse_commas(", ".join(display_items))
        out = dedupe_by_norm_display(out)
        return out.lower()


    # -------------------------
    # KEEP QUANTITIES = FALSE
    # -------------------------
    # expand 'or' alternatives BEFORE normalization
    raw_items = _expand_items([x for x in raw_list if x])
    ner_items = _expand_items([x for x in ner_list if x])

    # normalize each alternative separately
    raw_norm = {normalize_ingredient_no_qty(x) for x in raw_items if x}
    ner_norm = {normalize_ingredient_no_qty(x) for x in ner_items if x}

    # merge per mode
    if mode == "union":
        merged = sorted((raw_norm | ner_norm) - {""})
    elif mode == "intersection":
        merged = sorted((raw_norm & ner_norm) - {""})
    elif mode == "ner_only":
        merged = sorted(ner_norm - {""})
    else:  # "raw"
        merged = sorted(raw_norm - {""})

    # 4) join; (optional) final safety nets (idempotent here)
    out = ", ".join(m for m in merged if m)
    out = dedupe_by_norm_display(out)  
    out = _collapse_adjacent_dupes(out)
    return out.lower()

def format_recipe(title: str, directions: List[str]) -> str:
    """Title + numbered steps (if title exists)."""
    steps = []
    for i, step in enumerate(directions, 1):
        t = clean(step)
        if t:
            steps.append(f"Step {i}: {t}")
    steps_text = "\n".join(steps)
    title = (title or "").strip()
    if title:
        return f"Title: {title}\n{steps_text}" if steps_text else f"Title: {title}"
    return steps_text

def build_row(ing_text: str, recipe_text: str) -> Dict:
    return {"ingredients": ing_text, "recipe": recipe_text}


# -----------------------------
# Deterministic streaming split
# -----------------------------
def _hash_to_split(key: str, seed: int, train=0.9, val=0.05, test=0.05) -> str:
    m = hashlib.md5()
    m.update((key + f"#{seed}").encode("utf-8"))
    r = int(m.hexdigest(), 16) / (1 << 128)
    if r < train:
        return "train"
    elif r < train + val:
        return "val"
    else:
        return "test"

# -----------------------------
# Streaming writers
# -----------------------------
def stream_csv(args):
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    f_train = (outdir / "train.jsonl").open("w", encoding="utf-8")
    f_val   = (outdir / "val.jsonl").open("w", encoding="utf-8")
    f_test  = (outdir / "test.jsonl").open("w", encoding="utf-8")

    try:
        with args.raw.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # parse columns
                title    = (row.get(args.title_col, "") or "").strip()
                ing_list = parse_listish(row.get(args.ing_col, ""))
                dir_list = parse_listish(row.get(args.rec_col, ""))

                # optional NER
                ner_list = parse_listish(row.get(args.ner_col, "")) if args.ner_col else []

                # training input ingredients
                ing_input = merge_ingredients_lists(
                    raw_list=ing_list,
                    ner_list=ner_list,
                    mode=args.ner_mode,
                    keep_quantities=args.keep_quantities
                )
                if args.lower_ingredients:
                    ing_input = ing_input.lower()



                # recipe target (Title + numbered steps)
                recipe_text = format_recipe(title, dir_list)

                if not ing_input or not recipe_text:
                    continue

                obj = build_row(ing_input, recipe_text)

                # choose bucket (deterministic)
                key = title or ing_input  # prefer title if present
                bucket = _hash_to_split(key, args.seed, args.train, args.val, args.test)
                fh = f_train if bucket == "train" else f_val if bucket == "val" else f_test
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        f_train.close(); f_val.close(); f_test.close()

def stream_jsonl(args):
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    f_train = (outdir / "train.jsonl").open("w", encoding="utf-8")
    f_val   = (outdir / "val.jsonl").open("w", encoding="utf-8")
    f_test  = (outdir / "test.jsonl").open("w", encoding="utf-8")

    try:
        for obj in read_jsonl(args.raw):
            # expect {"ingredients": "...", "recipe": "..."} (no NER)
            ing = clean(obj.get("ingredients", ""))
            rec = clean(obj.get("recipe", ""))
            if not ing or not rec:
                continue

            if args.lower_ingredients:
                ing = ing.lower()

            row = build_row(ing, rec)
            key = ing
            bucket = _hash_to_split(key, args.seed, args.train, args.val, args.test)
            fh = f_train if bucket == "train" else f_val if bucket == "val" else f_test
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        f_train.close(); f_val.close(); f_test.close()

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    # safety: ratios must sum to 1
    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f">>> Streaming preprocess from {args.raw} → {args.outdir}")

    if args.format == "csv":
        stream_csv(args)
    else:
        stream_jsonl(args)

    print("✅ Wrote: train.jsonl, val.jsonl, test.jsonl")

if __name__ == "__main__":
    main()
