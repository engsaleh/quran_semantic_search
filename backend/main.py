import os, json, hashlib, re
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Vector search
import faiss
from rank_bm25 import BM25Okapi

# Embedding backends
from sentence_transformers import SentenceTransformer
# Prefer Instructor if available
INSTR_AVAILABLE = False
try:
    from InstructorEmbedding import INSTRUCTOR
    INSTR_AVAILABLE = True
except Exception:
    INSTR_AVAILABLE = False

# Optional Arabic stemming
STEM_AVAILABLE = False
try:
    from tashaphyne.stemming import ArabicLightStemmer
    ArStem = ArabicLightStemmer()
    STEM_AVAILABLE = True
except Exception:
    STEM_AVAILABLE = False


# ---------- إعداد البيئة والمسارات ----------
load_dotenv()
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_ID = os.getenv("MODEL_ID", "hkunlp/instructor-large")  # Instructor
FALLBACK_MODEL_ID = os.getenv("FALLBACK_MODEL_ID", "intfloat/multilingual-e5-large")
TOPK = int(os.getenv("TOPK", "10"))
PORT = int(os.getenv("PORT", "8000"))

DATA_JSON = os.path.join(DATA_DIR, "quran.json")
TANZIL_XML = os.path.join(DATA_DIR, "quran-simple.xml")


# ---------- دوال المعالجة ----------
TASHKEEL_RE = re.compile(r"[ؗ-ًؚ-ْٗ-ٰٟۖ-ۭ]")
NON_LETTER_RE = re.compile(r"[^\w\s؀-ۿ]")

def normalize_ar(s: str) -> str:
    s = s or ""
    s = TASHKEEL_RE.sub('', s)
    s = s.replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ٱ','ا')
    s = s.replace('ى','ي').replace('ة','ه').replace('ؤ','و').replace('ئ','ي')
    s = NON_LETTER_RE.sub(' ', s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def stem_tokens(tokens: List[str]) -> List[str]:
    if not STEM_AVAILABLE:
        return tokens
    out = []
    for w in tokens:
        try:
            ArStem.light_stem(w)
            stem = ArStem.get_stem() or w
        except Exception:
            stem = w
        out.append(stem)
    return out


# ---------- تحويل Tanzil XML إلى JSON ----------
def convert_tanzil_xml(xml_path: str, out_path: str) -> bool:
    if not os.path.exists(xml_path):
        return False
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        verses = []
        for sura in root.findall("sura"):
            sname = sura.get("name")
            snum = int(sura.get("index"))
            for aya in sura.findall("aya"):
                verses.append({
                    "surah": f"{snum} - {sname}",
                    "verse_number": int(aya.get("index")),
                    "ayah": aya.get("text", "").strip(),
                })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(verses, f, ensure_ascii=False, indent=2)
        print(f"✅ Converted Tanzil XML to JSON with {len(verses)} verses")
        return True
    except Exception as e:
        print("⚠️ Tanzil conversion failed:", e)
        return False


# ---------- تحميل البيانات ----------
if not os.path.exists(DATA_JSON):
    if not convert_tanzil_xml(TANZIL_XML, DATA_JSON):
        raise FileNotFoundError("Place data/quran.json or data/quran-simple.xml before running.")

print(f"📘 Loading data from: {DATA_JSON}")
with open(DATA_JSON, "r", encoding="utf-8") as f:
    DATA: List[Dict[str, Any]] = json.load(f)

raw_ayahs: List[str] = [d.get("ayah", "") for d in DATA]


# ---------- معالجة النصوص ----------
def preprocess_text(s: str) -> str:
    s = normalize_ar(s)
    toks = s.split()
    toks = stem_tokens(toks)
    return " ".join(toks)

corpus_texts: List[str] = [preprocess_text(t) for t in raw_ayahs]


# ---------- تحميل نموذج التضمين ----------
print(f"🧠 Loading embedding model: {MODEL_ID}")
if INSTR_AVAILABLE:
    try:
        instr = INSTRUCTOR(MODEL_ID)
        def embed_passages(texts: List[str]) -> np.ndarray:
            prompts = [["Represent the passage for retrieving relevant Arabic verses:", t] for t in texts]
            v = instr.encode(prompts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
            return v.astype(np.float32)
        def embed_query(text: str) -> np.ndarray:
            v = instr.encode([["Represent the question for retrieving relevant Arabic verses:", text]],
                             show_progress_bar=False, normalize_embeddings=True)
            return v.astype(np.float32)
        EMBED_NAME = MODEL_ID
    except Exception as e:
        print("⚠️ Instructor failed, falling back to SentenceTransformer:", e)
        INSTR_AVAILABLE = False

if not INSTR_AVAILABLE:
    embed_model = SentenceTransformer(FALLBACK_MODEL_ID)
    def embed_passages(texts: List[str]) -> np.ndarray:
        arr = embed_model.encode([f"passage: {t}" for t in texts], show_progress_bar=True, normalize_embeddings=True)
        return arr.astype(np.float32)
    def embed_query(text: str) -> np.ndarray:
        arr = embed_model.encode([f"query: {text}"], show_progress_bar=False, normalize_embeddings=True)
        return arr.astype(np.float32)
    EMBED_NAME = FALLBACK_MODEL_ID


# ---------- إنشاء أو تحميل التضمينات ----------
def compute_hash_key(texts: List[str], model_id: str) -> str:
    m = hashlib.sha1()
    m.update(model_id.encode("utf-8"))
    for t in texts[:1000]:
        m.update(t.encode("utf-8", "ignore"))
    return m.hexdigest()[:12]

hash_key = compute_hash_key(corpus_texts, EMBED_NAME)
EMBED_PATH = os.path.join(CACHE_DIR, f"emb_{hash_key}.npy")
INDEX_PATH = os.path.join(CACHE_DIR, f"faiss_{hash_key}.index")

if os.path.exists(EMBED_PATH):
    print(f"⚡ Loading cached embeddings: {EMBED_PATH}")
    corpus_embeddings = np.load(EMBED_PATH)
else:
    print("⚙️ Computing embeddings (first run)...")
    corpus_embeddings = embed_passages(corpus_texts)
    np.save(EMBED_PATH, corpus_embeddings)
    print(f"💾 Saved embeddings to {EMBED_PATH}")

faiss.normalize_L2(corpus_embeddings)

def build_or_load_faiss(emb: np.ndarray):
    dim = emb.shape[1]
    if os.path.exists(INDEX_PATH):
        print(f"⚡ Loading FAISS index: {INDEX_PATH}")
        return faiss.read_index(INDEX_PATH)
    nlist = min(2048, max(256, len(emb)//40))
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(emb)
    index.add(emb)
    faiss.write_index(index, INDEX_PATH)
    print(f"💾 Saved FAISS index (IVF, nlist={nlist}) to {INDEX_PATH}")
    return index

faiss_index = build_or_load_faiss(corpus_embeddings)


# ---------- BM25 ----------
tokenized_docs = [stem_tokens(t.split()) for t in corpus_texts]
bm25 = BM25Okapi(tokenized_docs)


# ---------- نماذج الإدخال ----------
class SearchRequest(BaseModel):
    query: str
    k: int = TOPK
    surah_filter: Optional[List[int]] = None
    add_context: bool = False
    semantic_only: bool = False


# ---------- البحث ----------
def surah_number_from_label(lbl: str) -> int:
    try:
        return int(str(lbl).split("-")[0].strip())
    except:
        return -1

def hybrid_candidates(query: str, n_sem: int = 200, n_bm: int = 200):
    q_prep = preprocess_text(query)
    q_emb = embed_query(q_prep)
    sims, idxs = faiss_index.search(q_emb, n_sem)
    idxs_sem, cos_sem = idxs[0].tolist(), sims[0].tolist()

    q_tokens = stem_tokens(q_prep.split())
    bm_scores = bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(corpus_texts))
    idxs_bm = np.argsort(bm_scores)[::-1][:n_bm].tolist() if len(bm_scores) else []

    seen, cand = set(), []
    max_bm = float(max(bm_scores)) if len(bm_scores) else 1.0
    max_bm = max_bm if max_bm > 0 else 1.0
    cos_map = {idxs_sem[i]: float(cos_sem[i]) for i in range(len(idxs_sem))}

    for i in idxs_sem + idxs_bm:
        if i in seen:
            continue
        seen.add(i)
        bm_norm = float(bm_scores[i] / max_bm) if len(bm_scores) else 0.0
        cos = float(cos_map.get(i, 0.0))
        c = {**DATA[i], "idx": int(i), "cos": cos, "bm25n": bm_norm}
        cand.append(c)

    alpha = 0.8 if len(q_tokens) < 4 else 0.6
    cand.sort(key=lambda c: alpha * c["cos"] + (1 - alpha) * c["bm25n"], reverse=True)
    return cand


# ---------- تطبيق FastAPI ----------
app = FastAPI(title="Quran & Hadith Semantic Search (Stemming + Instructor)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": EMBED_NAME,
        "items": len(DATA),
        "stemming": STEM_AVAILABLE,
        "instructor": INSTR_AVAILABLE,
    }


@app.post("/search")
async def search(req: SearchRequest):
    cand = hybrid_candidates(req.query)
    if req.surah_filter:
        sset = set(req.surah_filter)
        cand = [c for c in cand if surah_number_from_label(c.get("surah", "")) in sset]

    if req.semantic_only:
        cand = sorted(cand, key=lambda x: x["cos"], reverse=True)

    out = []
    for c in cand[:req.k]:
        idx = c["idx"]
        item = {
            "surah": c.get("surah"),
            "verse_number": c.get("verse_number"),
            "ayah": c.get("ayah"),
            "scores": {
                "cos": round(float(c.get("cos", 0)), 3),
                "bm25": round(float(c.get("bm25n", 0)), 3),
            },
            "why": "Hybrid (semantic+bm25)" if not req.semantic_only else "Semantic-only",
        }
        if req.add_context:
            prev_ayah = DATA[idx-1]["ayah"] if idx-1 >= 0 else None
            next_ayah = DATA[idx+1]["ayah"] if idx+1 < len(DATA) else None
            item["context"] = {"prev": prev_ayah, "next": next_ayah}
        out.append(item)

    return {"query": req.query, "results": out, "k": req.k}


# ---------- قائمة السور ----------
@app.get("/surahs")
async def list_surahs():
    """إرجاع قائمة السور من البيانات الحالية"""
    seen, surahs = set(), []
    for row in DATA:
        s = row.get("surah", "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        try:
            num, name = s.split(" - ", 1)
            surahs.append({"id": int(num.strip()), "name": name.strip()})
        except ValueError:
            continue
    surahs.sort(key=lambda x: x["id"])
    return {"surahs": surahs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
