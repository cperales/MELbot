#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lite_file_qa.py — "ChatGPT File Q&A"-style retrieval over your own files.

Key features (vs your original script):
- Multi-file ingestion (PDF, DOCX, TXT/MD) with per-page/per-file metadata
- Configurable chunking with overlap and sentence-awareness
- Batched embeddings + normalized vectors for cosine/IP search
- FAISS index with persistent vectors + a JSONL sidecar for chunk metadata
- MMR diversification on retrieval to reduce redundancy
- Prompt assembly with numbered snippets + transparent source list
- Simple generator for Ollama (default) — plug in your own if needed

Usage (example):
    python lite_file_qa.py ingest data/docs
    python lite_file_qa.py ask "¿Cuáles son las obligaciones del arrendador?"
    python lite_file_qa.py ask --k 6 --mmr 0.4 "Summarize key risks across all docs"

Requirements: transformers, torch, faiss-cpu, numpy, PyPDF2, python-docx (optional for DOCX)

Author: Adapted for Carlos
"""
import argparse
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss

# --------- Embedding backend (defaults to your MEL model) ---------
import torch
from transformers import AutoTokenizer, AutoModel

DEFAULT_MODEL_ID = os.environ.get("EMBED_MODEL", "IIC/MEL")  # Spanish-legal encoder by default

class Embedder:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None, max_batch: int = 16):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.enc = AutoModel.from_pretrained(model_id)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
        self.device = torch.device(device)
        self.enc = self.enc.to(self.device)
        self.max_batch = max_batch

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self.max_batch):
            batch = texts[i:i + self.max_batch]
            toks = self.tok(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            out = self.enc(**toks).last_hidden_state  # [B, T, H]
            mask = toks.attention_mask.unsqueeze(-1)
            emb = (out * mask).sum(dim=1) / mask.sum(dim=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy()
            vecs.append(emb)
        return np.vstack(vecs)

# --------- IO helpers ---------
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    import docx  # python-docx
except Exception:
    docx = None

def read_pdf(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 not installed. pip install PyPDF2")
    out = []
    with open(path, 'rb') as f:
        pdf = PdfReader(f)
        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                out.append((text, {"source": str(path), "type": "pdf", "page": i}))
    return out

def read_docx(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    if docx is None:
        raise RuntimeError("python-docx not installed. pip install python-docx")
    document = docx.Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs)
    if text.strip():
        return [(text, {"source": str(path), "type": "docx"})]
    return []

def read_txt(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text.strip():
        return [(text, {"source": str(path), "type": "text"})]
    return []

def load_files(root: Path) -> List[Tuple[str, Dict[str, Any]]]:
    items: List[Tuple[str, Dict[str, Any]]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        try:
            if ext in [".pdf"]:
                items.extend(read_pdf(p))
            elif ext in [".docx"]:
                items.extend(read_docx(p))
            elif ext in [".txt", ".md"]:
                items.extend(read_txt(p))
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}", file=sys.stderr)
    return items

# --------- Chunking ---------
_SENT_SPLIT = re.compile(r'(?<=[\.\?\!\:\;])\s+')

def chunk_text(text: str, max_len: int = 800, overlap: int = 120) -> List[str]:
    """
    Simple sentence-aware chunking:
    - split by sentence-like boundaries
    - pack sentences into ~max_len character chunks
    - add tail overlap to preserve context continuity
    """
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur_len + len(s) > max_len and cur:
            chunks.append(" ".join(cur))
            # overlap tail
            tail = " ".join(cur)[-overlap:]
            cur, cur_len = ([tail] if tail else []), len(tail)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks or ([text] if text else [])

# --------- Metadata store (JSONL) ---------
@dataclass
class ChunkRec:
    id: str
    text: str
    meta: Dict[str, Any]

class Store:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.workdir / "chunks.faiss"
        self.meta_path = self.workdir / "chunks.jsonl"
        self.dim: Optional[int] = None
        self.index: Optional[faiss.Index] = None
        self._count = 0

    def _load_meta(self) -> List[ChunkRec]:
        if not self.meta_path.exists():
            return []
        out = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                out.append(ChunkRec(**j))
        self._count = len(out)
        return out

    def _append_meta(self, recs: List[ChunkRec]):
        with self.meta_path.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        self._count += len(recs)

    def _load_index(self, dim: int):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.dim = dim
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.dim = dim

    def _save_index(self):
        assert self.index is not None
        faiss.write_index(self.index, str(self.index_path))

    @property
    def count(self) -> int:
        return self._count

# --------- Ingestion ---------
def ingest_folder(folder: Path, store: Store, embedder: Embedder, max_len=800, overlap=120, batch=64):
    docs = load_files(folder)
    if not docs:
        print("No text found to ingest.")
        return

    # Prepare chunks + metadata
    texts: List[str] = []
    recs: List[ChunkRec] = []

    for raw_text, meta in docs:
        chunks = chunk_text(raw_text, max_len=max_len, overlap=overlap)
        for ch in chunks:
            cid = str(uuid.uuid4())
            meta_enriched = dict(meta)
            meta_enriched["chunk_id"] = cid
            meta_enriched["chars"] = len(ch)
            recs.append(ChunkRec(id=cid, text=ch, meta=meta_enriched))
            texts.append(ch)

    # Embed in batches
    embs = []
    for i in range(0, len(texts), batch):
        E = embedder.encode(texts[i:i+batch])
        embs.append(E)
    X = np.vstack(embs).astype("float32")

    # FAISS index (IP with normalized vectors -> cosine similarity)
    store._load_index(X.shape[1])
    # Add vectors
    store.index.add(X)
    store._save_index()
    # Append metadata JSONL
    store._append_meta(recs)
    print(f"Ingested {len(recs)} chunks from {len(docs)} items. Total chunks: {store.count}")

# --------- Retrieval with optional MMR ---------
def _mmr(doc_vecs: np.ndarray, query_vec: np.ndarray, k: int, lambda_mult: float = 0.5) -> List[int]:
    """
    Maximal Marginal Relevance over cosine similarities.
    doc_vecs: [N, D] normalized
    query_vec: [D]
    """
    N = doc_vecs.shape[0]
    sims = doc_vecs @ query_vec  # [N]
    selected = []
    candidates = list(range(N))
    while candidates and len(selected) < k:
        if not selected:
            i = int(np.argmax(sims[candidates]))
            selected.append(candidates.pop(i))
            continue
        # diversity term: similarity to already selected
        sel_vecs = doc_vecs[selected]
        diversity = sel_vecs @ doc_vecs[candidates].T  # [len(selected), len(candidates)]
        max_div = diversity.max(axis=0)  # [len(candidates)]
        mmr_score = lambda_mult * sims[candidates] - (1 - lambda_mult) * max_div
        i = int(np.argmax(mmr_score))
        selected.append(candidates.pop(i))
    return selected

def retrieve(query: str, store: Store, embedder: Embedder, k: int = 5, ef: int = 50, mmr: Optional[float] = 0.35):
    # Load meta
    meta_list = store._load_meta()
    if not meta_list or store.index is None:
        # lazy-load index using recorded dim (fallback)
        # We read index if present
        if store.index_path.exists():
            store.index = faiss.read_index(str(store.index_path))
        else:
            raise RuntimeError("Index empty. Run 'ingest' first.")

    # Build id -> rec
    id_to_rec = {r.id: r for r in meta_list}

    # Query embedding
    q = embedder.encode([query]).astype("float32")
    # FAISS search (retrieve ef candidates to allow MMR filtering later)
    ef = max(ef, k)
    D, I = store.index.search(q, ef)
    # Map FAISS rows to metadata lines
    # We assume index vectors are in the same order as jsonl lines appended.
    hits: List[Tuple[ChunkRec, float, int]] = []
    for rank, idx in enumerate(I[0]):
        if idx < 0: 
            continue
        rec = meta_list[idx]
        hits.append((rec, float(D[0][rank]), idx))

    # Optional MMR re-ranking over the ef candidates
    if mmr is not None:
        # We need vectors for ef candidates to compute diversity; re-embed texts
        cand_texts = [h[0].text for h in hits]
        V = embedder.encode(cand_texts).astype("float32")
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        qv = q[0] / (np.linalg.norm(q[0]) + 1e-12)
        order = _mmr(V, qv, k=k, lambda_mult=1.0 - float(mmr))  # invert scale: higher mmr -> more diversity
        hits = [hits[i] for i in order]
    else:
        hits = hits[:k]

    return hits[:k]

# --------- Generator (Ollama by default) ---------
import requests
import openai
from dotenv import load_dotenv

def generate_ollama(prompt: str, model: str = None, base_url: str = None) -> str:
    model = model or os.environ.get("OLLAMA_MODEL", "jobautomation/OpenEuroLLM-Spanish")
    base_url = base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    resp = requests.post(f"{base_url}/api/generate",
                         json={"model": model, "prompt": prompt, "stream": False},
                         timeout=120)
    resp.raise_for_status()
    j = resp.json()
    return j.get("response", "")

def generate(prompt: str, **kwargs) -> str:
    model = "gpt-5-nano-2025-08-07"
    print(f"Model: {model}")
    load_dotenv()
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --------- Prompt assembly ---------
SYS_PROMPT = """Eres un asistente servicial. Usa SOLO los fragmentos de contexto proporcionados para responder.
Si la respuesta es incierta o falta información, di que no tienes suficiente información.
Mantén las respuestas concisas y cita los fragmentos con números como [1], [2] en el texto cuando corresponda.
"""

def build_prompt(question: str, hits: List[Tuple[ChunkRec, float, int]], lang: str = "auto") -> Tuple[str, List[Dict[str, Any]]]:
    # Prepare numbered snippets
    lines = []
    source_cards = []  # for transparent listing back to user
    for j, (rec, score, _) in enumerate(hits, start=1):
        src = rec.meta.get("source", "")
        pg = rec.meta.get("page", None)
        loc = f"{Path(src).name}" + (f":p{pg}" if pg else "")
        lines.append(f"[{j}] ({loc})\n{rec.text}")
        source_cards.append({
            "n": j,
            "source": src,
            "page": pg,
            "score": round(float(score), 4),
            "chunk_id": rec.id
        })

    user_prompt = f"""{SYS_PROMPT}

QUESTION:
{question}

CONTEXT SNIPPETS:
{chr(10).join(lines)}
"""
    return user_prompt, source_cards

# --------- High-level QA ---------
def answer(question: str, store: Store, embedder: Embedder, k=5, ef=60, mmr=0.35, model=None) -> Dict[str, Any]:
    hits = retrieve(question, store, embedder, k=k, ef=ef, mmr=mmr)
    prompt, sources = build_prompt(question, hits)
    text = generate(prompt, model=model)
    return {"answer": text, "sources": sources}

# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser(description="Lite File Q&A")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_ing = sub.add_parser("ingest", help="Ingest a folder of documents")
    ap_ing.add_argument("folder", type=str, help="Folder with files (pdf, docx, txt, md)")
    ap_ing.add_argument("--workdir", type=str, default="data/index", help="Where to store FAISS/JSONL")
    ap_ing.add_argument("--max-len", type=int, default=800)
    ap_ing.add_argument("--overlap", type=int, default=120)

    ap_ask = sub.add_parser("ask", help="Ask a question against the index")
    ap_ask.add_argument("question", type=str)
    ap_ask.add_argument("--workdir", type=str, default="data/index")
    ap_ask.add_argument("--k", type=int, default=5)
    ap_ask.add_argument("--ef", type=int, default=60)
    ap_ask.add_argument("--mmr", type=float, default=0.35)
    ap_ask.add_argument("--model", type=str, default=None)

    args = ap.parse_args()

    store = Store(Path(args.workdir))
    embedder = Embedder()

    if args.cmd == "ingest":
        ingest_folder(Path(args.folder), store, embedder, max_len=args.max_len, overlap=args.overlap)
    elif args.cmd == "ask":
        out = answer(args.question, store, embedder, k=args.k, ef=args.ef, mmr=args.mmr, model=args.model)
        print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
