"""Stage 2: hybrid evidence retrieval.

Three evidence sources, merged together:
  1. BM25 (keyword match)           over your local knowledge base
  2. Dense search (embeddings+FAISS) over your local knowledge base
  3. Wikipedia search API (optional) for general-purpose facts
"""
import re
import logging
import numpy as np
from rank_bm25 import BM25Okapi

from app import config

logger = logging.getLogger(__name__)

_embedder = None


def is_embedder_loaded():
    return _embedder is not None


def get_embedder():
    """Shared sentence-transformer (also used by the verifier for similarity scores)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s ...", config.EMBED_MODEL)
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def embed(texts):
    return get_embedder().encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self):
        self.passages = []          # list[{"text", "source"}]
        self.bm25 = None
        self._dense = None          # FAISS index or numpy matrix, built lazily
        self._matrix = None
        self._load_knowledge_base()

    # ---------- knowledge base ----------
    def _load_knowledge_base(self):
        config.KB_DIR.mkdir(parents=True, exist_ok=True)
        for path in sorted(config.KB_DIR.glob("*.txt")):
            for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="ignore")):
                block = " ".join(block.split())
                if len(block) >= 20:
                    self.passages.append({"text": block, "source": f"local:{path.name}"})
        if self.passages:
            self.bm25 = BM25Okapi([_tokenize(p["text"]) for p in self.passages])
        logger.info("Knowledge base: %d passages", len(self.passages))

    def _ensure_dense(self):
        if self._dense is not None or not self.passages:
            return
        vectors = embed([p["text"] for p in self.passages]).astype("float32")
        self._matrix = vectors
        try:
            import faiss
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self._dense = index
        except Exception as exc:
            logger.warning("FAISS unavailable (%s); using numpy search instead.", exc)
            self._dense = "numpy"

    def warmup(self):
        self._ensure_dense()

    # ---------- searches ----------
    def _bm25_search(self, query, k):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [int(i) for i in order if scores[i] > 0]

    def _dense_search(self, query, k):
        if not self.passages:
            return []
        self._ensure_dense()
        q = embed([query]).astype("float32")
        if self._dense == "numpy":
            sims = (self._matrix @ q[0])
            return [int(i) for i in np.argsort(sims)[::-1][:k]]
        _, idx = self._dense.search(q, min(k, len(self.passages)))
        return [int(i) for i in idx[0] if i >= 0]

    def _local_search(self, query, k):
        """Reciprocal-rank fusion of BM25 and dense rankings."""
        fused = {}
        for ranking in (self._bm25_search(query, k * 2), self._dense_search(query, k * 2)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.passages[i] for i in best]

    @staticmethod
    def _wikipedia_search(query, k, timeout):
        import requests
        params = {
            "action": "query", "generator": "search", "gsrsearch": query[:300],
            "gsrlimit": k, "prop": "extracts", "exintro": 1, "explaintext": 1,
            "exsentences": 5, "format": "json", "origin": "*",
        }
        headers = {"User-Agent": "hallu-xai-student-project/0.1"}
        resp = requests.get("https://en.wikipedia.org/w/api.php", params=params,
                            headers=headers, timeout=timeout)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        ordered = sorted(pages.values(), key=lambda p: p.get("index", 0))
        return [{"text": " ".join(p["extract"].split()), "source": f"wikipedia:{p['title']}"}
                for p in ordered if p.get("extract")]

    # ---------- public API ----------
    def retrieve(self, claim, question="", entities=None, use_wikipedia=None):
        k = config.TOP_K
        use_wiki = config.USE_WIKIPEDIA if use_wikipedia is None else use_wikipedia

        results = list(self._local_search(claim, k))

        if use_wiki:
            # A claim like "It was built in 1889" has no subject - borrow it from the question.
            query = claim if entities else f"{question} {claim}".strip()
            try:
                found = self._wikipedia_search(query, k, config.WIKIPEDIA_TIMEOUT)
                if not found and entities:
                    found = self._wikipedia_search(" ".join(entities[:3]), k, config.WIKIPEDIA_TIMEOUT)
                results.extend(found)
            except Exception as exc:
                logger.warning("Wikipedia lookup failed: %s", exc)

        seen, unique = set(), []
        for item in results:
            if item["text"] not in seen:
                seen.add(item["text"])
                unique.append(item)
        return unique
