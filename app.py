# Add these imports at top of your app.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
from nltk.corpus import wordnet as wn
import re
import numpy as np
from typing import List, Dict

# Ensure NLTK WordNet is available
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load models once (global)
_EMBED_MODEL_NAME = "all-mpnet-base-v2"  # high-quality; smaller alternative: "all-MiniLM-L6-v2"
_EMBEDDER = SentenceTransformer(_EMBED_MODEL_NAME)
_NLP = spacy.load("en_core_web_sm")

# ---------- Helpers ----------
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[\(\)\[\]\{\}:;,]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def expand_synonyms(word: str) -> List[str]:
    """Return a small set of synonyms from WordNet (lemmas)."""
    syns = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            syns.add(lemma.name().replace('_', ' '))
    # include original
    syns.add(word)
    return list(syns)

def split_into_sentences(text: str) -> List[str]:
    doc = _NLP(text)
    sents = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 2]
    # fallback: if spaCy couldn't split, fallback to whole text
    if not sents:
        return [text]
    return sents

# ---------- Core matching function ----------
def compute_skill_match_score(job_req_text: str, resume_text: str,
                              skill_weighting: Dict[str, float] = None,
                              thresholds: Dict[str, float] = None) -> Dict:
    """
    Returns a detailed dict:
      {
        'final_percent': float,
        'per_skill': { skill_text: {'best_similarity': float, 'score': float, 'evidence_sentence': str} },
        'debug': {...}
      }
    - skill_weighting: optional per-skill weight (sums not required; normalized)
    - thresholds: tunable thresholds, e.g. {'absent': 0.45, 'partial': 0.6, 'strong': 0.75}
    """
    if thresholds is None:
        thresholds = {'absent': 0.45, 'partial': 0.6, 'strong': 0.75}

    job_req_text = normalize_text(job_req_text)
    resume_text = normalize_text(resume_text)

    # 1) Extract candidate 'required skills' from job text via a simple heuristic:
    #    - find lines containing keywords like "skills", "requirements", or list items
    # For reliability, we also allow user to pass an explicit list of job skills (not covered here).
    # Simple heuristic: split job text into lines and look for comma-separated tokens
    lines = [l.strip() for l in job_req_text.splitlines() if l.strip()]
    candidate_skill_phrases = []
    for l in lines:
        # if the line seems like a skill list (contains commas or '•' bullet) or is short, treat as skill entries
        if ',' in l or '•' in l or len(l.split()) <= 6:
            parts = re.split(r'[,\u2022\-–—]+', l)
            for p in parts:
                p2 = p.strip()
                if len(p2) > 1:
                    candidate_skill_phrases.append(p2)
    # fallback: if nothing detected, use the whole job text as one skill phrase (broad)
    if not candidate_skill_phrases:
        candidate_skill_phrases = [job_req_text[:512]]

    # normalize dedupe
    candidate_skill_phrases = list(dict.fromkeys([normalize_text(s) for s in candidate_skill_phrases]))

    # 2) Build sentence embeddings for resume (split into sentences for fine-grained evidence)
    resume_sents = split_into_sentences(resume_text)
    resume_embeddings = _EMBEDDER.encode(resume_sents, convert_to_numpy=True, show_progress_bar=False)

    # 3) For each required skill phrase, compute embedding and max similarity against resume sentences
    per_skill = {}
    skill_embeddings = _EMBEDDER.encode(candidate_skill_phrases, convert_to_numpy=True, show_progress_bar=False)

    for i, skill in enumerate(candidate_skill_phrases):
        skill_emb = skill_embeddings[i].reshape(1, -1)
        sims = cosine_similarity(skill_emb, resume_embeddings).flatten()  # one score per resume sentence
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx]) if len(sims) > 0 else 0.0
        evidence = resume_sents[best_idx] if len(resume_sents) > 0 else ""
        # Secondary exact/synonym check to avoid false positives:
        exact_present = False
        # check exact token or synonyms in resume text
        tokens = [t.lemma_.lower() for t in _NLP(skill)]
        for tok in tokens:
            if tok and (re.search(r'\b' + re.escape(tok) + r'\b', resume_text)):
                exact_present = True
                break
            # synonyms
            for syn in expand_synonyms(tok)[:3]:  # limit synonyms to 3 per token
                if re.search(r'\b' + re.escape(syn.lower()) + r'\b', resume_text):
                    exact_present = True
                    break
            if exact_present:
                break

        # Convert similarity to score (0-100) using thresholds
        if best_sim < thresholds['absent'] and not exact_present:
            score = 0.0
        elif best_sim < thresholds['partial'] and not exact_present:
            # small partial credit proportional to (sim - absent) / (partial - absent)
            frac = (best_sim - thresholds['absent']) / max(1e-6, (thresholds['partial'] - thresholds['absent']))
            score = max(0.0, min(1.0, frac)) * 50.0  # up to 50% for weak evidence
        elif best_sim < thresholds['strong'] and not exact_present:
            frac = (best_sim - thresholds['partial']) / max(1e-6, (thresholds['strong'] - thresholds['partial']))
            score = 50.0 + frac * 30.0  # from 50 to 80
        else:
            # strong similarity or exact_present
            score = 85.0 + min(15.0, (best_sim - thresholds.get('strong', 0.75)) * 100.0)  # up to 100

        per_skill[skill] = {
            'best_similarity': round(best_sim, 4),
            'score': round(float(score), 2),
            'evidence_sentence': evidence,
            'exact_present': bool(exact_present)
        }

    # 4) Aggregate scores with weights
    if skill_weighting is None:
        # equal weighting by default
        weights = {s: 1.0 for s in per_skill.keys()}
    else:
        # normalize weights
        weights = {}
        for s in per_skill.keys():
            w = float(skill_weighting.get(s, 1.0)) if isinstance(skill_weighting, dict) else 1.0
            weights[s] = max(0.0, w)
    total_w = sum(weights.values()) or 1.0
    weighted_sum = sum(per_skill[s]['score'] * weights[s] for s in per_skill.keys())
    final_percent = float(weighted_sum / total_w)

    # If every per_skill score is 0 => final_percent should be exactly 0 (no fuzzy floor)
    if all(per_skill[s]['score'] == 0.0 for s in per_skill.keys()):
        final_percent = 0.0

    return {
        'final_percent': round(final_percent, 2),
        'per_skill': per_skill,
        'required_skills': list(per_skill.keys()),
        'debug': {
            'num_resume_sentences': len(resume_sents),
            'thresholds': thresholds
        }
    }
