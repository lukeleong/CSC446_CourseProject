import re
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
from rouge_score import rouge_scorer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab")

# --- Constants ---
ABSTRACTIVE_MODEL_NAME = "facebook/bart-large-cnn"
MAX_SAMPLES = 100
SUMMARY_MAX_LENGTH = 150
N_TOPICS = 3

# --- Enhanced Medical Stop Words ---
MEDICAL_STOP_WORDS = {
    "patient",
    "patients",
    "study",
    "studies",
    "result",
    "results",
    "method",
    "methods",
    "analysis",
    "group",
    "groups",
    "data",
    "showed",
    "show",
    "significant",
    "difference",
    "clinical",
    "the",
    "of",
    "and",
    "in",
    "with",
    "to",
    "was",
    "were",
    "for",
    "than",
    "or",
}


# --- Model Initialization ---
def init_summarizer():
    print("Loading summarization model...")
    device = 0 if torch.cuda.is_available() else -1
    try:
        tokenizer = AutoTokenizer.from_pretrained(ABSTRACTIVE_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(ABSTRACTIVE_MODEL_NAME)
        summarizer = pipeline(
            "summarization", model=model, tokenizer=tokenizer, device=device
        )
        print(f"Running on {'GPU' if device == 0 else 'CPU'}")
        return summarizer
    except Exception as e:
        print(f"Model loading failed: {e}. Using extractive only.")
        return None


summarizer = init_summarizer()


# --- Text Processing ---
def clean_text(text):
    """Standardize medical text while preserving key clinical terms."""
    if not isinstance(text, str):
        return ""

    # Preserve medical units/measurements
    text = re.sub(
        r"(\d+\s*mg|\d+\s*ml|\d+\s*%|p\s*[<>=]\s*\d+\.\d+)",
        lambda m: m.group().replace(" ", ""),
        text,
    )

    # Remove noise
    text = re.sub(r"[\[\]\{\}\"\'\\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --- Keyword Extraction ---
def extract_keywords(texts, top_n=15):
    """Extract meaningful medical keywords using TF-IDF."""
    vectorizer = TfidfVectorizer(
        stop_words=list(MEDICAL_STOP_WORDS), ngram_range=(1, 2), max_features=5000
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        scores = np.array(tfidf_matrix.sum(axis=0))[0]
        top_indices = np.argsort(scores)[-top_n:][::-1]
        return [
            term
            for term in vectorizer.get_feature_names_out()[top_indices]
            if not term.isdigit() and len(term) > 2
        ]
    except Exception:
        return []


# --- Summarization ---
def summarize_extractive(texts, max_sentences=5):
    """Select diverse, important sentences using TF-IDF centrality."""
    combined_text = " ".join(texts[:20])  # Process first 20 abstracts max
    sentences = [s for s in sent_tokenize(combined_text) if len(s.split()) > 5]

    if len(sentences) <= max_sentences:
        return combined_text

    vectorizer = TfidfVectorizer(stop_words=list(MEDICAL_STOP_WORDS))
    try:
        sentence_vectors = vectorizer.fit_transform(sentences)
        similarity_matrix = (sentence_vectors * sentence_vectors.T).toarray()
        centrality_scores = np.sum(similarity_matrix, axis=1).flatten()
        top_indices = np.argsort(centrality_scores)[-max_sentences:][::-1]
        return " ".join([sentences[i] for i in sorted(top_indices)])
    except Exception:
        return " ".join(sentences[:max_sentences])


def summarize_abstractive(text, max_length=SUMMARY_MAX_LENGTH):
    """Generate concise medical summary with BART."""
    if not summarizer or len(text.split()) < 20:
        return text[:max_length]

    try:
        summary = summarizer(
            text, max_length=max_length, min_length=50, do_sample=False, truncation=True
        )[0]["summary_text"]
        return re.sub(r"\s([.,])", r"\1", summary)  # Fix spacing
    except Exception:
        return text[:max_length]


# --- Topic Modeling ---
def identify_topics(abstracts, n_topics=N_TOPICS):
    """Extract clinically relevant topics using LDA."""
    vectorizer = TfidfVectorizer(
        stop_words=list(MEDICAL_STOP_WORDS), ngram_range=(1, 3), max_features=3000
    )

    try:
        X = vectorizer.fit_transform(abstracts)
        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, learning_method="online"
        )
        lda.fit(X)

        # Get top 10 terms per topic
        terms = []
        for topic in lda.components_:
            top_indices = topic.argsort()[:-11:-1]
            terms.append(
                [
                    term
                    for term in vectorizer.get_feature_names_out()[top_indices]
                    if not term.isdigit()
                ][:5]
            )  # Keep top 5 most relevant

        return lda.transform(X).argmax(axis=1), terms
    except Exception:
        return np.zeros(len(abstracts), dtype=int), []


def extract_clinical_elements(texts):
    """Extract key clinical components from medical texts with better pattern matching."""
    combined_text = " ".join(texts[:15])  # Analyze first 15 abstracts max

    stats = list(
        set(
            re.findall(
                r"(?:\d+(?:\.\d+)?\s*%|\d+\s*/\s*\d+|p\s*[<>]=?\s*0\.\d+)",
                combined_text,
            )
        )
    )[
        :3
    ]  # Keep top 3 unique

    outcomes = list(
        set(
            re.findall(
                r"(?:associated with|resulted in|led to|caused|improved|reduced|increased|decreased)\s+[\w\s,\-]+?(?:in|among|with|for)\s[\w\s,\-]+",
                combined_text,
                re.IGNORECASE,
            )
        )
    )[:3]

    populations = list(
        set(
            re.findall(
                r"(?:\d+\s*(?:patients|subjects|cases|individuals|participants))\s*(?:with|of|having|diagnosed with|suffering from)\s[\w\s,\-]+",
                combined_text,
                re.IGNORECASE,
            )
        )
    )[:2]

    return {"stats": stats, "outcomes": outcomes, "populations": populations}


def generate_clinical_summary(topic_terms, clinical_facts):
    """Generate a structured medical summary avoiding prompt leakage."""

    has_populations = bool(clinical_facts["populations"])
    has_outcomes = bool(clinical_facts["outcomes"])
    has_stats = bool(clinical_facts["stats"])

    parts = []

    if has_populations:
        parts.append(f"Key populations: {', '.join(clinical_facts['populations'])}")
    else:
        parts.append(f"Key populations: Study focused on {topic_terms[0]}")

    if has_outcomes:
        parts.append(f"Key outcomes: {', '.join(clinical_facts['outcomes'])}")

    if has_stats:
        parts.append(f"Statistical results: {', '.join(clinical_facts['stats'])}")

    return "\n".join(parts)
#rouge score
def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(reference, candidate)
# --- Main Pipeline ---
def main():
    print("Loading dataset...")
    dataset = load_dataset("TimSchopf/medical_abstracts", split="train")
    df = pd.DataFrame(dataset[:MAX_SAMPLES])

    # Handle column names
    abstract_col = next(
        (col for col in ["medical_abstract", "abstract"] if col in df.columns), None
    )
    if not abstract_col:
        raise ValueError("No abstract column found")

    df[abstract_col] = df[abstract_col].apply(clean_text).dropna()
    print(f"Processing {len(df)} abstracts...")

    results = {}
    class_labels = (
        sorted(df["condition_label"].unique())
        if "condition_label" in df.columns
        else ["all"]
    )

    for class_name in class_labels:
        class_df = (
            df[df["condition_label"] == class_name]
            if "condition_label" in df.columns
            else df
        )
        abstracts = class_df[abstract_col].tolist()

        if not abstracts:
            continue

        topic_labels, topic_terms = identify_topics(abstracts)

        topics = []
        for topic_id, terms in enumerate(topic_terms):
            topic_abstracts = [
                ab for ab, lbl in zip(abstracts, topic_labels) if lbl == topic_id
            ]

            clinical_facts = extract_clinical_elements(topic_abstracts)
            summary = generate_clinical_summary(terms, clinical_facts)

            # Only apply abstractive summarization is alot content
            if len(summary.split()) > 20:
                summary = summarize_abstractive(summary)
            else:
                # Fall back to extractive if not enough  content
                extract_summary = summarize_extractive(topic_abstracts)
                summary = summarize_abstractive(extract_summary)
            reference_summary = summarize_extractive(topic_abstracts)
            rouge_scores = compute_rouge(reference_summary, summary)

            topics.append({"id": topic_id + 1, "terms": terms, "summary": summary,"rouge_scores": rouge_scores})

        results[str(class_name)] = {"topics": topics}

    all_texts = df[abstract_col].tolist()
    results["keywords"] = [
        kw
        for kw in extract_keywords(all_texts, top_n=20)
        if not any(stop in kw for stop in MEDICAL_STOP_WORDS)
    ][
        :15
    ]  # Top 15 most relevant

    with open("medical_summaries_improved.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Processing complete.")

    #print rouge scores
    print(f"Topic {topic_id + 1} ROUGE Scores: {rouge_scores}")


if __name__ == "__main__":
    main()


# TODO
# --- Testing TODOs ---
# 1. ROUGE TEST SETUP
# 2. EXTRACTIVE SUMMARY TEST
# 3. ABSTRACTIVE SUMMARY TEST
# 4. KEYWORD EXTRACTION TEST
