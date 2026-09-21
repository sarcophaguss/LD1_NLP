import re

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

VOCAB_SIZE = 3000
LDA_SEED = 42
LDA_MAX_ITER = 15
TOP_N_WORDS = 10

documents = pd.read_csv("documents_1500.csv")

# Grammatical Lithuanian stop words only; template phrases and place names are
# intentionally kept for the Stage 3 preprocessing experiment.
LITHUANIAN_STOP_WORDS = {
    "ir", "ar", "bet", "kad", "tai", "yra", "į", "iš", "su", "po", "dėl", "nes",
    "kai", "ne", "o", "bei", "arba", "kaip", "šis", "ši", "šio", "šios", "jis",
    "ji", "jo", "jos", "jie", "jų", "savo", "buvo", "bus", "taip", "pat", "gali",
    "galima", "daug", "labai", "dar", "jau", "tik", "apie", "per", "prie", "nuo",
    "už", "be", "ant",
}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | LITHUANIAN_STOP_WORDS)

# only alphabetic tokens (letters incl. Lithuanian diacritics), 2+ chars, no bare numbers
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"

vectorizer = CountVectorizer(
    lowercase=True,
    token_pattern=TOKEN_PATTERN,
    stop_words=STOP_WORDS,
    max_features=VOCAB_SIZE,
)

dtm = vectorizer.fit_transform(documents["text"])
vocab = vectorizer.get_feature_names_out()

row_sums = dtm.sum(axis=1).A1
empty_mask = row_sums == 0

print("Dokumentų skaičius (prieš filtravimą):", dtm.shape[0])
print("Žodyno dydis:", dtm.shape[1])
print("Dokumentų be jokio žodyno termino:", int(empty_mask.sum()))

empty_examples = documents.loc[empty_mask, ["doc_id", "text"]].head(10)
print("\nPavyzdžiai (tušti po apdorojimo):")
for row in empty_examples.itertuples(index=False):
    print(f"[{row.doc_id}] {row.text}")

# Drop empty-vector documents before topic modelling; keep the rest aligned.
documents_kept = documents.loc[~empty_mask].reset_index(drop=True)
dtm_kept = dtm[~empty_mask]

print("\nDokumentų liko modeliavimui:", dtm_kept.shape[0])


def train_and_report(n_topics, dtm, vocab, documents, seed=LDA_SEED, max_iter=LDA_MAX_ITER, top_docs=3):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        random_state=seed,
        learning_method="batch",
    )
    doc_topic = lda.fit_transform(dtm)

    print(f"\n=== LDA, temų skaičius = {n_topics} (seed={seed}, max_iter={max_iter}) ===")
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:TOP_N_WORDS]
        top_words = [vocab[i] for i in top_indices]
        print(f"Tema {topic_idx}: {', '.join(top_words)}")

        top_doc_indices = doc_topic[:, topic_idx].argsort()[::-1][:top_docs]
        for doc_idx in top_doc_indices:
            weight = doc_topic[doc_idx, topic_idx]
            doc_id = documents.iloc[doc_idx]["doc_id"]
            text = documents.iloc[doc_idx]["text"]
            print(f"    [{doc_id}] (w={weight:.2f}) {text}")

    return lda, doc_topic


lda_a, doc_topic_a = train_and_report(n_topics=4, dtm=dtm_kept, vocab=vocab, documents=documents_kept)
lda_b, doc_topic_b = train_and_report(n_topics=7, dtm=dtm_kept, vocab=vocab, documents=documents_kept)

PROVISIONAL_N_TOPICS = 4

# Stage 3: recurring template suffixes observed across unrelated headlines.
# Removed as exact phrases (not single stop words) so real content, e.g. the
# word "markets" inside "Central bank cuts rates, markets rally...", stays.
TEMPLATE_PHRASES = [
    "amid concerns",
    "dėl rinkos reakcijos",
    "analitikų teigimu",
    "— analysts say",
    "analysts say",
    "as markets react",
    "in a surprise move",
    "po pranešimo",
    "after report",
    "— find out why",
    "find out why",
    "— read more",
    "read more",
    "you won't believe it",
    "— number 5 is shocking",
    "number 5 is shocking",
]


def strip_template_phrases(text, phrases=TEMPLATE_PHRASES):
    cleaned = text
    for phrase in phrases:
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
    return cleaned


documents_clean = documents_kept.copy()
documents_clean["text"] = documents_clean["text"].apply(strip_template_phrases)

vectorizer_clean = CountVectorizer(
    lowercase=True,
    token_pattern=TOKEN_PATTERN,
    stop_words=STOP_WORDS,
    max_features=VOCAB_SIZE,
)
dtm_clean = vectorizer_clean.fit_transform(documents_clean["text"])
vocab_clean = vectorizer_clean.get_feature_names_out()

row_sums_clean = dtm_clean.sum(axis=1).A1
empty_mask_clean = row_sums_clean == 0
print(f"\nŽodyno dydis po frazių pašalinimo: {len(vocab_clean)}")
print("Naujai tuščių dokumentų (dėl frazių pašalinimo):", int(empty_mask_clean.sum()))

documents_clean_kept = documents_clean.loc[~empty_mask_clean].reset_index(drop=True)
dtm_clean_kept = dtm_clean[~empty_mask_clean]

lda_c, doc_topic_c = train_and_report(
    n_topics=PROVISIONAL_N_TOPICS,
    dtm=dtm_clean_kept,
    vocab=vocab_clean,
    documents=documents_clean_kept,
)


def show_topic_shift(doc_id, label):
    idx_a = documents_kept.index[documents_kept["doc_id"] == doc_id][0]
    idx_c = documents_clean_kept.index[documents_clean_kept["doc_id"] == doc_id][0]
    print(f"\n{label} [doc_id={doc_id}]")
    print("  originalus tekstas:", documents_kept.loc[idx_a, "text"])
    print("  A (originalus apdorojimas) temų svoriai:", doc_topic_a[idx_a].round(2))
    print("  C (be šablonų frazių)      temų svoriai:", doc_topic_c[idx_c].round(2))


show_topic_shift(46954, "Sportas su 'as markets react' priesaga")
show_topic_shift(21479, "Apple/tech su 'analitikų teigimu' priesaga")

# --- Stage 4: final model = Run A (original preprocessing, 4 topics) ---

TOPIC_LABELS_A = {
    0: "Neaiški / mišri (finansai + clickbait žodžiai)",
    1: "Neaiški / mišri (sportas + verslas + clickbait žodžiai)",
    2: "Technologijos / DI",
    3: "Neaiški / mišri (finansai + tech + sportas per bendrą priesagą)",
}

print("\n=== Galutinio modelio (A, 4 temos) temų pavadinimai ===")
for topic_idx, label in TOPIC_LABELS_A.items():
    print(f"Tema {topic_idx}: {label}")

print("\n=== Tavo 20 antraščių priskyrimas galutiniame modelyje (A) ===")
sample_20 = pd.read_csv("sample_20_headlines.csv")
for row in sample_20.itertuples(index=False):
    matches = documents_kept.index[documents_kept["doc_id"] == row.doc_id]
    if len(matches) == 0:
        print(f"[{row.doc_id}] NEBĖRA 1500-dokumentų rinkinyje po dublikatų/tuščių filtro")
        continue
    idx = matches[0]
    weights = doc_topic_a[idx]
    top_topic = int(weights.argmax())
    print(f"[{row.doc_id}] {row.text}")
    print(f"    -> Tema {top_topic} ({TOPIC_LABELS_A[top_topic]}), svoris={weights[top_topic]:.2f}, visi svoriai={weights.round(2)}")

# --- Stage 5: inference on unseen headlines with the already-fitted model ---
# Uses `vectorizer` and `lda_a` as fitted on the 1500-doc collection; no retraining.


def predict_new_headlines(headlines):
    new_dtm = vectorizer.transform(headlines)
    new_term_counts = new_dtm.sum(axis=1).A1

    for text, term_count, weights in zip(headlines, new_term_counts, lda_a.transform(new_dtm)):
        print(f"\nAntraštė: {text}")
        if term_count == 0:
            print("    Nepakanka žodyno terminų prasmingam priskyrimui (tuščias vektorius).")
            continue
        top_topic = int(weights.argmax())
        print(f"    -> Tema {top_topic} ({TOPIC_LABELS_A[top_topic]}), svoris={weights[top_topic]:.2f}, visi svoriai={weights.round(2)}")


if __name__ == "__main__":
    print("\n=== Bandomasis nematytų antraščių apdorojimas (be pakartotinio mokymo) ===")
    predict_new_headlines([
        "Local council approves new cycling lanes across the city",
        "xyz qwrt zzq",
    ])

