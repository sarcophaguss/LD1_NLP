import pandas as pd

SEED = 42
SAMPLE_SIZE = 1500

df = pd.read_csv("Lab01_Materials/headlines_train.csv")
df["doc_id"] = df.index

df["text"] = df["text"].str.strip()
df = df[df["text"].notna() & df["text"].ne("")] # Paliekam eilutes, kuriose nera tuscio teksto.
n_after_empty_removed = len(df)

df = df.drop_duplicates(subset="text")
n_after_dedup = len(df)

if len(df) > SAMPLE_SIZE:
    documents = df.sample(n=SAMPLE_SIZE, random_state=SEED)
else:
    documents = df.copy()

documents = documents.reset_index(drop=True)

print("Iš viso eilučių žaliame faile:", len(pd.read_csv("Lab01_Materials/headlines_train.csv")))
print("Po tuščių/tarpų pašalinimo:", n_after_empty_removed)
print("Po dublikatų pašalinimo (unikalūs dokumentai):", n_after_dedup)
print("Pasirinkta dokumentų mokymui:", len(documents))
print("Random seed:", SEED)

documents.to_csv("documents_1500.csv", index=False)

SELECTION_SEED = 7
N_HEADLINES = 20

sample_20 = documents.sample(n=N_HEADLINES, random_state=SELECTION_SEED)
sample_20 = sample_20[["doc_id", "text"]].reset_index(drop=True)

print(f"\n20 antraščių atrankai (seed={SELECTION_SEED}):")
for row in sample_20.itertuples(index=False):
    print(f"[{row.doc_id}] {row.text}")

sample_20.to_csv("sample_20_headlines.csv", index=False)