from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"


def normalize_name(text: str) -> str:
    """Basic normalization used for the character baseline."""

    text = str(text).lower()

    return " ".join(
        "".join(
            char if char.isalnum() else " "
            for char in text
        ).split()
    )


def load_names(path: Path) -> pd.DataFrame:
    """Load only entity_id and business_name."""

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "business_name"],
    )


def main():

    print("Loading S2...")

    s2 = load_names(
        TRAIN_DIR / "train_source2.tsv"
    )

    print(f"S2 rows: {len(s2):,}")

    print("Loading S3...")

    s3 = load_names(
        TRAIN_DIR / "train_source3.tsv"
    )

    print(f"S3 rows: {len(s3):,}")

    sources = pd.concat(
        [s2, s3],
        ignore_index=True,
    )

    print(f"Combined rows: {len(sources):,}")

    print("\nNormalizing names...")

    names = sources["business_name"].map(
        normalize_name
    )

    print("Building character TF-IDF...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 4),
        min_df=2,
        max_features=1_000_000,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(names)

    print(
        f"TF-IDF matrix shape: {matrix.shape}"
    )

    print(
        f"Non-zero values: {matrix.nnz:,}"
    )

    print("\nBuilding nearest-neighbor index...")

    nn = NearestNeighbors(
        n_neighbors=20,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nn.fit(matrix)

    print("Index ready.")

    print("\nLoading a small S1 sample...")

    s1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "business_name"],
        nrows=1000,
    )

    s1_names = s1["business_name"].map(
        normalize_name
    )

    s1_matrix = vectorizer.transform(
        s1_names
    )

    distances, indices = nn.kneighbors(
        s1_matrix,
        n_neighbors=20,
    )

    candidate_counts = [
        len(set(row))
        for row in indices
    ]

    print("\n" + "=" * 70)
    print("E3 — CHARACTER TF-IDF RETRIEVAL SANITY TEST")
    print("=" * 70)

    print(
        f"S1 samples tested:        {len(s1):,}"
    )

    print(
        f"Candidates/S1:             {candidate_counts[0]}"
    )

    print(
        f"Average returned/S1:       "
        f"{sum(candidate_counts) / len(candidate_counts):.2f}"
    )

    print(
        "\nE3 sanity test completed."
    )


if __name__ == "__main__":
    main()