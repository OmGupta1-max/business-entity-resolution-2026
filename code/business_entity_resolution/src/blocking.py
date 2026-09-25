from pathlib import Path
import re
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"


def normalize_name(text: str) -> str:
    """
    Lightweight normalization for the first blocking baseline.
    """

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = text.lower()

    text = text.replace("&", " and ")

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_name_index(path: Path, chunksize: int = 100_000) -> dict:
    """
    Build:
        normalized business name -> entity IDs

    The file is processed in chunks.
    """

    index = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
        usecols=["entity_id", "business_name"],
    ):

        for entity_id, business_name in zip(
            chunk["entity_id"],
            chunk["business_name"],
        ):

            key = normalize_name(business_name)

            if not key:
                continue

            index.setdefault(key, []).append(entity_id)

    return index


def load_ground_truth(path: Path) -> dict:
    """
    Load ground truth as:

    S1 ID -> set of true S2/S3 IDs
    """

    truth = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
    ):

        for s1_id, matched_ids in zip(
            chunk["source1_entity_id"],
            chunk["matched_entity_ids"],
        ):

            if not matched_ids.strip():
                truth[s1_id] = set()
            else:
                truth[s1_id] = {
                    x.strip()
                    for x in matched_ids.split(",")
                    if x.strip()
                }

    return truth


def evaluate_blocking(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    ground_truth_path: Path,
):
    """
    Evaluate exact normalized-name blocking.
    """

    print("Building S2 name index...")
    s2_index = build_name_index(s2_path)

    print("Building S3 name index...")
    s3_index = build_name_index(s3_path)

    print("Loading ground truth...")
    truth = load_ground_truth(ground_truth_path)

    total_true_matches = 0
    recovered_true_matches = 0

    candidate_counts = []

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
        usecols=["entity_id", "business_name"],
    ):

        for s1_id, business_name in zip(
            chunk["entity_id"],
            chunk["business_name"],
        ):

            key = normalize_name(business_name)

            candidates = set()

            candidates.update(
                s2_index.get(key, [])
            )

            candidates.update(
                s3_index.get(key, [])
            )

            candidate_counts.append(len(candidates))

            true_matches = truth.get(s1_id, set())

            total_true_matches += len(true_matches)

            recovered_true_matches += len(
                true_matches.intersection(candidates)
            )

    candidate_series = pd.Series(candidate_counts)

    recall = (
        recovered_true_matches / total_true_matches
        if total_true_matches
        else 0
    )

    print("\n" + "=" * 70)
    print("E2 — EXACT NORMALIZED NAME BLOCKING")
    print("=" * 70)

    print(f"Total true matches:       {total_true_matches:,}")
    print(f"Recovered true matches:   {recovered_true_matches:,}")
    print(f"Candidate recall:         {recall:.4%}")

    print("\nCandidate statistics:")
    print(f"Average candidates/S1:    {candidate_series.mean():.2f}")
    print(f"Median candidates/S1:     {candidate_series.median():.0f}")
    print(f"P95 candidates/S1:        {candidate_series.quantile(0.95):.0f}")
    print(f"Maximum candidates/S1:    {candidate_series.max():,}")


if __name__ == "__main__":

    evaluate_blocking(
        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",
        TRAIN_DIR / "train_ground_truth.tsv",
    )