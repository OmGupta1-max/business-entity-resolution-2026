from pathlib import Path
from collections import Counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"


def read_tsv(
    path: Path,
    nrows: int | None = None,
    chunksize: int | None = None,
):
    """Read a TSV file safely."""
    return pd.read_csv(
        path,
        sep="\t",
        nrows=nrows,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
    )


def scan_basic_stats(path: Path, chunksize: int = 100_000) -> dict:
    """
    Scan a TSV file in chunks and calculate basic data-quality statistics.
    """

    total_rows = 0
    unique_ids = set()

    empty_name = 0
    empty_address = 0
    empty_country = 0

    country_counts = Counter()

    for chunk in read_tsv(path, chunksize=chunksize):

        total_rows += len(chunk)

        # Entity ID statistics
        unique_ids.update(chunk["entity_id"])

        # Empty-field statistics
        empty_name += (chunk["business_name"].str.strip() == "").sum()
        empty_address += (chunk["business_address"].str.strip() == "").sum()
        empty_country += (chunk["country"].str.strip() == "").sum()

        # Country distribution
        country_counts.update(chunk["country"].str.strip())

    duplicate_ids = total_rows - len(unique_ids)

    return {
        "file": path.name,
        "rows": total_rows,
        "unique_ids": len(unique_ids),
        "duplicate_ids": duplicate_ids,
        "empty_name": int(empty_name),
        "empty_address": int(empty_address),
        "empty_country": int(empty_country),
        "countries": country_counts,
    }


def print_stats(stats: dict) -> None:
    """Print statistics in a readable format."""

    print("\n" + "=" * 70)
    print(f"FILE: {stats['file']}")
    print("=" * 70)

    print(f"Total rows:          {stats['rows']:,}")
    print(f"Unique entity IDs:   {stats['unique_ids']:,}")
    print(f"Duplicate IDs:       {stats['duplicate_ids']:,}")

    print("\nEmpty values:")
    print(f"Business name:       {stats['empty_name']:,}")
    print(f"Business address:    {stats['empty_address']:,}")
    print(f"Country:             {stats['empty_country']:,}")

    print("\nCountries:")

    for country, count in stats["countries"].most_common():
        print(f"{country or '<EMPTY>':20} {count:,}")

def analyze_ground_truth(path: Path) -> None:
    """
    Analyze the match-count distribution in the training ground truth.
    """

    total_s1 = 0
    zero_matches = 0
    one_match = 0
    multiple_matches = 0

    match_counts = []

    s2_match_count = 0
    s3_match_count = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
    ):
        for matched_ids in chunk["matched_entity_ids"]:

            total_s1 += 1

            if matched_ids.strip() == "":
                zero_matches += 1
                match_counts.append(0)
                continue

            ids = [
                x.strip()
                for x in matched_ids.split(",")
                if x.strip()
            ]

            count = len(ids)
            match_counts.append(count)

            if count == 1:
                one_match += 1
            else:
                multiple_matches += 1

            for entity_id in ids:
                if entity_id.startswith("S2-"):
                    s2_match_count += 1
                elif entity_id.startswith("S3-"):
                    s3_match_count += 1

    print("\n" + "=" * 70)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 70)

    print(f"Total S1 records:       {total_s1:,}")
    print(f"Zero matches:           {zero_matches:,}")
    print(f"Exactly one match:      {one_match:,}")
    print(f"Multiple matches:       {multiple_matches:,}")

    if match_counts:
        print(f"Average matches/S1:     {sum(match_counts) / len(match_counts):.3f}")
        print(f"Median matches/S1:      {pd.Series(match_counts).median():.1f}")
        print(f"Maximum matches/S1:     {max(match_counts):,}")

    print("\nMatched entity distribution:")
    print(f"S2 matches:             {s2_match_count:,}")
    print(f"S3 matches:             {s3_match_count:,}")


def analyze_match_reuse(path: Path) -> None:
    """
    Check whether S2/S3 entities appear as matches
    for multiple S1 entities.
    """

    entity_to_s1_count = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
    ):

        for _, row in chunk.iterrows():

            s1_id = row["source1_entity_id"]
            matched_ids = row["matched_entity_ids"].strip()

            if not matched_ids:
                continue

            for entity_id in matched_ids.split(","):

                entity_id = entity_id.strip()

                if not entity_id:
                    continue

                if entity_id not in entity_to_s1_count:
                    entity_to_s1_count[entity_id] = 0

                entity_to_s1_count[entity_id] += 1

    s2_reused = sum(
        1
        for entity_id, count in entity_to_s1_count.items()
        if entity_id.startswith("S2-") and count > 1
    )

    s3_reused = sum(
        1
        for entity_id, count in entity_to_s1_count.items()
        if entity_id.startswith("S3-") and count > 1
    )

    max_s2_reuse = max(
        (
            count
            for entity_id, count in entity_to_s1_count.items()
            if entity_id.startswith("S2-")
        ),
        default=0,
    )

    max_s3_reuse = max(
        (
            count
            for entity_id, count in entity_to_s1_count.items()
            if entity_id.startswith("S3-")
        ),
        default=0,
    )

    print("\n" + "=" * 70)
    print("MATCH REUSE ANALYSIS")
    print("=" * 70)

    print(f"S2 entities matched to multiple S1s: {s2_reused:,}")
    print(f"S3 entities matched to multiple S1s: {s3_reused:,}")

    print(f"Maximum S2 reuse count:              {max_s2_reuse:,}")
    print(f"Maximum S3 reuse count:              {max_s3_reuse:,}")

    
def main():
    files = [
        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",
    ]

    for path in files:
        stats = scan_basic_stats(path)
        print_stats(stats)

    ground_truth_path = TRAIN_DIR / "train_ground_truth.tsv"

    analyze_ground_truth(ground_truth_path)
    analyze_match_reuse(ground_truth_path)

if __name__ == "__main__":
    main()