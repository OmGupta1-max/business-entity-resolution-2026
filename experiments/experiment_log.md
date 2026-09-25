# Experiment Log

## E0 — Environment and Dataset Setup

Status: Complete

- Python virtual environment created.
- Core dependencies installed successfully.
- Dataset copied locally.
- Official submission validator available.
- GitHub repository initialized.
- Dataset excluded from Git via `.gitignore`.

---

## E1 — Basic Dataset EDA

Status: Complete

### Source sizes

| Dataset | Rows | Unique IDs | Duplicate IDs |
|---|---:|---:|---:|
| Train Source 1 | 2,206,821 | 2,206,821 | 0 |
| Train Source 2 | 5,034,616 | 5,034,616 | 0 |
| Train Source 3 | 5,285,603 | 5,285,603 | 0 |

Total source records: 12,527,040.

### Missing values

| Dataset | Empty business name | Empty business address | Empty country |
|---|---:|---:|---:|
| Train Source 1 | 0 | 0 | 0 |
| Train Source 2 | 0 | 168,967 | 0 |
| Train Source 3 | 0 | 175,916 | 0 |

Implication:

Address cannot be a mandatory blocking condition because approximately 3.3% of S2/S3 records have no address.

### Training countries

Training sources contain:

- US
- India

The pipeline must not hard-code the country universe because the test data includes France.

### Ground truth

| Statistic | Result |
|---|---:|
| Total S1 records | 2,206,821 |
| Zero matches | 123,247 |
| Exactly one match | 119,157 |
| Multiple matches | 1,964,417 |
| Average matches/S1 | 3.461 |
| Median matches/S1 | 3 |
| Maximum matches/S1 | 11 |
| S2 matches | 3,693,619 |
| S3 matches | 3,944,746 |

Implications:

- The task is not one-to-one matching.
- The decision layer must support zero-to-many predictions per S1.
- The system must explicitly support no-match predictions.
- Selecting only the single highest-scoring candidate is insufficient.

### Match reuse

| Entity type | Matched to multiple S1s | Maximum reuse |
|---|---:|---:|
| S2 | 0 | 1 |
| S3 | 0 | 1 |

Training ground truth therefore shows a one-to-many relationship from S1 to S2/S3, while each matched S2/S3 entity belongs to only one S1.

This structure may be tested as a decision-layer constraint, but should not be assumed to be a universal hard rule without validation.

### E1 conclusion

The dataset is large and memory-sensitive. Source 1 is complete and deduplicated, while S2/S3 contain missing addresses. Ground truth is predominantly multi-match and contains approximately 5.6% no-match S1 records. Candidate generation must therefore prioritize high recall, while the final decision layer must support zero-to-many matches.

---

## E2 — Exact Normalized Name Blocking

Status: Complete

### Results

| Metric | Result |
|---|---:|
| Total true matches | 7,638,365 |
| Recovered true matches | 1,958,580 |
| Candidate recall | 25.6414% |
| Average candidates/S1 | 11.10 |
| Median candidates/S1 | 2 |
| P95 candidates/S1 | 69 |
| Maximum candidates/S1 | 1,071 |

### Conclusion

Exact normalized business-name blocking alone is insufficient because it recovers only 25.64% of true matches.

It will be retained as a potentially useful component of a future multi-block union because it produces relatively small candidate sets, but it cannot be used as the primary blocking strategy.

### Next experiment

Evaluate stronger character-based blocking strategies while monitoring candidate recall and candidate-set size.

---

## E3-A — Full Character TF-IDF Blocking

Status: Rejected for current hardware

### Attempt

A character 3-4 gram TF-IDF representation was attempted over all S2/S3 business names.

Total records:

10,320,219

Configuration:

- analyzer: character
- ngram_range: (3, 4)
- min_df: 2
- max_features: 1,000,000
- sublinear_tf: True

### Result

The experiment failed during sparse-matrix construction because the system could not allocate an additional 1.66 GiB array.

Error:

NumPy ArrayMemoryError.

### Decision

Do not use a full S2+S3 character TF-IDF matrix on the current ~8 GB RAM machine.

Character similarity may still be useful later for pairwise features on a much smaller candidate set.

### Lesson

Character similarity should be applied after blocking rather than creating a massive global TF-IDF index.