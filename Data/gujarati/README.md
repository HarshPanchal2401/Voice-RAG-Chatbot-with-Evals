# Gujarati Golden Evaluation Dataset for Voice & Text RAG

## Overview
- **Language**: Gujarati (`gu`, `gu-IN`)
- **Total Valid Golden Records**: 18,463
- **Average Question Length**: 32.65 characters
- **Average Answer Length**: 99.62 characters
- **Average Positive Passages**: 1.07

## Dataset Files
| Filename | Records | Purpose |
|---|---|---|
| `golden_dataset_full.jsonl` | 18,463 | Complete golden ground-truth evaluation set |
| `golden_dataset_sample_500.jsonl` | 500 | Standard evaluation benchmark |
| `golden_dataset_sample_100.jsonl` | 100 | Rapid CI/CD test benchmark |
| `dataset_summary.json` | - | Summary statistics and type distributions |

## Query Type Breakdown
```json
{
  "DESCRIPTION": 11346,
  "PERSON": 484,
  "LOCATION": 1407,
  "NUMERIC": 3528,
  "ENTITY": 1698
}
```
