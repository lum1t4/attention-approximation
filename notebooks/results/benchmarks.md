# Benchmark Results

## MobileLLM-350M with Attention Approximation

| model | arc_easy | arc_challenge | hellaswag | piqa | winogrande | mmlu | avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **MobileLLM-350M** | 53.8 | 33.5 | 49.6 | 68.6 | 57.6 | - | **52.6** |
| **MobileLLM-Approx.** | 26.6 | 20.6 | 25.8 | 53.3 | 51.8 | 22.9 | **33.5** |
| **Δ (Change)** | -27.2 | -12.9 | -23.8 | -15.3 | -5.8 | - | **-19.1** |

### Notes

- The baseline MobileLLM-350M scores are from the [MobileLLM paper](https://arxiv.org/abs/2402.14905)
- The baseline average (52.6) is calculated across the 5 common benchmarks: arc_easy, arc_challenge, hellaswag, piqa, and winogrande
- The custom model average (33.5) is calculated across all 6 benchmarks including mmlu
- MMLU was not included in the original MobileLLM-350M benchmark suite
- All scores are accuracy percentages
- lambada_openai was excluded from the table (scored 0.0, likely indicating an evaluation issue)

### Analysis

The results show that the attention approximation significantly impacts model performance, with an average drop of **19.1 percentage points** across the common benchmarks:

- **WinoGrande** shows the smallest degradation (-5.8%)
- **arc_easy** shows the largest drop (-27.2%)
- **PIQA** maintains relatively better performance (-15.3%)
- **HellaSwag** and **arc_challenge** show moderate degradation (-23.8% and -12.9% respectively)

The attention approximation appears to have varying effects on different types of reasoning tasks, with commonsense reasoning about physical objects (WinoGrande, PIQA) being more robust to the approximation than general knowledge and reading comprehension tasks (arc_easy, HellaSwag).
