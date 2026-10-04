# Decision log

| #     | Date       | Decision                                                                           | Why                                                                                                    |
| ----- | ---------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| D-001 | 2026-10-03 | Run the model locally (Ollama); never send BES answers to cloud APIs or chat tools | BES terms: no passing data to third parties; GDPR                                                      |
| D-002 | 2026-10-03 | Evaluate on wave 25 (May 2023)                                                     | BES codebook: waves 26+ were coded by GPT-4o, not humans                                               |
| D-003 | 2026-10-03 | Model: Llama 3.1 8B (4.9 GB)                                                       | Already installed; fits the 3–8B plan; tested on this BES task in a 2025 study                         |
| D-004 | 2026-10-03 | Keep the test set at 1,200                                                         | Median 6.7 s/answer (17 calls, 16 GB RAM, CPU/GPU: CPU) -> about 4.5 h for both schemes                |
| D-005 | 2026-10-03 | Keep "Uncoded" answers as their own class in both schemes                          | BES gives them a 50-category code but no 12-category one; a real coding tool must handle vague answers |
| D-006 | 2026-10-03 | Stratified sample by 50-category code (seed 2026): dev 300, test 1,200             | Keeps the wave's category mix; the seed makes it reproducible                                          |

| D-007 | 2026-10-04 | Codebooks drafte from the BES coding rules, then revised in a review against the 300 dev answers; I checked every change | details in docs/codebook_review.md |
| D-008 | 2026-10-04 | Checked the 50→12 grouping in the W25 data: Asylum→Immigration, Election outcome→Other, Transport→Austerity/spending | The BES PDF table leaves these three out, and 3 label spellings differ from the data; the data wins |

| D-09 | 2026-10-04 | Baselines: "always the most common category" and TF-IDF (letter chunks) + logistic regression with default settings, trained on W23–W24 with evaluation respondents left out | Gives the AI a fair, simple bar to beat; dev set only, test untouched |
