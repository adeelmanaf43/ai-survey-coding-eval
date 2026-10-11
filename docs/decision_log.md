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

| D-010 | 2026-10-04 | Report accuracy separately for answers seen word for word in training vs new answers | Most answers repeat phrases the classifier has memorised; the split shows where its accuracy comes from |

| D-011 | 2026-10-05 | Llama 3.1 8B via Ollama: temperature 0, seed 2026, answers restricted to codebook names, codebook in the system message, one scheme at a time, saved after every answer | Reproducible; always a valid category; the codebook is read once and reused (about 7 s per answer instead of minutes); the run can resume after a stop |

| D-012 | 2026-10-05 | Froze the codebooks and prompt after the first dev run (prompt versions 459a2e64 and b0fcfe68) | Dev results were at the ceiling (~96%); more tuning on 300 answers would overfit, and the test run must use exactly what was tested on dev |

| D-013 | 2026-10-08 | Test "second-opinion" routing: answers where Llama and the classifier disagree go to a person; measure the workload and the final accuracy | Llama's own "unsure" flag caught nothing; disagreement between two different methods is a cheap, practical warning sign |

| D-0174| 2026-10-09 | Run Llama on the CPU only (`"num_gpu": 0`) for the test run | Ollama updated itself from 0.35.0 (dev run) to 0.35.1, which put 3 of 33 model layers on the laptop's old graphics chip (Quadro M1000M, via Vulkan); the graphics driver crashed ("device lost") on the codebook prompt. A one-change-at-a-time check (scripts/diagnose_ollama.py) showed only the GPU path crashed. On the CPU only, the crashing answer worked and 10 of 10 dev answers got the same codes as in the dev run. Prompt versions unchanged (459a2e64, b0fcfe68) |

| D-018 | 2026-10-11 | Human benchmark: I coded a random 100 of the 1,200 test answers (seed 2026) myself with the 12-category codebook, blind to the BES codes and model outputs, timing each answer | A human reference point on the same answers as the models, and minutes per answer for the cost estimate |
