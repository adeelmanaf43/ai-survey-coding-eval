# Decision log

| #     | Date       | Decision                                                                           | Why                                                                                     |
| ----- | ---------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| D-001 | 2026-10-03 | Run the model locally (Ollama); never send BES answers to cloud APIs or chat tools | BES terms: no passing data to third parties; GDPR                                       |
| D-002 | 2026-10-03 | Evaluate on wave 25 (May 2023)                                                     | BES codebook: waves 26+ were coded by GPT-4o, not humans                                |
| D-003 | 2026-10-03 | Model: Llama 3.1 8B (4.9 GB)                                                       | Already installed; fits the 3–8B plan; tested on this BES task in a 2025 study          |
| D-004 | 2026-10-03 | Keep the test set at 1,200                                                         | Median 6.7 s/answer (17 calls, 16 GB RAM, CPU/GPU: CPU) -> about 4.5 h for both schemes |
