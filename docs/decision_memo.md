# Decision memo: coding open-ended survey answers with a local AI model

**To:** Head of a survey team that codes about 10,000 open-ended answers a year by hand
**From:** Adeel Manaf
**Date:** 16 October 2026
**Subject:** Should we code open-ended answers automatically, and how?

## Recommendation

Code open-ended answers with two automatic methods, a local AI model (Llama) and a classifier trained on past codes, and have a trained coder check only the answers where the two disagree, about 5% of them. In a test on 1,200 BES answers, this gave about 99% agreement with the BES coders (best case), compared with 95% for Llama alone, while cutting human time from about 22 hours to 1–3.5 hours for 10,000 answers a year. **Condition:** these results come from short "most important issue" answers, so test the setup on a sample of your own questions before switching.

## Why

- **The AI model alone is already close to a trained coder.** With only a written codebook, Llama 3.1 8B gave the same code as the BES coders for 95.4% of answers (12 categories) and 95.9% (50 categories). Two trained coders agree on about 95%.
- **The two methods catch each other's mistakes.** They agree on 95% of answers, and those agreed codes are 98.7–99.2% correct. The 5% where they disagree hold most of Llama's mistakes (46 of 55 on 12 categories), so that is where a person's time is best spent.
- **No answer leaves your computers.** Everything runs on a 16 GB laptop, with no cloud service and no per-answer fees.

## Options compared (10,000 answers a year)

| Option                                               | Agreement with the BES coders | Human time         | Cost at £15/h | Note                                              |
| ---------------------------------------------------- | ----------------------------- | ------------------ | ------------- | ------------------------------------------------- |
| All by hand                                          | about 95% (between two coders) | about 22 hours     | about £340    | Today's method                                    |
| Llama alone                                          | 95–96%                        | none               | none          | No past codes needed                              |
| Classifier alone                                     | 96–97%                        | none               | none          | Needs past codes; weaker on new wording (75–81%)  |
| **Both, with a trained coder checking disagreements** | **about 99% (best case)**     | **1–3.5 hours**    | **£18–53**    | **Recommended**                                   |

The automatic options also need about 16–22 hours of unattended laptop time a year. Human time is based on my own coding pace in a timed test (8 seconds per answer on average); checking a disagreement is assumed to take one to three times as long, because disagreements are the hard answers.

## Risks and how to handle them

| Risk | How to handle it |
| --- | --- |
| Other questions are harder: longer answers that mention several issues | Pilot on a few hundred of your own answers, coded by hand as well, before switching |
| The 99% assumes the checker always gives the right code | Use trained coders for the check. In my test, an untrained coder with the same codebook agreed with the BES coders on only 83% |
| Some mistakes are shared by both methods and never reach a person (9–15 per 1,200 answers) | Spot-check a small random sample of the answers both methods agree on |
| No past codes for a new question, so no classifier | Start with Llama plus a check of a random sample; train the classifier once a few thousand answers are coded |
| A software update changes the model's behaviour | Fix the versions, and re-run a small set of already-coded answers after any update |

## Decision needed

Approve a pilot on the next wave: run the setup on 500 answers alongside normal hand coding, then compare agreement and time. Switch if agreement with the hand codes is at least 98% and the time saving holds.

Full method, results and code: [README](../README.md) · every design decision: [decision log](decision_log.md)
