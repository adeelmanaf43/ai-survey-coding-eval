# Codebook review (4 Oct 2026)

I reviewed codebook_50.json against the 300 wave 25 dev answers with an AI assistant, focusing on the largest categories and look-alike pairs. The assistant proposed the changes below, and I checked each one before accepting it (D-007). Sharing the dev answers with the assistant went against D-001 and is recorded as D-009.

Changes:

- Pol-neg vs Partisan-neg: a named party wins, even when the answer is about corruption; an unnamed "government" is Pol-neg.
- Racism/discrimination vs Inequality: class-based unfairness is Inequality.
- Housing vs Living costs: housing costs are Housing.
- Immigration vs Asylum: illegal immigration is Immigration; only answers naming asylum seekers or refugees are Asylum.
- Austerity: public services and their funding in general, not only cuts.
- International trade: self-sufficiency, such as producing our own energy.
- Pol values-left: complaints that politics is too right-wing.
- Smaller: mental health (Health), employment (Unemployment), drug policy (Crime), cyber security (Defence), relationship with Europe (Europe).
- Matching edits in codebook_12.json: Health, Inequality, Other.

Not changed: one dev answer that was only the word "corruption" was coded Morals, while the BES guide puts corruption under Pol-neg. One example isn't enough to change a rule (that would be overfitting to the dev set), and it shows human coders aren't perfectly consistent either.
