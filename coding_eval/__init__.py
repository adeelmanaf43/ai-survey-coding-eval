"""Can a small local LLM code survey answers like trained human coders? (BES "most important issue")

Modules:
    config      settings from config.toml
    data        reading and writing files; building the evaluation set
    baseline    the TF-IDF + logistic regression classifier
    llm_coder   the prompt, Llama's replies and the Ollama run
    metrics     accuracy, macro-F1, kappa, McNemar, bootstrap intervals, review curve
    evaluate    every results/ file from the saved codes
    review_rule the review-rule curve, its cost and the chart
Run it with:  python -m coding_eval --help
"""
