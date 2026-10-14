"""The command line: python -m coding_eval <command>. Run python -m coding_eval --help for the list."""
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="python -m coding_eval",
        description="Can a small local LLM code survey answers like trained human coders? "
                    "Settings are in config.toml.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    commands.add_parser("build", help="build the wave 25 evaluation set: dev 300, test 1,200 (needs the BES files in data/raw)")
    split_help = "dev while building; test once, for the final evaluation"
    p = commands.add_parser("baseline", help="train the TF-IDF classifier and code one split (needs the BES files)")
    p.add_argument("--split", choices=["dev", "test"], required=True, help=split_help)
    p = commands.add_parser("code", help="Llama codes one split through Ollama, saving after every answer")
    p.add_argument("--split", choices=["dev", "test"], required=True, help=split_help)
    p.add_argument("--prompt", choices=["A", "B"], default="B",
                   help="B: definitions plus examples, the frozen prompt (default); A: definitions only")
    p.add_argument("--limit", type=int, help="code only the first N answers (quick test)")
    commands.add_parser("review-sheet", help="the blind reading sheet of 30 of Llama's mistakes (holds answers: stays on your laptop)")
    commands.add_parser("metrics", help="recompute every file in results/ from the saved codes")
    args = parser.parse_args()

    if args.command == "build":
        from .data import build_eval_set
        build_eval_set()
    elif args.command == "baseline":
        from .baseline import train_and_predict
        train_and_predict(args.split)
    elif args.command == "code":
        from .llm_coder import code_answers
        code_answers(args.split, args.prompt, args.limit)
    elif args.command == "review-sheet":
        from .evaluate import review_sheet
        review_sheet()
    elif args.command == "metrics":
        from .evaluate import run_all
        run_all()


if __name__ == "__main__":
    main()
