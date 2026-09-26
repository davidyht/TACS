import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build TyDiQA one-shot validation files from local train/dev JSON files."
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default="../data/eval/tydiqa",
        help="TyDiQA directory containing tydiqa-goldp-v1.1-train/dev.json",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_shot", type=int, default=1)
    ap.add_argument(
        "--write_less_one_shot",
        action="store_true",
        help="Also write tydiqa-one-shot.json for LESS validation loader.",
    )
    return ap.parse_args()


def extract_examples(squad_obj: Dict) -> List[Dict]:
    examples = []
    for article in squad_obj.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                qid = qa.get("id", "")
                examples.append(
                    {
                        "id": qid,
                        "lang": qid.split("-")[0] if "-" in qid else "english",
                        "context": context,
                        "question": qa.get("question", ""),
                        "answers": qa.get("answers", []),
                        "_article": article,
                    }
                )
    return examples


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    train_path = data_dir / "tydiqa-goldp-v1.1-train.json"
    dev_path = data_dir / "tydiqa-goldp-v1.1-dev.json"

    with train_path.open("r", encoding="utf-8") as f:
        train_obj = json.load(f)
    with dev_path.open("r", encoding="utf-8") as f:
        dev_obj = json.load(f)

    dev_examples = extract_examples(dev_obj)
    langs = sorted({x["lang"] for x in dev_examples})

    train_examples = extract_examples(train_obj)
    by_lang = {lang: [] for lang in langs}
    for ex in train_examples:
        if ex["lang"] in by_lang:
            by_lang[ex["lang"]].append(ex)

    picked_articles = []
    picked_for_less = {}
    for lang in langs:
        pool = by_lang[lang]
        if len(pool) < args.n_shot:
            raise ValueError(
                f"Not enough train examples for language '{lang}': "
                f"need {args.n_shot}, got {len(pool)}"
            )
        sampled = random.sample(pool, args.n_shot)
        picked_for_less[lang] = []
        for ex in sampled:
            picked_articles.append(ex["_article"])
            picked_for_less[lang].append(
                {
                    "id": ex["id"],
                    "lang": ex["lang"],
                    "context": ex["context"],
                    "question": ex["question"],
                    "answers": ex["answers"],
                }
            )

    one_shot_valid_dir = data_dir / "one-shot-valid"
    one_shot_valid_dir.mkdir(parents=True, exist_ok=True)

    out_dev = one_shot_valid_dir / "tydiqa-goldp-v1.1-dev.json"
    out_examples = one_shot_valid_dir / "tydiqa-goldp-v1.1-dev-examples.json"
    with out_dev.open("w", encoding="utf-8") as f:
        json.dump({"version": "tydiqa-goldp-v1.1", "data": picked_articles}, f, ensure_ascii=False)
    with out_examples.open("w", encoding="utf-8") as f:
        json.dump(picked_for_less, f, ensure_ascii=False)

    if args.write_less_one_shot:
        with (data_dir / "tydiqa-one-shot.json").open("w", encoding="utf-8") as f:
            json.dump(picked_for_less, f, ensure_ascii=False)

    print(f"Wrote one-shot eval file: {out_dev}")
    print(f"Wrote one-shot examples: {out_examples}")
    if args.write_less_one_shot:
        print(f"Wrote LESS one-shot file: {data_dir / 'tydiqa-one-shot.json'}")
    print(f"Languages: {langs}")


if __name__ == "__main__":
    main()
