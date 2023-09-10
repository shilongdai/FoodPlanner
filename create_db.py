import argparse
import json
import chromadb
from meal import RecipeMealInput
from meal import format_recipes
import config


parser = argparse.ArgumentParser(description="Create Chroma DB",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("src")
parser.add_argument("dest")


def create_db_entries(records, col):
    cur_id = col.count()
    documents = []
    ids = []
    classified = []
    thought = []
    valid = []
    for r in records:
        recipe_combo = [RecipeMealInput(**p) for p in r["recipes"]]
        recipe_formatted = format_recipes(recipe_combo)
        documents.append(recipe_formatted)
        ids.append(str(cur_id))
        if r["changed"]:
            if r["appropriate"]["valid"]:
                classified.append("fneg")
            else:
                classified.append("fpos")
        else:
            classified.append("accurate")
        thought.append(r["appropriate"]["thought"])
        valid.append(r["appropriate"]["valid"])
        cur_id += 1

    return {
        "documents": documents,
        "ids": ids,
        "metadatas": [{"thought": t, "classified": c, "valid": v} for t, c, v in zip(thought, classified, valid)]
    }


if __name__ == "__main__":
    args = parser.parse_args()
    client = chromadb.PersistentClient(path=args.dest)

    with open(args.src, "r") as fp:
        records = json.load(fp)

    col = client.get_or_create_collection(name=config.MEAL_VALID_EXAMPLE_COL)
    col.add(**create_db_entries(records, col))
