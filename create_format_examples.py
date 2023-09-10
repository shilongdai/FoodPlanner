import json
import re


if __name__ == "__main__":
    src_data = "../data/meal-1693866518.json"
    output_data = "../data/format_examples.json"

    records = []
    with open(src_data, "r") as fp:
        for l in fp:
            records.append(json.loads(l))

    filter_regex = re.compile(r"Thought process:(.+)Final response:", flags=re.DOTALL)
    examples = []
    for r in records:
        matcher = filter_regex.match(r["appropriate"]["raw"])
        if matcher:
            appropriate_dict = dict(r["appropriate"])
            appropriate_dict["thought"] = matcher.group(1).strip()
            examples.append(appropriate_dict)

    with open(output_data, "w") as fp:
        json.dump(examples, fp)
