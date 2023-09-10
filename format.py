import argparse
import json

parser = argparse.ArgumentParser(description="Formatting Tool",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--format", "-fm", action="store_true")
parser.add_argument("-tc", "--truncate", action="store", type=int)
parser.add_argument("-c", "--count", action="store_true")
parser.add_argument("src")
parser.add_argument("dest")

if __name__ == "__main__":
    args = parser.parse_args()
    org_list = []
    print(args)
    if args.format:
        with open(args.src, "r") as fp:
            for l in fp:
                record = json.loads(l)
                org_list.append(record)
    else:
        with open(args.src, "r") as fp:
            org_list = json.load(fp)

    if args.count:
        positive = 0
        negative = 0
        for l in org_list:
            if l["appropriate"]["valid"]:
                positive += 1
            else:
                negative += 1
        print("Positive: {pos}\nNegative: {neg}".format(pos=positive, neg=negative))

    result = list(org_list)
    if args.truncate:
        result = result[:args.truncate]

    with open(args.dest, "w") as fp:
        json.dump(result, fp)
