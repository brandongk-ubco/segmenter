import os
import json
import csv
from collections import abc
import sys
from .get_configs import get_configs

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "../")
path = os.path.abspath(path)


def combine_directory(directory):

    configs = get_configs(directory)

    rows = []
    headers = set(["folds", "class", "key", "type"])
    for root, _dirs, files in os.walk(directory):

        relpath = root.replace(directory, "")[1:]
        tokenized_path = relpath.split("/")

        if "in_class_results.json" in files:

            key = tokenized_path[0]
            confighash = tokenized_path[1]
            clazz = tokenized_path[2]
            folds = [int(f[4:]) for f in tokenized_path[4].split("-")]

            configrow = configs[confighash]

            in_class = os.path.join(root, "in_class_results.json")
            # out_of_class = os.path.join(root, "out_of_class_results.json")
            for k, filepath in [("in_class", in_class)]:
                with open(filepath, "r") as logfile:
                    results = json.load(logfile)
                    row = {}
                    row.update(configrow)
                    row.update(results)
                    row["class"] = clazz
                    row["folds"] = folds
                    row["key"] = key
                    row["type"] = k
                    rows.append(row)
                    for header in row.keys():
                        headers.add(header)

    if len(rows) == 0:
        return

    os.makedirs(os.path.join(directory, "results"), exist_ok=True)
    filepath = os.path.join(directory, "results", "eval.csv")

    print("Writing %s" % filepath)
    with open(filepath, "w") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=sorted(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    combine_directory(path)
