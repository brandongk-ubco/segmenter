#!/usr/bin/env python

import os
import json
import csv
from collections import abc
import sys
from .get_configs import get_configs

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "../../output")
path = os.path.abspath(path)


def combine_directory(directory):

    configs = get_configs(directory)

    rows = []
    headers = set(["fold", "class", "key"])
    for root, dirs, files in os.walk(directory):

        relpath = root.replace(directory, "")[1:]
        tokenized_path = relpath.split("/")
        trainlogs = os.path.join(root, "logs", "train.csv")

        if "logs" in dirs and "early_stopping.json" in files and os.path.isfile(trainlogs):
            key = tokenized_path[0]
            confighash = tokenized_path[1]
            clazz = tokenized_path[2]
            fold = int(tokenized_path[3][4:])

            configrow = configs[confighash]

            with open(trainlogs, "r") as logfile:
                reader = csv.DictReader(logfile)
                for csvrow in reader:
                    row = {}
                    row.update(configrow)
                    row.update(csvrow)
                    row["class"] = clazz
                    row["fold"] = fold
                    row["key"] = key
                    rows.append(row)
                    for header in row.keys():
                        headers.add(header)

    if len(rows) == 0:
        return

    os.makedirs(os.path.join(directory, "results"), exist_ok=True)

    filepath = os.path.join(directory, "results", "train.csv")

    print("Writing %s" % filepath)
    with open(filepath, "w") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=sorted(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    combine_directory(path)
