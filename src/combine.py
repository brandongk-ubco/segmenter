import os
import json
import csv
import collections

output = os.environ.get("DIRECTORY", "/output")
outdir = os.path.abspath(output)

configs = {}

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_config(directory, confighash):
   
   if confighash not in configs:
      configpath = os.path.join(directory, confighash, "config.json")

      with open(configpath, "r") as configfile:
         config = json.load(configfile)

      configs[confighash] = flatten(config)

   return configs[confighash]

def is_complete(directory):
   with open(os.path.join(directory, "early_stopping.json"), "r") as json_file:
      early_stopping = json.load(json_file)
   return early_stopping["wait"] >= early_stopping["patience"]

def combine_directory(directory, key):
   rows = []
   headers = set(["fold", "class"])
   for root, dirs, files in os.walk(directory):
      if "logs" not in dirs:
         continue

      trainlogs = os.path.abspath(os.path.join(root, "logs", "train.csv"))

      if not os.path.isfile(trainlogs):
         continue

      confighash = root.split("/")[-3]
      clazz = root.split("/")[-2]
      fold = int(root.split("/")[-1][4:])
      configrow = get_config(directory, confighash)

      if not is_complete(root):
         print("WARNING: %s is not complete!" % os.path.join(root))

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
   
   print("Writing %s.csv" % directory)
   with open("%s.csv" % directory, "w") as outfile:
      writer = csv.DictWriter(outfile, fieldnames=sorted(headers))
      writer.writeheader()
      for row in rows:
         writer.writerow(row)

if __name__ == "__main__":
   for directory in [f for f in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, f))]:
      combine_directory(os.path.abspath(os.path.join(outdir, directory)), directory)

