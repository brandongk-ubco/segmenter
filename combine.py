import os
import json
import csv

outdir = os.environ.get("DIRECTORY", "./output")
outdir = os.path.abspath(outdir)

configs = {}
rows = []
headers = set(["fold", "class"])

for root, dirs, files in os.walk(outdir):
   if "train.csv" not in files:
      continue
   trainlogs = os.path.abspath(os.path.join(root, "train.csv"))
   logstoken = trainlogs.replace(outdir, "")[1:].split("/")[:3]
   confighash = logstoken[0]
   if confighash not in configs:
     configpath = os.path.join(outdir, confighash, "config.json")
     with open(configpath, "r") as configfile:
        config = json.load(configfile)
     configs[confighash] = config
   configrow = {}
   configrow.update(configs[confighash])

   print(trainlogs)
   with open(trainlogs, "r") as logfile:
      reader = csv.DictReader(logfile)
      for csvrow in reader:
         row = {}
         row.update(configrow)
         row.update(csvrow)
         row["class"] = logstoken[1]
         row["fold"] = int(logstoken[2][4:])
         rows.append(row)
         for header in row.keys():
            headers.add(header)

with open("./combined.csv", "w") as outfile:
  writer = csv.DictWriter(outfile, fieldnames=sorted(headers))
  writer.writeheader()
  for row in rows:
     writer.writerow(row)
