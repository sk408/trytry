import glob
import re

for f in glob.glob("src/**/*.py", recursive=True):
    with open(f, encoding="utf8") as file:
        for i, l in enumerate(file):
            if ">" in l and ("'" in l or '"' in l):
                print(f"{f}:{i+1}:{l.strip()}")
