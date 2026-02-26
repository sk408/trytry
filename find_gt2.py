import re

with open("gt_results.txt", encoding="utf-16") as f:
    for line in f:
        if re.search(r"[><=]\s*['\"]", line):
            print(line.strip())
