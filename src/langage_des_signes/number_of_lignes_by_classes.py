import csv
from collections import Counter

path = r'C:\Users\josue\Desktop\TRM\rna\data\LangageDesSignes\data_formatted.csv'

counter = Counter()

with open(path) as f:
    reader = csv.reader(f)
    for row in reader:
        label = row[-5:]  # les 5 derniers éléments = one-hot
        class_id = label.index('1') + 1
        counter[class_id] += 1

print(counter)
