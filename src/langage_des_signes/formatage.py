import csv

data_path = r'C:\Users\josue\Desktop\TRM\rna\data\LangageDesSignes\data.csv'


def one_hot(label):
    vector = [0] * 5
    vector[int(label) - 1] = 1
    return vector


with open(data_path, 'r') as infile, open('data_formatted.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Skip the header row
    next(reader)

    for row in reader:
        label = row[0]
        features = row[1:]
        one_hot_vector = one_hot(label)
        writer.writerow(features + [str(x) for x in one_hot_vector])