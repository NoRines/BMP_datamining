import csv

data = []

with open('data/adult_preprocess.csv') as f:
    csvreader = csv.reader(f, delimiter=',')

    for row in csvreader:
        data.append(row)


cols = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

outdata = []


for row in data:
    outdata.append([row[i] for i in cols])


with open('data/adult_wo_fnlwgt.csv', mode='w') as f:
    csvwriter = csv.writer(f, delimiter=',')

    for row in outdata:
        csvwriter.writerow(row)
