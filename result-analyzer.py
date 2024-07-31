from sk import *
import csv
def csv_to_dict(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        # Reading the CSV file
        csv_reader = csv.reader(file)
        # Extracting the header (first row) as the keys
        keys = [ str.strip(i) for i in next(csv_reader)]

        # Initializing the dictionary with keys and empty lists as values
        data_dict = {key: [] for key in keys}
        # Populating the dictionary with data from each row
        for row in csv_reader:
            for key, value in zip(keys, row):
                try:
                    data_dict[key].append(float(str.strip(value)))
                except:
                    pass
    
    return data_dict

def changeOrder(results):
    columnOrdered = {}
    for treatmentKey,treatres in results.items():
        for col,res in treatres.items():
            try:
                columnOrdered[col].update({treatmentKey:res})
            except:
                columnOrdered[col] = {treatmentKey:res}

    return columnOrdered



import os

resultLists = {}
for filename in os.listdir('reg results/wine_quality/'):
    if filename[-4:]=='.csv':
        resultLists[filename[:-4]] = csv_to_dict('reg results/wine_quality/'+filename)

results = changeOrder(resultLists)
for col,res in results.items():
    if col != 'time':
        print(f"For Column {col}")
        Rx.show(Rx.sk(Rx.data(**res)))
        print('\n\n')
    else:
        [print(f"{t} : \t{time} seconds") for t,time in res.items()]