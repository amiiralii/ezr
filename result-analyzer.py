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
                data_dict[key].append(float(str.strip(value)))
    
    return data_dict

# Example usage:
file_path = 'reg-dists.csv'
result_reg = csv_to_dict(file_path)

file_path = 'lr.csv'
result_lr = csv_to_dict(file_path)

file_path = 'reg-loglike.csv'
result_reg2 = csv_to_dict(file_path)

for i,j in result_reg.items():
    a = {}
    a['reg-dists'] = j
    a['lr'] = result_lr[i]
    a['reg-loglike'] = result_reg2[i]
    print(f"For Column {i}")
    Rx.show(Rx.sk(Rx.data(**a)))
    print('\n\n')