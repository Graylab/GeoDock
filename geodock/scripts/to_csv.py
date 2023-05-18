import csv

list1 = ["db5_test_unbound_0", "db5_test_unbound_1", "db5_test_unbound_2", "db5_test_unbound_3"]  # replace with your input file name
list2 = ["db5_test_bound_0", "db5_test_bound_1", "db5_test_bound_2", "db5_test_bound_3"]# replace with your input file name
names = list1 + list2
dir_name = "../ground_truth/db5_test_bound/"
ext_name = "_gt.pdb"

for name in names:
    with open(name+'.txt', "r") as f_in, open(name+'.csv', "w", newline="") as f_out:
        writer = csv.writer(f_out)
        cols = []
        for line in f_in:
            line_cols = line.strip().split(" ")
            pairs = [(line_cols[i], line_cols[i+1]) for i in range(0, len(line_cols)-2, 2)]
            pairs.append(('id', line_cols[-1][len(dir_name):-len(ext_name)]))
            if not cols:
                cols = [pair[0] for pair in pairs]
                writer.writerow(cols)
            writer.writerow([pair[1] for pair in pairs])

