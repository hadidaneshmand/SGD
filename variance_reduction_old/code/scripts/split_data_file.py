#!/usr/bin/python

# Split the input file into two files

import random
import sys

input_filename = sys.argv[1]
print input_filename

p = 0.5 # proportion of samples in dataset 1
if len(sys.argv) > 2:
    p = float(sys.argv[2])

print 'Proportion of samples for dataset 1 is ' + str(p)

output_file_1 = open(input_filename + ".out1", "w")
output_file_2 = open(input_filename + ".out2", "w")

with open(input_filename) as f:
    count = sum(1 for line in f)

print 'Total number of samples = ' + str(count)

count_1 = int(round(p * count))

print 'Number of samples for file 1 = ' + str(count_1)

shuffled_ids = range(0,count)
random.shuffle(shuffled_ids)

with open(input_filename, "r") as f:
    lines = f.readlines()
    for i in range(0, count_1):
        idx = shuffled_ids[i]
        output_file_1.write(lines[idx])
    for i in range(count_1, len(lines)):
        idx = shuffled_ids[i]
        output_file_2.write(lines[idx])


output_file_1.close()
output_file_2.close()
#input_file.close()
