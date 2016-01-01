#!/usr/bin/python

# Script to convert libsvm a file with -1/1 class labels to a file with 0/1 class labels.

import sys

input_filename = sys.argv[1]
print input_filename

output_file = open(input_filename + ".out", "w")

with open(input_filename, "r") as f:
    for line in f:
        tokens = line.split(' ')
        cl = tokens[0]
        n = len(cl)
        if cl == '-1':
            new_cl = '0'
        else:
            new_cl = '1'
        new_line = new_cl + line[n:]
        output_file.write(new_line)

output_file.close()
#input_file.close()
