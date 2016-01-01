#!/usr/bin/python

# This file can be used to sort the scores of a set of experiments stored in the current directory
# The name of the directories to be considered for sorting is specified with -d (regular expressions accepted)
# Example: cd config; python ../scripts/sortScore.py -d config_rcv1_s.*
# Author: Aurelien Lucchi

import commands
import configIO
import itertools
import math
import os
import re
import time
from optparse import OptionParser

def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def trunc(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    slen = len('%.*f' % (n, f))
    return float(str(f)[:slen])

parser = OptionParser()
parser.add_option("-a", "--alg_type", dest="alg_type", default="sgd", help="Algorithm type")
parser.add_option("-d", "--dir_name", dest="dir_name", default="training_files_", help="Directory name")
parser.add_option("-e", "--exact_search", dest="exact_search", default=0, help="If 1, search for the exact directory name")
parser.add_option("-f", "--file_idx", dest="file_idx", default="0", help="Index of the experiment id to be displayed")
parser.add_option("-i", "--name_to_ignore", dest="name_to_ignore", default="", help="Name to ignore")
parser.add_option("-m", "--max_iteration_to_show", dest="max_iteration_to_show", default=-1, help="Maximum iterations to show")
parser.add_option("-o", "--obj_type", dest="obj_type", default=0, help="Objective type")
parser.add_option("-r", "--round_min_loss", dest="round_min_loss", default=0, help="round_min_loss")
parser.add_option("-s", "--loss_type", dest="loss_type", default=0, help="loss type. 0 = training, 1 = test, 2 = test with best training loss")
parser.add_option("-t", "--max_delta_time", dest="max_delta_time", default=-1, help="max_delta_time")

(options, args) = parser.parse_args()

dir_name = options.dir_name
dir_name = dir_name.replace('+', '.*')

max_iteration_to_show = int(options.max_iteration_to_show)
max_delta_time = int(options.max_delta_time)
showRecentFilesOnly = max_delta_time != -1
name_to_ignore = options.name_to_ignore
round_min_loss = int(options.round_min_loss) == 1
file_idx = options.file_idx
exact_search = int(options.exact_search) == 1

if int(options.obj_type) == 0:
    obj_name = 'error'
else:
    obj_name = 'loss'

loss_filename = options.alg_type + '_' + obj_name + file_idx + '.txt'
output_png_file = 'training_' + obj_name + '.png'
loss_type = int(options.loss_type)
if loss_type == 1:
    loss_filename = options.alg_type + '_val_' + obj_name + file_idx + '.txt'
    output_png_file = 'val_' + obj_name + '.png'

fig_idx = 1
list_min_loss = []
list_iteration_ids = []
list_n_iterations = []
list_sample_evalutions = []
list_times = []
labels = []
list_step_ids = []

list_files = []
recursive_mode = False
if recursive_mode == True:
    for _dir in os.listdir('.'):
        if os.path.isdir(_dir):
            for sub_dir in os.listdir(_dir):
                list_files.append(_dir + '/' + sub_dir)
        else:
            list_files.append(_dir)
else:
    for _dir in os.listdir('.'):
        list_files.append(_dir)
        
for _dir in list_files:
    if not os.path.isdir(_dir):
        continue
    if exact_search:
        if dir_name != _dir:
            continue
    else:
        m = re.search(dir_name, _dir)
        if m is None:
            continue

    if name_to_ignore != '':
        m = re.search(name_to_ignore, _dir)
        if m is not None:
            continue

    if showRecentFilesOnly:
        delta_time = time.time() - os.path.getmtime(_dir)
        if delta_time > max_delta_time:
            continue

    output_dir = _dir + '/'
    if not os.path.isdir(output_dir):
        continue

    min_loss = 1.0
    filename_min_loss = ''
    loss = [0]
    idx_loss = 0

    # Check if configuration file exists
    config_file_pattern = _dir[0:10]
    config_filename = ''
    for i in os.listdir(_dir):
        if i[0:len(config_file_pattern)] != config_file_pattern or i[len(i)-3:len(i)] != 'txt':
            continue
        config_filename = _dir + '/' + i

    loggingStep = 1
    
    # Uncomment this for back-compatibility with old logging files that did not contain the number of sample evaluations
    #if config_filename != '':
    #    loggingStep_str = configIO.read(config_filename, 'loggingStep')
    #	if loggingStep_str != '':
    #    	loggingStep = int(loggingStep_str)

    # Check if loss file exists 
    filename = ''
    for i in os.listdir(output_dir):
        if i[0:len(loss_filename)] != loss_filename:
            continue
        filename = output_dir + i

    if filename != '':
        f = open(filename)
        lines = f.readlines()

        list_loss = []
        iterationId = 0 # starts at 0 to skip first line that contains header information
        iterationId_min_loss = 0
        loggingId_min_loss = 0
        sample_evalutions = 0
        for l in lines:
            tokens = l.split()
            # back compatibility with old file format that used to contain the loss only.
            # New file format containts iteration number, loss
            if len(tokens) == 2:
                str_loss = tokens[1]
            else:
                str_loss = tokens[0]
            if isFloat(str_loss):
		loss = float(str_loss)
                list_loss.append(loss)
                idx_loss = idx_loss + 1
                if (loss < min_loss) or (iterationId == 0):
                    min_loss = loss
                    filename_min_loss = filename
                    iterationId_min_loss = iterationId
                    if len(tokens) == 2:
                        sample_evalutions = int(tokens[0])
                    else:
                        sample_evalutions = iterationId 

            iterationId = iterationId + 1
            #if max_iteration_to_show != -1 and (iterationId*loggingStep) > max_iteration_to_show:
	    if max_iteration_to_show != -1 and sample_evalutions > max_iteration_to_show:
                break
        f.close()

        if round_min_loss:
            min_loss = trunc(min_loss, 2)
            # re-iterate through the list to pick first index corresponding to the max value.
            for iterationId in range(0, len(lis_loss)):
                if list_loss[iterationId] > min_loss:
                    iterationId_min_loss = iterationId
                    break
                
        label_name = _dir
        labels.append(label_name)
        list_min_loss.append(min_loss)
        list_step_ids.append(iterationId_min_loss)
        list_iteration_ids.append(iterationId_min_loss*loggingStep)
        list_n_iterations.append(len(lines)*loggingStep)
        list_sample_evalutions.append(sample_evalutions)
        fig_idx = fig_idx + 1

order = [i[0] for i in sorted(enumerate(list_min_loss), key=lambda x:x[1])]

for i in range(0,len(order)):
    idx = order[i]
    sc = '%.5f' % list_min_loss[idx]
    iterationId = '%d' % list_iteration_ids[idx]
    n_iterations = '%d' % list_n_iterations[idx]
    nPointEvaluations = '%d' % list_sample_evalutions[idx]
    print labels[idx] + '\t\t' + iterationId + '/' + n_iterations + '\t\t' + nPointEvaluations + '\t\t' + sc

if loss_type == 2:
    #print '******************** TEST SET ********************'
    list_min_test_loss = []
    for idx in range(0,len(order)):
        iterationId = '%d' % list_iteration_ids[idx]

        _dir = labels[idx]

        filename = labels[idx] + '/' + options.alg_type + '_test_' + obj_name + file_idx + '.txt'
        if os.access(filename, os.F_OK):
            f = open(filename)
            lines = f.readlines()            
            if list_step_ids[idx] >= len(lines):
                # usually happens when algorithm is still writing results to the test file
                list_min_test_loss.append(0)
            else:
                l = lines[list_step_ids[idx]]
                tokens = l.split()
                if len(tokens) >= idx_test and isFloat(tokens[idx_test]):
                    list_min_test_loss.append(float(tokens[idx_test]))

    # Display according to sorted order
    print '******************** TEST SET (SORTED) ********************'
    order_test = [i[0] for i in sorted(enumerate(list_min_test_loss), key=lambda x:x[1])]
    for i in range(0,len(order_test)):
        idx = order_test[i]
        sc = '%.5f' % list_min_test_loss[idx]
        iterationId = '%d' % list_iteration_ids[idx]
        n_iterations = '%d' % list_n_iterations[idx]
        print labels[idx] + '\t\t' + iterationId + '/' + n_iterations + '\t\t' + sc
