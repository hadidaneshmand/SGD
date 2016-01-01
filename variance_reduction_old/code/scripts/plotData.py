import collections
import commands
import os
import re
import numpy as np
from pylab import *
import itertools
import time

def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_nColumnsFromFile(filename):
    f = open(filename)
    lines = f.readlines()

    if len(lines) == 0:
        return 0

    # check if first line is a header
    first_idx = 0
    l = lines[0]
    l_s = l.split()
    if not isFloat(l_s[0]):
        #print l_s[0] + ' is not a digit'
        first_idx = 1

    l = lines[first_idx]
    l_s = l.split()
    f.close()

    return len(l_s)

def readColumnFromFile(col_idx, filename):
    f = open(filename)
    lines = f.readlines()

    if len(lines) == 0:
        return []

    # check if first line is a header
    first_idx = 0
    l = lines[0]
    l_s = l.split()
    if not isFloat(l_s[0]):
        #print l_s[0] + ' is not a digit'
        first_idx = 1

    objs = [0 for x in range(len(lines)-first_idx)]
    obj_idx = 0
    for li in range(first_idx, len(lines)):
        l = lines[li]
        l_s = l.split()
        if len(l_s) == 0:
            break
        if len(l_s) <= col_idx:
            obj = 0
        else:
            obj = float(l_s[col_idx])
        objs[obj_idx] = obj
        obj_idx = obj_idx + 1
    f.close()
    return objs

def getListCs():
    colors = ['b','g','r','c','m','y','k']
    styles = ['-','--','-.',':','.','o','v','^','v']
    listOfLists = [styles, colors]
    list_cs = list(itertools.product(*listOfLists))    
    return list_cs

def plotData_Std(dir_names, filename, show_legend = 1, col_idx = 0, coeff_ylim = -1, exact_search = 0):

    current_time = time.time()

    filename_noext, ext = os.path.splitext(filename)
    ldirs = os.listdir('.')
    ndirs = len(ldirs)

    fig_idx = 0
    list_means = []
    list_stds = []
    list_iteration_ids = []
    labels = []

    for dir_name in dir_names:

        for _dir in ldirs:

            if not os.path.isdir(_dir):
                continue

	    if exact_search:
		if dir_name != _dir:
		    continue
	    else:
		m = re.search(dir_name, _dir)
		if m is None:
		    continue

            lfiles = os.listdir(_dir)
            list_objs_for_dir = []
            list_iteraton_ids_for_dir = []
            for _filename in lfiles:

                fullpath = _dir + '/' + _filename

                m = re.search(filename, fullpath)
                if m is None:
                    #print dir_name + ' not found in ' + _dir
                    continue

                if os.path.getsize(fullpath) == 0:
                    continue

                nColumns = get_nColumnsFromFile(fullpath)
                if nColumns == 1:
                    objs = readColumnFromFile(col_idx, fullpath)
                    objs = [1.0 if math.isnan(x) else x for x in objs]
                    list_objs_for_dir.append(objs)
                else:
                    # read first column that contains iteration ids
                    iteraton_ids = readColumnFromFile(0, fullpath)
                    list_iteraton_ids_for_dir.append(iteraton_ids)

                    objs = readColumnFromFile(1, fullpath)
                    objs = [1.0 if math.isnan(x) else x for x in objs]
                    # Only add to the list of objects if there is more than 1 line
                    if len(objs) > 1:
	                list_objs_for_dir.append(objs)

            if len(list_objs_for_dir) > 0:
                min_len = len(list_objs_for_dir[0])
                for i in range(1, len(list_objs_for_dir)):
                    if min_len > len(list_objs_for_dir[i]):
                        min_len = len(list_objs_for_dir[i])

                for i in range(0, len(list_objs_for_dir)):
                    list_objs_for_dir[i] = list_objs_for_dir[i][0:min_len]

                mean_obj = np.mean(list_objs_for_dir, axis=(0))
                std_obj = np.std(list_objs_for_dir, axis=(0))
                list_means.append(mean_obj)
                list_stds.append(std_obj)

                label_name = _dir
                labels.append(label_name)

                # todo: iterate over all the list of iteration ids to make sure they are consistent
                if len(list_iteraton_ids_for_dir) > 0:

                    longest_list_idx = 0
                    longest_list_length = 0
                    for i in range(0, len(list_iteraton_ids_for_dir)):
                        if len(list_iteraton_ids_for_dir[i]) > longest_list_length:
                            longest_list_length = len(list_iteraton_ids_for_dir[i])
                            longest_list_idx = i

                    list_iteration_ids.append(list_iteraton_ids_for_dir[longest_list_idx])

                fig_idx = fig_idx + 1

    common_label = os.path.commonprefix(labels)

    for i in range(0,fig_idx):
        labels[i] = ''.join(labels[i].split(common_label))

    n_colors = len(list_means)
    colorarray = np.random.random_sample((n_colors, 3))
    colorarray[:, -1] = 0.6
    colorarray = np.vstack([colorarray, colorarray])

    # Output curves
    if fig_idx > 0:

        min_y = 99999999999
        # Subtract the minimum value
        for i in range(0,len(list_means)):
	    if any(isnan(list_means[i])):
	        continue

            l = len(list_means[i])
            if len(list_iteration_ids) > 0:
                x = list_iteration_ids[i][0:l]
            else:
                x = range(0,l)
            y = list_means[i][0:l]
            min_y = min(min_y, min(y))

        min_y = 0

        print 'min_y ' + str(min_y)

        # Make plots
        list_cs = getListCs()
        fig = figure(num=fig_idx, figsize=(12, 10), dpi=100)
        clf
        ax = fig.add_subplot(1, 1, 1)
        for i in range(0,len(list_means)):
	    if any(isnan(list_means[i])):
	        continue

            l = len(list_means[i])
            cs = list_cs[i][0] + list_cs[i][1]
            if len(list_iteration_ids) > 0:
                x = list_iteration_ids[i][0:l]
            else:
                x = range(0,l)
            y = list_means[i][0:l]
            y = y - min_y + (1e-3*min_y)

            max_y = 1e8
            bb = y>max_y
            print 'before ' + str(sum(bb))

            y[y == inf] = max_y
            y[y == -inf] = max_y
            y[y > max_y] = max_y

            bb = y>max_y
            print 'after ' + str(sum(bb))

            e = list_stds[i][0:l]
            # Overlay lines and error bars
            base_line, = ax.plot(x, y, linewidth=1.0, color = colorarray[i])
            #errorbar(x, y, e, linestyle='None', fmt='o', color = colorarray[i])

            ax.fill_between(x, y - e, y + e, alpha=0.5, facecolor=base_line.get_color())
	    if np.sum(y) != 0:
                ax.set_yscale('log')
            ax.legend(loc=0)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Objective Value")
            #ax.set_title("")

            if coeff_ylim > 0:
                mean_y = np.mean(y)
                std_y = np.std(y)
                ylim([mean_y - coeff_ylim*std_y, mean_y + coeff_ylim*std_y])

        if show_legend == 1:
            legend(labels, loc='upper right')
        figtext(.3,.05,common_label)
        if col_idx == 0:
            output_file = os.path.basename(filename_noext) + '.png'
        else:
            output_file = os.path.basename(filename_noext) + str(col_idx) + '.png'
        print 'output_file ' + output_file
        try:
            savefig(output_file)
        except:
            pass
        clf
        close(fig_idx)

    list_objs_collection = collections.namedtuple('list', ['id', 'mean', 'std', 'legend'])
    list_objs = list_objs_collection(list_iteration_ids, list_means, list_stds, labels)
    return list_objs

def computeDerivative(dir_name, filename, col_idx = 0, exact_search = 0):

    current_time = time.time()

    filename_noext, ext = os.path.splitext(filename)
    ldirs = os.listdir('.')
    ndirs = len(ldirs)

    fig_idx = 0
    list_means = []
    list_stds = []
    list_iteration_ids = []


    for _dir in ldirs:

        if not os.path.isdir(_dir):
            continue

        if exact_search:
            if dir_name != _dir:
                continue
        else:
            m = re.search(dir_name, _dir)
            #print 'H ' + dir_name + ' ' + _dir + ' ' + str(m is None)
            if m is None:
                continue

        lfiles = os.listdir(_dir)
        list_objs_for_dir = []
        list_iteraton_ids_for_dir = []
        for _filename in lfiles:

            fullpath = _dir + '/' + _filename

            m = re.search(filename, fullpath)
            if m is None:
                # print dir_name + ' not found in ' + _dir
                continue

            if os.path.getsize(fullpath) == 0:
                continue

            nColumns = get_nColumnsFromFile(fullpath)
            if nColumns > 1:
                objs = readColumnFromFile(col_idx, fullpath)
                objs = [1.0 if math.isnan(x) else x for x in objs]
                list_objs_for_dir.append(objs)
            else:
                # read first column that contains iteration ids
                iteraton_ids = readColumnFromFile(0, fullpath)
                list_iteraton_ids_for_dir.append(iteraton_ids)

                objs = readColumnFromFile(1, fullpath)
                objs = [1.0 if math.isnan(x) else x for x in objs]
                list_objs_for_dir.append(objs)

        if len(list_objs_for_dir) > 0:
            min_len = len(list_objs_for_dir[0])
            for i in range(1, len(list_objs_for_dir)):
                if min_len > len(list_objs_for_dir[i]):
                    min_len = len(list_objs_for_dir[i])

            for i in range(0, len(list_objs_for_dir)):
                list_objs_for_dir[i] = list_objs_for_dir[i][0:min_len]

            mean_obj = np.mean(list_objs_for_dir, axis=(0))
            list_means.append(mean_obj)

            # todo: iterate over all the list of iteration ids to make sure they are consistent
            if len(list_iteraton_ids_for_dir) > 0:

                longest_list_idx = 0
                longest_list_length = 0
                for i in range(0, len(list_iteraton_ids_for_dir)):
                    if len(list_iteraton_ids_for_dir[i]) > longest_list_length:
                        longest_list_length = len(list_iteraton_ids_for_dir[i])
                        longest_list_idx = i

                list_iteration_ids.append(list_iteraton_ids_for_dir[longest_list_idx])

            fig_idx = fig_idx + 1



    # Output curves
    res = 0
    for i in range(0,len(list_means)):
        if any(isnan(list_means[i])):
            continue

        l = len(list_means[i])
        if len(list_iteration_ids) > 0:
            x = list_iteration_ids[i][0:l]
        else:
            x = range(0,l)
        y = list_means[i][0:l]
        res = res + sum(np.absolute(np.diff(y)))

    return res

def findConvergencePoint(fullpath, col_idx = 0, exact_search = 0):

    list_means = []
    list_iteration_ids = []

    list_objs_for_dir = []
    list_iteraton_ids_for_dir = []

    # read first column that contains iteration ids
    iteraton_ids = readColumnFromFile(0, fullpath)
    list_iteraton_ids_for_dir.append(iteraton_ids)

    objs = readColumnFromFile(col_idx, fullpath)
    objs = [1.0 if math.isnan(x) else x for x in objs]
    list_objs_for_dir.append(objs)

    if len(list_objs_for_dir) > 0:
        min_len = len(list_objs_for_dir[0])
        for i in range(1, len(list_objs_for_dir)):
            if min_len > len(list_objs_for_dir[i]):
                min_len = len(list_objs_for_dir[i])

        for i in range(0, len(list_objs_for_dir)):
            list_objs_for_dir[i] = list_objs_for_dir[i][0:min_len]

        mean_obj = np.mean(list_objs_for_dir, axis=(0))
        list_means.append(mean_obj)

        # todo: iterate over all the list of iteration ids to make sure they are consistent
        if len(list_iteraton_ids_for_dir) > 0:

            longest_list_idx = 0
            longest_list_length = 0
            for i in range(0, len(list_iteraton_ids_for_dir)):
                if len(list_iteraton_ids_for_dir[i]) > longest_list_length:
                    longest_list_length = len(list_iteraton_ids_for_dir[i])
                    longest_list_idx = i

            list_iteration_ids.append(list_iteraton_ids_for_dir[longest_list_idx])

    # Output curves
    list_cum = []
    for i in range(0,len(list_means)):
        if any(isnan(list_means[i])):
            continue

        l = len(list_means[i])
        if len(list_iteration_ids) > 0:
            x = list_iteration_ids[i][0:l]
        else:
            x = range(0,l)
        y = list_means[i][0:l]
        list_cum.append(np.absolute(np.diff(y)))

    #print list_cum
    # print np.cumsum(list_cum)
    L = np.diff(np.cumsum(list_cum))
    # print L
    res = 0
    try:
        res = next(x[0] for x in enumerate(L) if x[1] < 0.002)
    except StopIteration:
        res = -1
        # do nothing

    return list_iteration_ids[0][res]

def findConvergencePoint_dir(dir_name, filename, col_idx = 0, exact_search = 0):

    current_time = time.time()

    filename_noext, ext = os.path.splitext(filename)
    ldirs = os.listdir('.')
    ndirs = len(ldirs)

    fig_idx = 0
    list_means = []
    list_iteration_ids = []

    for _dir in ldirs:

        if not os.path.isdir(_dir):
            continue

        if exact_search:
            #print 'dir ' + _dir + ' ' + dir_name + ' ' + str(dir_name != _dir)
            if dir_name != _dir:
                continue
        else:
            m = re.search(dir_name, _dir)
            #print 'H ' + dir_name + ' ' + _dir + ' ' + str(m is None)
            if m is None:
                continue

        lfiles = os.listdir(_dir)
        list_objs_for_dir = []
        list_iteraton_ids_for_dir = []
        for _filename in lfiles:

            fullpath = _dir + '/' + _filename

            m = re.search(filename, fullpath)
            if m is None:
                # print dir_name + ' not found in ' + _dir
                continue

            if os.path.getsize(fullpath) == 0:
                continue

            # read first column that contains iteration ids
            iteraton_ids = readColumnFromFile(0, fullpath)
            list_iteraton_ids_for_dir.append(iteraton_ids)

            objs = readColumnFromFile(col_idx, fullpath)
            objs = [1.0 if math.isnan(x) else x for x in objs]
            list_objs_for_dir.append(objs)

        if len(list_objs_for_dir) > 0:
            min_len = len(list_objs_for_dir[0])
            for i in range(1, len(list_objs_for_dir)):
                if min_len > len(list_objs_for_dir[i]):
                    min_len = len(list_objs_for_dir[i])

            for i in range(0, len(list_objs_for_dir)):
                list_objs_for_dir[i] = list_objs_for_dir[i][0:min_len]

            mean_obj = np.mean(list_objs_for_dir, axis=(0))
            list_means.append(mean_obj)

            # todo: iterate over all the list of iteration ids to make sure they are consistent
            if len(list_iteraton_ids_for_dir) > 0:

                longest_list_idx = 0
                longest_list_length = 0
                for i in range(0, len(list_iteraton_ids_for_dir)):
                    if len(list_iteraton_ids_for_dir[i]) > longest_list_length:
                        longest_list_length = len(list_iteraton_ids_for_dir[i])
                        longest_list_idx = i

                list_iteration_ids.append(list_iteraton_ids_for_dir[longest_list_idx])



    # Output curves
    list_cum = []
    for i in range(0,len(list_means)):
        if any(isnan(list_means[i])):
            continue

        l = len(list_means[i])
        if len(list_iteration_ids) > 0:
            x = list_iteration_ids[i][0:l]
        else:
            x = range(0,l)
        y = list_means[i][0:l]
        list_cum.append(np.absolute(np.diff(y)))


    #print np.cumsum(list_cum)
    L = np.diff(np.cumsum(list_cum))
    # print L
    res = 0
    try:
        res = next(x[0] for x in enumerate(L) if x[1] < 0.001)
        res = list_iteration_ids[0][res]
    except StopIteration:
        res = -1
        # do nothing

    return res
     
