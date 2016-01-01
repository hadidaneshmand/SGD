#!/usr/bin/python

import configIO
import itertools
import os
import Queue
import signal
import subprocess
import sys
import time
import thread
import threading
import datetime

# control-c signal handler
def signal_handler(signal, frame):
    global threads
    print ""
    print ""
    print 'Control-C pressed, quitting...'
    
    for t in threads:
        t.join()
    sys.exit(0)
        
signal.signal(signal.SIGINT, signal_handler)

class  QueueThread(threading.Thread):
    def __init__(self, _alg_type, _eta0, _t0, _lambda, _svrg_pSamples, _npasses, _nsamplesperpass, _svrg_outer_pSamples, _hessian_pSamples):
        threading.Thread.__init__(self)
        self.alg_type = _alg_type
        self.eta0 = _eta0
        self.t0 = _t0
        self.lambda0 = _lambda
        self.svrg_pSamples = _svrg_pSamples
        self.npasses = _npasses
        self.nsamplesperpass = _nsamplesperpass
        self.svrg_outer_pSamples = _svrg_outer_pSamples
        self.hessian_pSamples = _hessian_pSamples

    def run(self):

        s = open("log_batch.txt","a")
        s.write("acquire\n")
        s.close()
        threadLimiter.acquire()

        try:

            self.output_dir = output_dir_prefix
            self.output_dir = self.output_dir + '_a' + str(self.alg_type)
            self.output_dir = self.output_dir + '_s' + str(self.eta0)
            self.output_dir = self.output_dir + '_t' + str(self.t0)
            self.output_dir = self.output_dir + '_l' + str(self.lambda0)

            # sampling methods
            self.output_dir = self.output_dir + '_p' + str(self.svrg_pSamples)

            self.output_dir = self.output_dir + '_o' + str(self.svrg_outer_pSamples)

            self.output_dir = self.output_dir + '_e' + str(self.npasses)

            self.output_dir = self.output_dir + '_n' + str(self.nsamplesperpass)

            self.output_dir = self.output_dir + '_h' + str(self.hessian_pSamples)

            #self.output_dir = self.output_dir + '_b1000'

            print self.output_dir

            overwrite_dir = False
            existing_dir = os.access(self.output_dir, os.F_OK)
            self.cmd = ''

            # Create output directory if it does not exist. 
            if not existing_dir or overwrite_dir:

                if not existing_dir:
                    print 'mkdir ' + self.output_dir
                    if exec_cmd:
                        os.mkdir(self.output_dir)

                # Copy config file
                self.cmd = 'cp ' + input_file_fullpath + ' ' + self.output_dir
                print self.cmd
                if exec_cmd:
                    os.system(self.cmd)

                # Change directory
                print 'cd ' + self.output_dir
                if exec_cmd:
                    os.chdir(self.output_dir)

                    configIO.write(input_file,'alg_type', self.alg_type)
                    configIO.write(input_file,'eta0', self.eta0)
                    configIO.write(input_file,'T0', self.t0)
                    configIO.write(input_file,'lambda', self.lambda0)
                    configIO.write(input_file,'svrg_pSamples', self.svrg_pSamples)
                    configIO.write(input_file,'svrg_outer_pSamples', self.svrg_outer_pSamples)
                    configIO.write(input_file,'npasses', self.npasses)
                    configIO.write(input_file,'nsamplesperpass', self.nsamplesperpass)
                    configIO.write(input_file,'hessian_psamples', self.hessian_pSamples)

                    datapath_rel = configIO.read(input_file,'datapath')
                    configIO.write(input_file,'datapath', data_path + datapath_rel)

                    configIO.write(input_file,'logdir', '')

                    # bsub -We 100:00 java
                    #self.cmd = 'bsub -We 100:00 -R rusage[mem=10000] java -cp .:' + lib_path + ':' + data_path + 'src/ optimize ' + input_file
                    self.cmd = 'java -cp .:' + lib_path + ':' + data_path + 'src/ optimize ' + input_file
                    if redirect_output:
                        self.cmd = self.cmd + ' > out.txt 2>&1'
                    print self.cmd
                    if exec_cmd:
                        os.system(self.cmd)
            
        finally:
            s = open("log_batch.txt","a")
            s.write("release " + self.cmd + "\n")
            s.close()
            threadLimiter.release()


if len(sys.argv)<2:
    print 'Script name has to be passed as argument'
    sys.exit(-1)

input_file_fullpath = sys.argv[1]
input_file = os.path.basename(input_file_fullpath)

exec_cmd = False
if len(sys.argv)>2:
    if sys.argv[2] == 'True' or sys.argv[2] != '0':
        print 'exec_cmd is set to TRUE'
        exec_cmd = True
    elif sys.argv[2] == 'False' or sys.argv[2] == '0':
        print 'exec_cmd is set to FALSE'
        #exec_cmd = False
    else:
        print 'Unknown second parameter'

redirect_output = False
if len(sys.argv)>3:
    if sys.argv[3] == 'True' or sys.argv[3] == '1':
        print 'redirect_output is set to TRUE'
        redirect_output = True
    elif sys.argv[3] == 'False' or sys.argv[3] == '0':
        print 'redirect_output is set to FALSE'
        #redirect_output = False
    else:
        print 'Unknown third parameter'

output_dir_prefix, extension = os.path.splitext(input_file_fullpath)
output_dir_prefix = os.getcwd() + '/' + output_dir_prefix

#data_path = '/root/sgd/'
data_path = '/cluster/home/alucchi/sgd/'
#data_path = '/cluster/home03/infk/alucchi/src/workspace/sgd/'
data_path = '/Users/hadi/Documents/Education/SGD/variance_reduction/code/'
lib_path = data_path + 'lib/EJML-core-0.26.jar'

##########################

alg_type = ['sgd']
eta0s = [1e-3]
t0s = [-1]
lambdas = [1e-4]
svrg_pSamples = [-1]
npasses = [5]
nsamplesperpass = [500]
svrg_outer_pSamples = [-1]
hessian_psamples = [-1]

listOfLists = [alg_type, eta0s, t0s, lambdas, svrg_pSamples, npasses, nsamplesperpass, svrg_outer_pSamples, hessian_psamples]
#list_cs = list_cs + list(itertools.product(*listOfLists))

list_cs = list(itertools.product(*listOfLists))


##########################

alg_type = ['saga']
# eta0s = [1e-1]
# t0s = [-1, 10]
# lambdas = [0]
# svrg_pSamples = [-1]
# npasses = [50]
# nsamplesperpass = [10]
# svrg_outer_pSamples = [1e-3, 5e-3]
# hessian_psamples = [5e-4, 1e-4]
#eta0s = [1e-2, 5e-3]
eta0s = [1e-2, 1e-3]
t0s = [-1]
lambdas = [1e-4]
svrg_pSamples = [-1]
npasses = [200]
nsamplesperpass = [500]
svrg_outer_pSamples = [-1]
hessian_psamples = [-1]

listOfLists = [alg_type, eta0s, t0s, lambdas, svrg_pSamples, npasses, nsamplesperpass, svrg_outer_pSamples, hessian_psamples]
#list_cs = list_cs + list(itertools.product(*listOfLists))

#list_cs = list(itertools.product(*listOfLists))

##########################

maximumNumberOfThreads = 100
threadLimiter = threading.BoundedSemaphore(maximumNumberOfThreads)

queues = []
threads = []

print ""
print "Starting max of " + str(maximumNumberOfThreads) + " threads, please wait..."

working_dir = os.getcwd()

now = datetime.datetime.now()

for i in range(0,len(list_cs)):
    l = list_cs[i]
    newQueue = Queue.Queue()
    time.sleep(4)
    os.chdir(working_dir)
    t = QueueThread(l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8])
    t.start()
    queues.append(newQueue)
    threads.append(t)
                    
while True:
    if exec_cmd == True:
        time.sleep(2)
    for q in range(len(queues)):
        if queues[q].empty():
            continue

    if exec_cmd == True:
        #print "\x1b[2J"	# clear screen
        #print "\x1b[1;1f"	# clear screen
        print ""
        print input_file_fullpath
        print str(now)
        print ""
        print "Press Control+C to quit"
	
print "Terminating..."

for t in threads:
    t.join()
sys.exit(0)
