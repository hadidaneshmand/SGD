import commands
import os

basedir = '../'

lfiles = os.listdir(basedir)

sgd_loss = 0
n = 0
for i in lfiles:
    root, ext = os.path.splitext(i)
    if root.startswith('sgd_loss') and ext == '.txt':
        st = commands.getstatusoutput('tail -n1 ' + basedir + i)
        sgd_loss = sgd_loss + float(st[1])
        n = n + 1

sgd_loss = sgd_loss/n

print sgd_loss

svgr_loss = 0
n = 0
for i in lfiles:
    root, ext = os.path.splitext(i)
    if root.startswith('svgr_loss') and ext == '.txt':
        st = commands.getstatusoutput('tail -n1 ' + basedir + i)
        svgr_loss = svgr_loss + float(st[1])
        n = n + 1

svgr_loss = svgr_loss/n

print svgr_loss

