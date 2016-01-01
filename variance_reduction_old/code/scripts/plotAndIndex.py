from optparse import OptionParser
import commands
import configIO
import os
import plotData
import re
import sys

pathname = os.path.dirname(sys.argv[0])  
#print 'full path =', os.path.abspath(pathname)

parser = OptionParser()
parser.add_option("-d", "--dir_name", dest="dir_name", default="training_files_", help="Directory name")
parser.add_option("-o", "--output_name", dest="output_name", default="output", help="Output directory name")

(options, args) = parser.parse_args()

ldirs = os.listdir('.')
ndirs = len(ldirs)

output_dir = os.environ['HOME'] + '/public_html/'
if not os.access(output_dir, os.R_OK):
    os.mkdir(output_dir)

f = open(output_dir + options.output_name + 'Index.html', "w")
f.write('<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en-us">\n')
f.write('<head>\n')
f.write('  <TITLE>Experiments - plots</TITLE>\n')
f.write('  <link rel="stylesheet" href="jquery/themes/blue/style.css" type="text/css" id="" media="print, projection, screen" />\n')
f.write('  <script type="text/javascript" src="jquery/jquery-latest.js"></script>\n')
f.write('  <script type="text/javascript" src="jquery/jquery.tablesorter.js"></script>\n')
f.write('  <script type="text/javascript" src="jquery/addons/pager/jquery.tablesorter.pager.js"></script>\n')
f.write('  <script type="text/javascript" src="data.json"></script>\n')
f.write('  <script type="text/javascript" src="myscript.js"></script>\n')
f.write('<script type="text/javascript" id="js">$(document).ready(function() {\n')
f.write('  // call the tablesorter plugin\n')
f.write('  $("table").tablesorter({\n')
f.write('  // sort on the first column and third column, order asc\n')
f.write('  sortList: [[0,0],[2,0]]\n')
f.write('  });\n')
f.write('}); </script>\n')
f.write('</head>\n')

f.write('<body onload="load()">\n')
f.write('Alg_type: ')
f.write('<input type="text" name="alg_type" id="alg_type" value="*" size=5>')
f.write(' Training_scores (x>?): ')
f.write('<input type="text" name="training_scores" id="training_scores" value="*" size=5>')
f.write(' Derivative (x>?): ')
f.write('<input type="text" name="derivative" id="derivative" value="*" size=5>')
f.write(' Eta0: ')
f.write('<input type="text" name="eta0" id="eta0" value="*" size=5>')
f.write(' T0: ')
f.write('<input type="text" name="T0" id="T0" value="*" size=5>')
f.write(' lambda: ')
f.write('<input type="text" name="lambda" id="lambda" value="*" size=5>')
f.write(' nsamplesperpass: ')
f.write('<input type="text" name="nsamplesperpass" id="nsamplesperpass" value="*" size=5>')
f.write(' svgr_pSamples: ')
f.write('<input type="text" name="svgr_pSamples" id="svgr_pSamples" value="*" size=5>')
f.write(' svgr_outer_pSamples: ')
f.write('<input type="text" name="svgr_outer_pSamples" id="svgr_outer_pSamples" value="*" size=5>')
f.write(' hessian_psamples: ')
f.write('<input type="text" name="hessian_psamples" id="hessian_psamples" value="*" size=5>')
f.write('<button onClick=\'deleteAllRows();return false;\'>Delete</button>')
f.write('<button onClick=\'reloadRows();return false;\'>Filter</button>')
f.write('<p>\n')
f.write('<table cellspacing="1" class="tablesorter" id="mytable">\n')
f.write('<thead>\n')
f.write('<tr>\n')
f.write('<th>Name</th>\n')
f.write('<th>Alg type</th>\n')
f.write('<th>Training score</th>\n')
f.write('<th>Test score</th>\n')
f.write('<th>Validation score</th>\n')
f.write('<th>Training loss</th>\n')
f.write('<th>Derivative</th>\n')
f.write('<th>Convergence point</th>\n')
f.write('<th>Nb iterations</th>\n')
f.write('<th>Eta0</th>\n')
f.write('<th>T0</th>\n')
f.write('<th>lambda</th>\n')
f.write('<th>nsamplesperpass</th>\n')
f.write('<th>svgr_pSamples</th>\n')
f.write('<th>svgr_outer_pSamples</th>\n')
f.write('<th>hessian_psamples</th>\n')
f.write('</tr>\n')
f.write('</thead>\n')
f.write('<tbody>\n')


fdata = open(output_dir + 'data.json', "w")
fdata.write('var mydata = [')

iteration_col_idx = 2
score_col_idx = 3

idx = 0
for _dir in ldirs:

    if not os.path.isdir(_dir):
        continue
    m = re.search(options.dir_name, _dir)
    if m is None:
        #print dir_name + ' not found in ' + _dir                                                                                                                                                                                        
        continue

    # Check if configuration file exists
    config_file_pattern = _dir[0:10]
    config_filename = ''
    for i in os.listdir(_dir):
        if i[0:len(config_file_pattern)] != config_file_pattern or i[len(i)-3:len(i)] != 'txt':
            continue
        config_filename = _dir + '/' + i

    _dir = _dir.replace('+', '.*')

    cmd = 'python ' + pathname + '/plotAll.py -e 1 -c 0 -d ' + _dir + ' -o ' + options.output_name + str(idx) + '/'
    print cmd
    os.system(cmd)

    derivative = plotData.computeDerivative(_dir, 'svgr_cl_loss', 1)
    if derivative == 0:
        derivative = plotData.computeDerivative(_dir, 'sgd_loss', 1)
    if derivative == 0:
        derivative = plotData.computeDerivative(_dir, 'saga_loss', 1)

    convergence_point = plotData.findConvergencePoint_dir(_dir, 'svgr_cl_loss', 1, 1)
    if convergence_point == 0:
        convergence_point = plotData.findConvergencePoint_dir(_dir, 'sgd_loss', 1, 1)
    if convergence_point == 0:
        convergence_point = plotData.findConvergencePoint_dir(_dir, 'saga_loss', 1, 1)
    

    # Training score

    best_training_score = '0'
    n_iterations = '0'

    cmd = pathname + '/sortScore.py -e 1 -s 0 -o 1 -d ' + _dir
    out = commands.getoutput(cmd)
    tokens = out.split('\t\t')
    if len(tokens) > 2:
        best_training_score = tokens[score_col_idx]
    if len(tokens) > 1:
        n_iterations = tokens[iteration_col_idx]

    cmd = pathname + '/sortScore.py -e 1 -s 0 -a svgr_cl -o 1 -d ' + _dir
    out = commands.getoutput(cmd)
    tokens = out.split('\t\t')
    if len(tokens) > 2:
        best_training_score = tokens[score_col_idx]
    if len(tokens) > 1:
        n_iterations = tokens[iteration_col_idx]

    cmd = pathname + '/sortScore.py -e 1 -s 0 -a saga -o 1 -d ' + _dir
    out = commands.getoutput(cmd)
    tokens = out.split('\t\t')
    if len(tokens) > 2:
        best_training_score = tokens[score_col_idx]
    if len(tokens) > 1:
        n_iterations = tokens[iteration_col_idx]

    # Test score

    cmd = pathname + '/sortScore.py -e 1 -s 2 -o 1 -d ' + _dir
    out = commands.getoutput(cmd)
    lines = out.split('\n')
    test_score = '0'
    if len(lines) > 2:
        tokens = lines[2].split('\t\t')
        if len(tokens) > 2:
            test_score = tokens[score_col_idx]

    # Validation score

    val_score = '0'
    n_val_iterations = '0'

    cmd = pathname + '/sortScore.py -e 1 -s 1 -o 1 -d ' + _dir
    print cmd
    out = commands.getoutput(cmd)
    tokens = out.split('\t\t')
    if len(tokens) > 2:
        val_score = tokens[score_col_idx]

    if len(tokens) > 1:
        n_val_iterations = tokens[iteration_col_idx]


    cmd = pathname + '/sortScore.py -e 1 -a svgr_cl -s 1 -o 1 -d ' + _dir
    print cmd
    out = commands.getoutput(cmd)
    tokens = out.split('\t\t')
    if len(tokens) > 2:
        val_score = tokens[score_col_idx]

    if len(tokens) > 1:
        n_val_iterations = tokens[iteration_col_idx]

    cmd = pathname + '/sortScore.py -e 1 -a saga -s 1 -o 1 -d ' + _dir
    print cmd
    out = commands.getoutput(cmd)
    tokens = out.split('\t\t')
    if len(tokens) > 2:
        val_score = tokens[score_col_idx]

    if len(tokens) > 1:
        n_val_iterations = tokens[iteration_col_idx]

    # Training loss

    best_training_loss = '0'
    n_iterations = '0'

    cmd = pathname + '/sortScore.py -e 1 -s 0 -o 1 -d ' + _dir
    out = commands.getoutput(cmd)
    tokens = out.split('\t\t')
    if len(tokens) > 2:
        best_training_loss = tokens[score_col_idx]

    cmd = pathname + '/sortScore.py -e 1 -s 0 -o 1 -a svgr_cl -d ' + _dir
    out = commands.getoutput(cmd)
    tokens = out.split('\t\t')
    if len(tokens) > 2:
        best_training_loss = tokens[score_col_idx]

    cmd = pathname + '/sortScore.py -e 1 -s 0 -o 1 -a saga -d ' + _dir
    out = commands.getoutput(cmd)
    tokens = out.split('\t\t')
    if len(tokens) > 2:
        best_training_loss = tokens[score_col_idx]

    # Read attributes from config file
    eta0 = configIO.read(config_filename,'eta0')
    T0 = configIO.read(config_filename,'T0')
    nsamplesperpass = configIO.read(config_filename,'nsamplesperpass')
    svgr_pSamples = configIO.read(config_filename,'svgr_pSamples')
    svgr_outer_pSamples = configIO.read(config_filename,'svgr_outer_pSamples')
    hessian_psamples = configIO.read(config_filename,'hessian_psamples')
    alg_type = configIO.read(config_filename,'alg_type')
    alambda = configIO.read(config_filename,'lambda')

    if idx != 0:
        fdata.write(',')

    fdata.write('{')
    fdata.write('"name" : "' + _dir + '", ')
    fdata.write('"link" : "' + options.output_name + str(idx) + '", ')
    fdata.write('"alg_type" : "' + alg_type + '", ')
    fdata.write('"train_score" : "' + best_training_score + '", ')
    fdata.write('"test_score" : "' + test_score + '", ')
    fdata.write('"val_score" : "' + val_score + '", ')
    fdata.write('"train_loss" : "' + best_training_loss + '", ')
    fdata.write('"derivative" : "' + str(derivative) + '", ')
    fdata.write('"convergence_point" : "' + str(convergence_point) + '", ')
    fdata.write('"n_iterations" : "' + n_iterations + '", ')
    fdata.write('"eta0" : "' + eta0 + '", ')
    fdata.write('"T0" : "' + T0 + '", ')
    fdata.write('"lambda" : "' + alambda + '", ')
    fdata.write('"nsamplesperpass" : "' + nsamplesperpass + '", ')
    fdata.write('"svgr_pSamples" : "' + svgr_pSamples + '", ')
    fdata.write('"svgr_outer_pSamples" : "' + svgr_outer_pSamples + '", ')
    fdata.write('"hessian_psamples" : "' + hessian_psamples + '"')
    fdata.write('}')

    idx += 1

f.write('</tbody>\n')
f.write('</table>')
f.write('</p>\n')
#f.write('</CENTER>\n')

f.write('</body>\n')
f.write('</html>\n')
f.close()

fdata.write('];')
fdata.close()


cmd = 'scp -r /root/public_html/* alucchi@web-login.inf.ethz.ch:/home/alucchi/public_html/plots/'
print cmd
os.system(cmd)
