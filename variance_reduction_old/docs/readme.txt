Compiling Info
 javac -d build/ -cp lib/Jama-1.0.3.jar:lib/EJML-core-0.26.jar:lib/jcommon-1.0.23.jar:lib/jfreechart-1.0.19.jar:lib/java-lsh-0.6.jar src/data/*.java src/plot/*.java src/opt/utils.java src/lsh/*.java src/stat/*.java src/opt/loss/*.java src/opt/*.java src/opt/config/*.java src/opt/externalcodes/*.java src/opt/firstorder/*.java src/opt/firstorder/old/*.java src/backup/*.java src/*.java
 java -cp .:build/ name_of_java_main_file address2config_file

Runing on euler
 bsub -R "rusage[mem=20480]" -W 48:30 -o "filename" < *.sh

Git commands
 git add sth
 git commit -m "message"
 git push orgin master
 git pull 
 
Details of packages: 
opt: optimization package
+ opt.firstorder: all optimization methods
++++ opt.firstorder.EulerNewton.java: implementation of Euler-Newton method to test run eulernewton_test.java 
+ opt.loss: all loss functions 


