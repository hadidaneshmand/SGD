javac -d build/ -cp lib/EJML-core-0.26.jar:lib/jcommon-1.0.23.jar:lib/jfreechart-1.0.19.jar:lib/java-lsh-0.6.jar src/data/*.java src/plot/*.java src/opt/utils.java src/lsh/*.java src/stat/*.java src/opt/loss/*.java src/opt/*.java src/opt/config/*.java src/opt/firstorder/*.java src/opt/firstorder/old/*.java src/*.java
java -cp .:build/ AdaptSAGA_Real
bsub -R "rusage[mem=20480]" -W 48:30
git add sth
git commit -m "message"
git push orgin master
git pull 
