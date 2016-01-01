How to compile the code?
javac src/Config.java src/optimize.java src/utils.java src/data/DataPoint.java src/data/DensePoint.java src/data/IOTools.java src/data/Point.java src/data/SparsePoint.java src/data/SparseVector.java src/data/Matrix.java src/data/SparseMatrix.java src/data/ST.java -cp lib/*

How to run the code?
java -cp .:src/ optimize name_config_file

E.g.
java -cp .:src/ optimize config/config_heart.txt

See the files in the config directory for examples of valid configuration files.

Plot loss
python scripts/plotAll.py -d log_heart
