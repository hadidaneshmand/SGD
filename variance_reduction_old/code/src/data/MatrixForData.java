package data;

import org.ejml.simple.SimpleMatrix;

public class MatrixForData extends SimpleMatrix {
	public SimpleMatrix mult(DataPoint p) {
		SimpleMatrix out = new SimpleMatrix(numRows(), numCols());
		
		if(numCols()!=p.getDimension())
		for(int i =0;i< this.numRows();i++){
			for(int j=0;j< numCols();j++){ 
				
			}
		}
		return out;
	}
}
