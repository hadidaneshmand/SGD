package data;

import org.ejml.simple.SimpleMatrix;

/*************************************************************************
 *  Compilation:  javac Matrix.java
 *  Execution:    java Matrix
 *
 *  A bare-bones immutable data type for M-by-N matrices.
 *
 *************************************************************************/

public class Matrix {
	public int M = 0; // number of rows
	public int N = 0; // number of columns
	public SimpleMatrix m = null;

    // create M-by-N matrix of 0's
    public Matrix(int M, int N, double init_value) {
    	this.M = M;
    	this.N = N;
    	m = new SimpleMatrix(M, N);
    	init(init_value);
    }

    // create N-by-N matrix of 0's
    public Matrix(int N, double init_value) {
    	this.N = N;
    	m = new SimpleMatrix(N, N);
    	init(init_value);
    }
	
    // create M-by-N matrix 
    public Matrix(int M, int N) {
    	this.M = M;
    	this.N = N;
    	m = new SimpleMatrix(M, N);
    	init(0);
    }

    // create N-by-N matrix
    public Matrix(int N) {
    	this.N = N;
    	m = new SimpleMatrix(N, N);
    	init(0);
    }
    
    // create N-by-N matrix
    public Matrix(SimpleMatrix _m) {
    	m = _m;
    	M = _m.numRows();
    	N = _m.numCols();
    }
    
    // set all the entries in the matrix to the specified value
    // Warning: this will create a fully dense matrix!
    public void init(double v) {
    	m.set(v);
    }    
    
    // copy constructor
    private Matrix(Matrix A) { m = A.m; M = A.M; N = A.N; }

    // create and return the N-by-N identity matrix
    public static Matrix identity(int N) {
    	SimpleMatrix Im = SimpleMatrix.identity(N);
    	Matrix I = new Matrix(Im);
        return I;
    }
    
    public Matrix invert() {
    	return new Matrix(m.invert());
    }
    
    public Matrix transpose() {
    	SimpleMatrix Cm = m.transpose(); 
    	Matrix C = new Matrix(Cm);
        return C;
    }
    
    public double get(int i, int j) {
    	return m.get(i,j);
    }
    
    public void set(int i, int j, double value) {
    	m.set(i,j, value);
    }
    
    // return Frobenius norm
    public double frobeniusNorm() {
    	return m.normF();
    }

    // return C = this.m + B
    public Matrix plus(Matrix B) {
    	SimpleMatrix Cm = m.plus(B.m);
    	Matrix C = new Matrix(Cm);
        return C;
    }


    // return C = A - B
    public Matrix minus(Matrix B) {
    	SimpleMatrix Cm = m.minus(B.m);
    	Matrix C = new Matrix(Cm);
        return C;
    }

    // does A = B exactly?
    public boolean eq(Matrix B) {
    	return m.isIdentical(B.m, 1e-30);
    }
    
    public void set(Matrix A, int offset_m, int offset_n) {
        for (int i = 0; i < A.M; i++) {
            for (int j = 0; j < A.N; j++) {
            	m.set(offset_m+i, offset_n+j, A.m.get(i, j));
            }
        }
    }

    // return C = A * B
    public Matrix times(Matrix B) {
    	SimpleMatrix Cm = m.mult(B.m);
    	Matrix C = new Matrix(Cm);
        return C;
    }

    // return C = A * d
    public Matrix times(double d) {
    	SimpleMatrix Cm = m.scale(d);
    	Matrix C = new Matrix(Cm);
        return C;
    }
    
    // return A*p
    public DataPoint times(DataPoint p) {

    	if(p instanceof SparsePoint) {
    		SparsePoint s = (SparsePoint) p;
    		return this.times(s);
    	} else {
    		DensePoint d = (DensePoint) p;
    		return this.times(d);
    	}
    }
    
    // return A*p
    public SparsePoint times(SparsePoint s) {
        SimpleMatrix A = this.m;
        SparsePoint output = new SparsePoint();
        for (int i = 0; i < this.N; i++)  { // go over columns first
        	double dp = 0;
        	for (int j : s.featureSet()) { // go over rows next
        		dp += A.get(i,j)*s.get(j);
        	}
        	output.set(i, dp);
        }
        return output;
    }
    
    // return A*p
    public DensePoint times(DensePoint p) {
    	SimpleMatrix A = this.m;
        int n = p.dimension();
        DensePoint output = new DensePoint(this.N);
        System.out.println("this.N="+this.N);
        System.out.println("this.M="+this.M);
        System.out.println("n="+n);
        for (int i = 0; i < this.N; i++)  { // go over columns first
        	int dp = 0;
//        	System.out.println("i="+i);
        	for (int j = 0; j < n; j++) { // go over rows next
//        		System.out.println("p["+j+"]="+p.get(j));
        		dp += A.get(i,j)*p.get(j);
        	}
//        	System.out.println("d["+i+"]"+dp);
        	output.set(i, dp);
        }
        return output;
    }

    public double getMinEigenvalue() {
    	return m.eig().getEigenvalue(m.eig().getIndexMin()).getReal();
    }
    public double getMaxEigenvalue() {
    	return m.eig().getEigenvalue(m.eig().getIndexMax()).getReal();
    }
    public double getConditionNumber() {
    	return getMinEigenvalue()/getMaxEigenvalue();
    }
    
    // print matrix to standard output
    public void show() {
    	System.out.println(m.toString());
    }
}
