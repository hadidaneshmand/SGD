package data;

/*************************************************************************
 *  Compilation:  javac SparseMatrix.java
 *  Execution:    java SparseMatrix
 *  
 *  A sparse, square matrix, implementing using two arrays of sparse
 *  vectors, one representation for the rows and one for the columns.
 *
 *  For matrix-matrix product, we might also want to store the
 *  column representation.
 *
 *************************************************************************/

public class SparseMatrix {
    private final int N;           // N-by-N matrix
    private SparseVector[] rows;   // the rows, each row is a sparse vector

    // initialize an N-by-N matrix of all 0s
    public SparseMatrix(int N) {
        this.N  = N;
        rows = new SparseVector[N];
        for (int i = 0; i < N; i++) rows[i] = new SparseVector(N);
    }

    // create and return the N-by-N identity matrix
    public static SparseMatrix identity(int N) {
    	SparseMatrix I = new SparseMatrix(N);
        for (int i = 0; i < N; i++)
        	I.rows[i].put(i, 1);
        return I;
    }
    
    // set all the entries in the matrix to the specified value
    // Warning: this will create a fully dense matrix!
    public void init(double v) {
    	if(v != 0) {
	        for (int i = 0; i < N; i++) {
	            for (int j = 0; j < N; j++) {
	            	rows[i].put(j, v);
	            }
	        }
    	}
    }
    
    // put A[i][j] = value
    public void put(int i, int j, double value) {
        if (i < 0 || i >= N) throw new RuntimeException("Illegal index");
        if (j < 0 || j >= N) throw new RuntimeException("Illegal index");
        rows[i].put(j, value);
    }

    // return A[i][j]
    public double get(int i, int j) {
        if (i < 0 || i >= N) throw new RuntimeException("Illegal index");
        if (j < 0 || j >= N) throw new RuntimeException("Illegal index");
        return rows[i].get(j);
    }

    // return the number of nonzero entries (not the most efficient implementation)
    public int nnz() { 
        int sum = 0;
        for (int i = 0; i < N; i++)
            sum += rows[i].nnz();
        return sum;
    }

    // return the matrix-vector product b = Ax
    public SparseVector times(SparseVector x) {
        SparseMatrix A = this;
        if (N != x.size()) throw new RuntimeException("Dimensions disagree");
        SparseVector b = new SparseVector(N);
        for (int i = 0; i < N; i++)
            b.put(i, A.rows[i].dot(x));
        return b;
    }
    
    // return C = A * B
    public SparseMatrix times(SparseMatrix B) {
    	SparseMatrix A = this;
        if (A.N != B.N) throw new RuntimeException("Illegal matrix dimensions.");
        SparseMatrix C = new SparseMatrix(B.N);
        for (int i = 0; i < C.N; i++)
            for (int j = 0; j < C.N; j++)
                for (int k = 0; k < A.N; k++) {
                    double value = C.rows[i].get(j) + (A.rows[i].get(k) * B.rows[k].get(j));
                    C.rows[i].put(j, value);
                }
        return C;
    }

    // return C = A - B
    public SparseMatrix minus(SparseMatrix B) {
        SparseMatrix A = this;
        if (A.N != B.N) throw new RuntimeException("Dimensions disagree");
        SparseMatrix C = new SparseMatrix(N);
        for (int i = 0; i < N; i++)
            C.rows[i] = A.rows[i].minus(B.rows[i]);
        return C;
    }
    
    // return C = A + B
    public SparseMatrix plus(SparseMatrix B) {
        SparseMatrix A = this;
        if (A.N != B.N) throw new RuntimeException("Dimensions disagree");
        SparseMatrix C = new SparseMatrix(N);
        for (int i = 0; i < N; i++)
            C.rows[i] = A.rows[i].plus(B.rows[i]);
        return C;
    }

    // return Frobenius norm
    public double frobeniusNorm() {
    	double sq_norm = 0;
        for (int i = 0; i < this.N; i++) {
            for (int j = 0; j < this.N; j++) {
            	double value = rows[i].get(j);
            	sq_norm += value*value;
            }
        }
        return Math.sqrt(sq_norm);
    }    
    
    public void set(SparseMatrix A, int offset_m, int offset_n) {
        for (int i = 0; i < A.N; i++) {
            for (int j = 0; j < A.N; j++) {
                this.rows[offset_m+i].put(offset_n+j, A.rows[i].get(j));
            }
        }
    }
    
    // return a string representation
    public String toString() {
        String s = "N = " + N + ", nonzeros = " + nnz() + "\n";
        for (int i = 0; i < N; i++) {
            s += i + ": " + rows[i] + "\n";
        }
        return s;
    }

    // return C = A * B
    public SparseMatrix times(double d) {
        SparseMatrix A = this;
        SparseMatrix C = new SparseMatrix(A.N);
        for (int i = 0; i < C.N; i++)
            for (int j = 0; j < C.N; j++) {
            	double value = C.rows[i].get(j) + (A.rows[i].get(j) * d);
            	C.rows[i].put(j, value);
            }
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
        SparseMatrix A = this;
        SparsePoint output = new SparsePoint();
        for (int i = 0; i < A.N; i++)  { // go over columns first
        	double dp = 0;
        	for (int j : s.featureSet()) { // go over rows next
        		dp += A.rows[i].get(j)*s.get(j);
        	}
        	output.set(i, dp);
        }
        return output;
    }
    
    // return A*p
    public DensePoint times(DensePoint p) {
        SparseMatrix A = this;
        int n = p.dimension();
        DensePoint output = new DensePoint();
        for (int i = 0; i < A.N; i++)  { // go over columns first
        	int dp = 0;
        	for (int j = 0; j < n; ++j) { // go over rows next
        		dp += A.rows[i].get(j)*output.get(j);
        	}
        	output.set(i, dp);
        }
        return output;
    }
    
    // create and return the transpose of the invoking matrix
    public SparseMatrix transpose() {
    	SparseMatrix A = new SparseMatrix(N);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            	A.rows[i].put(j, this.rows[j].get(i));
        return A;
    }
    
    // print matrix to standard output
    public void show() {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                System.out.printf("%9.4f ", this.rows[j].get(i));
            System.out.println();
        }
    }
    
    // test client
    public static void main(String[] args) {
        SparseMatrix A = new SparseMatrix(5);
        SparseVector x = new SparseVector(5);
        A.put(0, 0, 1.0);
        A.put(1, 1, 1.0);
        A.put(2, 2, 1.0);
        A.put(3, 3, 1.0);
        A.put(4, 4, 1.0);
        A.put(2, 4, 0.3);
        x.put(0, 0.75);
        x.put(2, 0.11);
        System.out.println("x     : " + x);
        System.out.println("A     : " + A);
        System.out.println("Ax    : " + A.times(x));
        System.out.println("A + A : " + A.plus(A));
    }

}
