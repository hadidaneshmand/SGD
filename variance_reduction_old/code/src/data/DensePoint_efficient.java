
package data;


import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import javax.management.RuntimeErrorException;

import org.ejml.simple.SimpleMatrix;

/**
 * Implementation of a dense point (using a Java Vector of Double values)
 * 
 * @author martin
 *
 */
public class DensePoint_efficient extends DataPoint {
	
	public double[] point;
	int n;
	
	public static DensePoint_efficient zero(int dim){ 
		DensePoint_efficient out = new DensePoint_efficient(dim); 
		for(int i=0;i<dim;i++){ 
			out.set(i, 0.0);
		}
		 
		return out;
	}
	public static DensePoint_efficient one(int dim){ 
		DensePoint_efficient out = new DensePoint_efficient(dim); 
		for(int i=0;i<dim;i++){ 
			out.set(i, 1.0);
		}
		 
		return out;
	}

	public DensePoint_efficient(int d) {
		point = new double[d];
		for(int i = 0; i < d; ++i) {
			point[i]=0;
		}
		this.n = d;
	}
	

	@Override
	public Point abs() {
		int n = point.length;
		Point output = new DensePoint_efficient(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, Math.abs(point[i]));
		}
		
		return output;
	}
	
	@Override
	public double get(int i) {
		return point[i];
	}

	@Override
	public void set(int i, double value) {
		point[i] =value;
	}

	@Override
	public double scalarProduct(Point b) {
		double result = 0;
		for (int i = 0; i < point.length; i++) {
			result += point[i] * b.get(i);
		}
		return result;
	}

	@Override
	public double scalarProductIgnoringMostFeatures(Point b, int takeEveryKthFeature) {
		double result = 0;
		for (int i = 0; i < point.length; i++) {
			if  (i % takeEveryKthFeature == 0)
				result += this.get(i) * b.get(i);
		}
		return result;
	}
	
	@Override
	public Point divide(Point p) {
		Point output = new DensePoint_efficient(n);
		for (int i = 0; i < n; i++) {
			output.set(i, point[i] / p.get(i));
		}
		return output;
	}
	
	@Override
	public Point multiply(double m) {
		
		Point output = new DensePoint_efficient(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, point[i]*m);
		}
		
		return output;
	}

	@Override
	public Point multiply(Point p) {
		Point output = new DensePoint_efficient(n);
		for (int i = 0; i < n; i++) {
			output.set(i, this.get(i) * p.get(i));
		}
		return output;
	}
	
	@Override
	public Point normalize() {
		Point output = new DensePoint_efficient(n);
		double _norm = getNorm();
		for (int i = 0; i < n; i++) {
			double pi = point[i];
			output.set(i, pi/_norm);
		}
		return output;
	}
	
	@Override
	public double getNorm() {
		double _norm = 0;
		for (int i = 0; i < n; i++) {
			double pi = point[i];
			_norm += pi*pi;
		}
				
		return Math.sqrt(_norm);
	}
	
	@Override
	public Point negSign() {
		Point output = new DensePoint_efficient(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, -Math.signum(point[i]));
		}
		
		return output;
	}
	
	@Override
	public Point add(Point s) {
		Point output = new DensePoint_efficient(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, point[i]+s.get(i));
		}
		
		return output;
	}
	
	@Override
	public Point replicate(int c) {
		Point output = new DensePoint_efficient(n*c);
		int idx = 0;
		for(int j = 0; j < c; ++j) {
			for(int i = 0; i < n; ++i) {
				output.set(idx, point[i]);
				++idx;
			}
		}
		return output;		
	}
	
	@Override
	public Point sign() {
		Point output = new DensePoint_efficient(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, Math.signum(point[i]));
		}
		
		return output;
	}
	
	@Override
	public Point sub(int start, int end) {
		int n = end - start;
		DataPoint output = new DensePoint_efficient(n);
		for(int i = start; i < end; ++i) {
			output.set(i-start, point[i]);
		}
		return output;
	}
	
	@Override
	public Point subtract(Point s) {
		Point output = new DensePoint_efficient(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, point[i]-s.get(i));
		}
		
		return output;
	}
	
	@Override
	public Point subtract(double m) {
		Point output = new DensePoint_efficient(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, point[i]-m);
		}
		
		return output;
	}
	
	private static final Random generator = new Random();
	private static final int maxValue = 1000; // maximum absolute value of each coordinate
	
	/**
	 * Generates a number of random points
	 * 
	 * @param numPoints
	 * @param dimension
	 * @return
	 */
	public static List<Point> randomPoints(int numPoints, int dimension) {
		List<Point> points = new Vector<Point>();
		for (int i = 0; i < numPoints; i++) {
			points.add( randomPoint(dimension) );
		}
		return points;
	}
	
	/**
	 * Generates a list of random, labeled points, half being from the positive
	 * orthant, the other half being from the negative one.
	 * 
	 * @param numPoints
	 * @param dimension
	 * @return
	 */
	public static List<Point> randomPointsTwoPolytope(int numPoints, int dimension) {
		List<Point> points = new Vector<Point>(numPoints);
		int half = numPoints/2+1;
		for (int i = 0; i < half; i++) {
			points.add(randomPointTwoPolytope(dimension, 1));
		}
		for (int i = half; i < numPoints; i++) {
			points.add(randomPointTwoPolytope(dimension, -1));
		}
		return points;
	}
	
	/**
	 * Generates a random point, uniformly at random from the grid [0..1000]^dimension
	 */
	public static Point randomPoint(int dimension) {
		double[] values = new double[dimension];
		for (int d = 0; d < dimension; d++)
			values[d] = (double) generator.nextInt(maxValue);
		DensePoint_efficient out = new DensePoint_efficient(dimension); 
		out.point = values;
		return out;
	}
	
	/**
	 * Generates a random labeled point, uniformly at random from the grid
	 * [0..1000]^dimension if the sign is one, and from the negative orthant
	 * otherwise.
	 * 
	 * @param dimension
	 * @param sign
	 * @return
	 */
	public static DataPoint randomPointTwoPolytope(int dimension, int sign) {
		double[] values = new double[dimension];
		double value;
		for (int d = 0; d < dimension; d++){
			value = (double) generator.nextInt(maxValue);
			values[d] = sign * value;
		}
		DensePoint_efficient out = new DensePoint_efficient(dimension); 
		out.point = values;
		out.setLabel(sign);
		return out;
	}

	
	/**
	 * Outputs a list of dense points, the first line containing the dimension
	 * (number of features) of each point.
	 * 
	 * @param points
	 * @return
	 */
	public static String pointsToString(List<Point> points) {
		String s = "";
		s += (points.isEmpty() ? "none" : ((DensePoint_efficient)points.get(0)).dimension()) + "\n";
		for (Point point : points) {
			String ps = point.toString();
			s += ps.substring(1, ps.length()-1).replace(",", " ") + "\n";
		}
		return s;
	}

	/**
	 * The dimension (number of features) of this point.
	 * @return
	 */
	public int dimension() {
		return n;
	}
	
	@Override
	public String toString() {
		String point_str = ""; 
		for(int i=0;i<n;i++){ 
			if(i<n-1){ 
				point_str +=point[i] + ",";
			}
			else{
				point_str+= point[i];
			}
		}
		return "DensePoint (label "+this.getLabel()+"): "+point_str;
	}

	
	
	@Override
	public Point sqrt() {
		Point output = new DensePoint_efficient(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, Math.sqrt(point[i]));
		}
		
		return output;
	}
	
	@Override
	public Point threshold() {
		Point output = new DensePoint_efficient(n);
		
		for(int i = 0; i < n; ++i) {
			double value = point[i];
			output.set(i, (value>0)?value:0);
		}
		
		return output;
	}
	
	@Override
	public Matrix crossProduct(Point b, int featureDim) {
		DensePoint_efficient d = (DensePoint_efficient) b;
		Matrix result = new Matrix(featureDim, featureDim);
		for(int i = 0; i < n; ++i) {
			for(int j = 0; j < n; ++j) {
				result.set(i, j, this.get(i) * d.get(j));
			}
		}
		return result;
	}
	
	@Override
	public SparseMatrix crossProduct_sparse(Point b, int featureDim) {
		DensePoint_efficient d = (DensePoint_efficient) b;
		SparseMatrix result = new SparseMatrix(featureDim);
		for(int i = 0; i < n; ++i) {
			for(int j = 0; j < n; ++j) {
				result.put(i, j, this.get(i) * d.get(j));
			}
		}
		return result;
	}
	
	@Override
	public Double[] toArray() {
		throw new RuntimeErrorException(null, "Efficient dataPoint doesn't have to array function"); 
	}
	
	@Override
	public void writeToFile(String filename) {
		throw new RuntimeErrorException(null, "Can not write to a file with efficient datapoint"); 
	}
		
	
	@Override
	public Iterator<Double> iterator() {
		throw new RuntimeErrorException(null, "Efficient dataPoint doesn't have an iterator"); 
	}

	@Override
	public int getDimension() {
		return point.length;
	}

	@Override
	public DataPoint times(SimpleMatrix p) {
		int d = getDimension();
		DataPoint out = new DensePoint_efficient(d);
		if(d!=p.numCols()){ 
			throw new RuntimeException("dimensions miss-match!!!");
		}
		for(int i=0;i<getDimension();i++){ 
			double s = 0; 
			for(int j=0;j<p.numCols();j++){
				s+= p.get(i, j)*point[j]; 
			}
			out.set(i, s);
		}
		return out;
	}
	@Override
	public DataPoint clone_data() {
		int d = getDimension(); 
		DensePoint_efficient out = new DensePoint_efficient(d); 
		out.n = this.n; 
		out.setLabel(getLabel());
		out.point = new double[d]; 
		for(int i = 0 ;i<d;i++){
			out.point[i] = point[i];  
		}
		return out;
	}
	
}
