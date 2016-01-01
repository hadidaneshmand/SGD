
package data;


import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Vector;

/**
 * Implementation of a dense point (using a Java Vector of Double values)
 * 
 * @author martin
 *
 */
public class DensePoint extends DataPoint {
	
	public List<Double> point;

	
	public static DensePoint zero(int dim){ 
		DensePoint out = new DensePoint(dim); 
		for(int i=0;i<dim;i++){ 
			out.set(i, 0.0);
		}
		return out;
	}
	public DensePoint() {
		point = new Vector<Double>();
	}

	public DensePoint(int n) {
		point = new Vector<Double>();
		for(int i = 0; i < n; ++i) {
			point.add(0.0);
		}
	}
	
	public DensePoint(Double... coords) {
		point = Arrays.asList(coords);
	}

	@Override
	public Point abs() {
		int n = point.size();
		Point output = new DensePoint(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, Math.abs(point.get(i)));
		}
		
		return output;
	}
	
	@Override
	public double get(int i) {
		return point.get(i);
	}

	@Override
	public void set(int i, double value) {
		point.set(i, value);
	}

	@Override
	public double scalarProduct(Point b) {
		double result = 0;
		for (int i = 0; i < point.size(); i++) {
			result += this.get(i) * b.get(i);
		}
		return result;
	}

	@Override
	public double scalarProductIgnoringMostFeatures(Point b, int takeEveryKthFeature) {
		double result = 0;
		for (int i = 0; i < point.size(); i++) {
			if  (i % takeEveryKthFeature == 0)
				result += this.get(i) * b.get(i);
		}
		return result;
	}
	
	@Override
	public Point divide(Point p) {
		int n = point.size();
		Point output = new DensePoint(n);
		for (int i = 0; i < point.size(); i++) {
			output.set(i, this.get(i) / p.get(i));
		}
		return output;
	}
	
	@Override
	public Point multiply(double m) {
		int n = point.size();
		Point output = new DensePoint(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, point.get(i)*m);
		}
		
		return output;
	}

	@Override
	public Point multiply(Point p) {
		int n = point.size();
		Point output = new DensePoint(n);
		for (int i = 0; i < point.size(); i++) {
			output.set(i, this.get(i) * p.get(i));
		}
		return output;
	}
	
	@Override
	public Point normalize() {
		Point output = new DensePoint(point.size());
		double _norm = getNorm();
		for (int i = 0; i < point.size(); i++) {
			double pi = point.get(i);
			output.set(i, pi/_norm);
		}
		return output;
	}
	
	@Override
	public double getNorm() {
		double _norm = 0;
		for (int i = 0; i < point.size(); i++) {
			double pi = point.get(i);
			_norm += pi*pi;
		}
				
		return Math.sqrt(_norm);
	}
	
	@Override
	public Point negSign() {
		int n = point.size();
		Point output = new DensePoint(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, -Math.signum(point.get(i)));
		}
		
		return output;
	}
	
	@Override
	public Point add(Point s) {
		int n = point.size();
		Point output = new DensePoint(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, point.get(i)+s.get(i));
		}
		
		return output;
	}
	
	@Override
	public Point replicate(int c) {
		int n = point.size();
		Point output = new DensePoint(n*c);
		int idx = 0;
		for(int j = 0; j < c; ++j) {
			for(int i = 0; i < n; ++i) {
				output.set(idx, point.get(i));
				++idx;
			}
		}
		return output;		
	}
	
	@Override
	public Point sign() {
		int n = point.size();
		Point output = new DensePoint(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, Math.signum(point.get(i)));
		}
		
		return output;
	}
	
	@Override
	public Point sub(int start, int end) {
		int n = end - start;
		DataPoint output = new DensePoint(n);
		for(int i = start; i < end; ++i) {
			output.set(i-start, point.get(i));
		}
		return output;
	}
	
	@Override
	public Point subtract(Point s) {
		int n = point.size();
		Point output = new DensePoint(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, point.get(i)-s.get(i));
		}
		
		return output;
	}
	
	@Override
	public Point subtract(double m) {
		int n = point.size();
		Point output = new DensePoint(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, point.get(i)-m);
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
		Double[] values = new Double[dimension];
		for (int d = 0; d < dimension; d++)
			values[d] = (double) generator.nextInt(maxValue);
		return new DensePoint(values);
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
		Double[] values = new Double[dimension];
		double value;
		for (int d = 0; d < dimension; d++){
			value = (double) generator.nextInt(maxValue);
			values[d] = sign * value;
		}
		DensePoint p = new DensePoint(values);
		p.setLabel(sign);
		return p;
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
		s += (points.isEmpty() ? "none" : ((DensePoint)points.get(0)).dimension()) + "\n";
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
		return point.size();
	}
	
	@Override
	public String toString() {
		return "DensePoint (label "+this.getLabel()+"): "+point.toString();
	}

	@Override
	public Iterator<Double> iterator() {
		return point.iterator();
	}
	
	@Override
	public Point sqrt() {
		int n = point.size();
		Point output = new DensePoint(n);
		
		for(int i = 0; i < n; ++i) {
			output.set(i, Math.sqrt(point.get(i)));
		}
		
		return output;
	}
	
	@Override
	public Point threshold() {
		int n = point.size();
		Point output = new DensePoint(n);
		
		for(int i = 0; i < n; ++i) {
			double value = point.get(i);
			output.set(i, (value>0)?value:0);
		}
		
		return output;
	}
	
	@Override
	public Matrix crossProduct(Point b, int featureDim) {
		int n = point.size();
		DensePoint d = (DensePoint) b;
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
		int n = point.size();
		DensePoint d = (DensePoint) b;
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
		return (Double[]) point.toArray();
	}
	
	@Override
	public void writeToFile(String filename) {
		writeToFile(filename, false);
	}
		
	public void writeToFile(String filename, boolean append) {		
		String s = Integer.toString((int)getLabel());
		int idx = 1;
		for(Iterator<Double> iter = point.iterator(); iter.hasNext(); ) {
			s += " " + idx + ":" + iter.next();
			++idx;
		}
		try {
			PrintWriter out = new PrintWriter(new FileWriter(filename, append));
			out.println(s);
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
