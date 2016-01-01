
package data;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Implementation of a sparse point (using a Java HashMap to Double values)
 * 
 * @author martin
 *
 */
public class SparsePoint extends DataPoint {
	
	private Map<Integer,Double> point;

	public SparsePoint() {
		point = new HashMap<Integer, Double>();
	}
	
	public SparsePoint(Map<Integer,Double> point) {
		this.point = point;
	}

	@Override
	public Point abs() {
		Point output = new SparsePoint();
		for (int i : point.keySet()) {
			output.set(i, Math.abs(point.get(i)));
		}
				
		return output;
	}
	
	@Override
	public Point add(Point p) {
		Point output = null;	
		if(p instanceof SparsePoint) {
			output = new SparsePoint();
			for (int i : point.keySet()) {
				output.set(i, point.get(i));
			}	
			SparsePoint s = (SparsePoint) p;
			for (int i : s.point.keySet()) {
				double value = 0;
				if(point.containsKey(i)) {
					value = point.get(i);
				}
				output.set(i, value+s.get(i));
			}
		} else {
			DensePoint d = (DensePoint) p;
			int n = d.dimension();
			output = new DensePoint(n);
			for(int i = 0; i < n; i++) {
				output.set(i, p.get(i));
			}
			for (int i : point.keySet()) {
				output.set(i, point.get(i)+output.get(i));
			}	
			
		}
		
		return output;
	}
	
	@Override
	public Matrix crossProduct(Point p, int n) {
		Matrix result = new Matrix(n, n);
		SparsePoint s = (SparsePoint) p;
		for (int i : point.keySet()) {
			for (int j : s.point.keySet()) {
				result.set(i, j, this.get(i) * s.get(j));
			}
		}
		return result;
	}
	
	@Override
	public SparseMatrix crossProduct_sparse(Point p, int n) {
		SparseMatrix result = new SparseMatrix(n);
		SparsePoint s = (SparsePoint) p;
		for (int i : point.keySet()) {
			for (int j : s.point.keySet()) {
				result.put(i, j, this.get(i) * s.get(j));
			}
		}
		return result;
	}
	
	@Override
	public Point divide(Point p) {
		Point output = new SparsePoint();
		for (int i : point.keySet()) {
			output.set(i, this.get(i) / p.get(i));
		}
		return output;
	}
	
	@Override
	public double get(int i) {
		if (point.containsKey(i))
			return point.get(i);
		else
			return 0;
	}

	@Override
	public Point multiply(double m) {
		Point output = new SparsePoint();
		for (int i : point.keySet()) {
			output.set(i, point.get(i)*m);
		}
				
		return output;
	}
	
	@Override
	public Point multiply(Point p) {
		Point output = new SparsePoint();
		for (int i : point.keySet()) {
			output.set(i, this.get(i) * p.get(i));
		}
		return output;
	}
	
	@Override
	public Point negSign() {
		Point output = new SparsePoint();
		for (int i : point.keySet()) {
			output.set(i, -Math.signum(point.get(i)));
		}
				
		return output;
	}

	@Override
	public Point normalize() {
		Point output = new SparsePoint();
		double _norm = getNorm();
		for (int i : point.keySet()) {
			double pi = point.get(i);
			output.set(i, pi/_norm);
		}
		return output;
	}
	
	@Override
	public double getNorm() {
		double _norm = 0;
		for (int i : point.keySet()) {
			double pi = point.get(i);
			_norm += pi*pi;
		}
				
		return Math.sqrt(_norm);
	}
	
	@Override
	public Point subtract(double m) {
		Point output = new SparsePoint();
		for (int i : point.keySet()) {
			output.set(i, point.get(i)-m);
		}
				
		return output;
	}
	
	@Override
	public Point subtract(Point p) {
		return add(p.multiply(-1.0));
	}	
	
	public void set(Point p) {
		if(p instanceof SparsePoint) {
			SparsePoint s = (SparsePoint) p;
			for (int i : s.point.keySet()) {
				point.put(i, s.get(i));
			}
		} else {
			DensePoint d = (DensePoint) p;
			int n = d.dimension();
			for(int i = 0; i < n; ++i) {
				point.put(i, p.get(i));
			}
		}
	}
	
	@Override
	public void set(int i, double value) {
		point.put(i, value);
	}

	@Override
	public double scalarProduct(Point b) {
		double result = 0;
		for (int i : point.keySet()) {
			result += this.get(i) * b.get(i);
		}
		return result;
	}

	@Override
	public double scalarProductIgnoringMostFeatures(Point b, int takeEveryKthFeature) {
		double result = 0;
		for (int i : point.keySet()) {
			if  (i % takeEveryKthFeature == 0)
				result += this.get(i) * b.get(i);
		}
		return result;
	}

	public Iterator<Double> iterator() {
		return point.values().iterator();
	}
	
	public Set<Integer> featureSet() {
		return point.keySet();
	}
	
	@Override
	public Point sqrt() {
		Point output = new SparsePoint();
		for (int i : point.keySet()) {
			output.set(i, Math.sqrt(point.get(i)));
		}
				
		return output;
	}
	
	@Override
	public String toString() {
		return "SparsePoint (label "+this.getLabel()+"): "+point.toString();
	}
	
	@Override
	public Point threshold() {
		Point output = new SparsePoint();
		for (int i : point.keySet()) {
			double value = point.get(i);
			output.set(i, (value>0)?value:0);
		}
				
		return output;
	}
	
	@Override
	public Double[] toArray() {
		Object[] o = point.values().toArray();
		int n = o.length;
		Double[] d = new Double[n];
		for(int i = 0; i < n; ++i) {
			d[i] = (Double)o[i];
		}
		return d;
	}

	@Override
	public void writeToFile(String filename) {
		boolean append = true;
		String s = Double.toString(getLabel());
		
		Map<Integer, Double> treeMap = new TreeMap<Integer, Double>(point);
		for (Integer key : treeMap.keySet()) {
			double value = point.get(key);
			s += " " + key + ":" + new DecimalFormat("#0.0000").format(value);
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

	@Override
	public Point replicate(int c) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Point sign() {
		Point output = new SparsePoint();
		for (int i : point.keySet()) {
			output.set(i, Math.signum(point.get(i)));
		}
				
		return output;
	}
	
	@Override
	public Point sub(int start, int end) {
		SparsePoint output = new SparsePoint();		
		for(int i = start; i < end; ++i) {
			if(point.containsKey(i)) {
				output.set(i-start, point.get(i));
			}
		}
		return output;
	}	
	
}
