package data;

/**
 * Abstract class for everything we need to do with a geometric point.
 * Every point should be iterable on its index set of non-zero features.
 * 
 * @author martin
 *
 */
public abstract class Point implements Iterable<Double> {
	
	public abstract Point abs();
	
	public abstract Point add(Point s);
	
	public double angle(Point p) {		
		double dp = this.scalarProduct(p);
		double sq_norm = (Math.sqrt(this.squaredNorm()) * Math.sqrt(p.squaredNorm()));
		if(sq_norm < 1e-30) {
			return 0;
		} else {
			double cos = dp/sq_norm;
			cos = Math.min(cos, 1.0);
			cos = Math.max(cos, -1.0);
			return Math.acos(cos);	
		}
	}	
	
	public abstract Matrix crossProduct(Point b, int featureDim);
	
	public abstract SparseMatrix crossProduct_sparse(Point b, int featureDim);
	
	/**
	 * Get the i-th coordinate (or feature) of this point
	 * @param i
	 * @return
	 */
	public abstract double get(int i);
	
	public abstract Point divide(Point p);
	
	public abstract Point multiply(double d);
	
	public abstract Point multiply(Point p);
	
	/**
	 * Get negative sign
	 * @return
	 */
	public abstract Point negSign();
	
	public abstract Point normalize();
	
	public abstract double getNorm();
	
	/**
	 * Replicate point
	 * @param c number of copies
	 * @return
	 */	
	public abstract Point replicate(int c);
	
	public abstract Point sub(int start, int end);
	
	/**
	 * Set the i-th coordinate (or feature) of this point
	 * @param i
	 * @return
	 */
	public abstract void set(int i, double value);
	
	/**
	 * Scalar product in the original space where the points live
	 * @param b
	 * @return
	 */
	public abstract double scalarProduct(Point b);
	/**
	 * Scalar product that ignores most features but only takes every
	 * takeEveryKthFeature into account (random feature selection)
	 * @param b
	 * @param takeEveryKthFeature
	 * @return
	 */
	public abstract double scalarProductIgnoringMostFeatures(Point b, int takeEveryKthFeature);

	/**
	 * Scalar product in the original space where the points live
	 */
	public static double scalarProduct(Point a, Point b) {
		return a.scalarProduct(b);
	}
	
	public abstract Point sign();
	
	/**
	 * Squared norm in the original space where the point lives
	 * @return
	 */
	public double squaredNorm() {
		return scalarProduct(this,this);
	}
	
	/**
	 * This gives you the squared length of the difference vector between two points
	 * @param b
	 * @return
	 */
	public double squaredNormOfDifferenceTo(Point b) {
		return this.squaredNorm() + b.squaredNorm() - 2 * scalarProduct(this,b);
	}
	
	/**
	 * This gives you the squared length of the difference vector between two points
	 * @param b
	 * @return
	 */
	public static double squaredNormOfDifference(Point a, Point b) {
		return a.squaredNormOfDifferenceTo(b);
	}

	public abstract Point sqrt();
	
	public abstract Point subtract(Point s);

	public abstract Point subtract(double d);
	
	public abstract Point threshold();
	
	public abstract Double[] toArray();
	
	public abstract void writeToFile(String filename);
	
}
