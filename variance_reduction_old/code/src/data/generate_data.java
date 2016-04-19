package data;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class generate_data {
	public static List<DataPoint> guassian_linear(double noise_var, int n, DataPoint b,int d,double conditionFactor){ 
		DensePoint y_t = new DensePoint(n); 
		List<DataPoint> data = new ArrayList<DataPoint>(); 
		Random r = new Random();
		Matrix X = new Matrix(n,d);
		double base = Math.pow(n, 1.0/(2*(1./conditionFactor)*(d-1)));
		for(int i = 0;i <n;i++){ 
			DataPoint Xi = new SparsePoint(); 
			for(int j=0;j<d;j++){ 
				double xij = r.nextGaussian()/(Math.pow(base, j)); 
				Xi.set(j, xij);
				X.set(i, j, xij);
			}
//			Xi = (DataPoint) Xi.normalize();
			for(int j=0;j<d;j++){ 
				X.set(i, j, Xi.get(j));
			}
			double yi = Xi.scalarProduct(b);
			y_t.set(i,yi+noise_var*r.nextGaussian()); 
			Xi.setLabel(yi+noise_var*r.nextGaussian());
			data.add(Xi); 
		}
		Matrix xxt = (X.transpose().times(X));
		xxt = xxt.times(1.0/n);
		System.out.println("L:"+xxt.getMaxEigenvalue());
		System.out.println("mu:"+xxt.getMinEigenvalue());
//		System.out.println("distances:"+b_n_star.squaredNormOfDifferenceTo(b));
		return data; 
	}
}
