package backup;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import opt.Adapt_Strategy;
import opt.config.Config;
import opt.firstorder.SAGA_Adapt;
import opt.firstorder.old.SAGA_old;
import opt.loss.LeastSquares;
import opt.loss.Loss_static;
import data.DataPoint;
import data.DensePoint;
import data.Matrix;
import data.Result;
import data.SparsePoint;


public class AdaptSAGA_DependencyTest {
	public static void main(String[] args) {
		Config cf = new Config(); 
		cf.lossType = Config.LossType.REGRESSION; 
		cf.lambda = 0; 
		cf.nSamplesPerPass = 1;
		cf.initType = Config.InitType.ZERO;
		Random r = new Random(); 
		int d = 20; 
		cf.featureDim = d; 
		ArrayList<Double> ns = new  ArrayList<Double>(); 
		
		int rep = 10;
	    ArrayList<String> names = new ArrayList<String>();
	    names.add("saga"); 
	    names.add("saga_adapt"); 
	    names.add(""); 
	    names.add("Slop_One");
	    names.add("t");
	    Result result = new Result(names); 
	  
	    for(int k = 0;k<rep;k++){
	    	ArrayList<Double> errors = new ArrayList<Double>();
			for(int i=9;i<18;i++){
				System.out.println("i:"+i);
				int n = (int) Math.pow(2, i);
				if(k==0){
					ns.add((double) Math.log(n)); 
				}
				SparsePoint b = new SparsePoint(); 
				for(int j = 0;j < d;j++){ 
					b.set(j, r.nextDouble());
				}
				b =  (SparsePoint) b.normalize();
				double condfactor = 0.5;
				List<DataPoint> data = generateData(0.1,n, b, d,condfactor);
				  System.out.println("normdata:"+data.get(0).getNorm());
				DataPoint b_n_star = AdaptSAGA_SYN.regression(data,d);
//				DataPoint saga_b = saga_adapt(data,2*n, cf,1./Math.pow(n, 0.5),Math.pow(n, 0.5));
//				double error = Math.log(b_n_star.squaredNormOfDifferenceTo(saga_b));
				Loss_static loss = new LeastSquares(data, d);
				loss.setLambda(0);
				Adapt_Strategy as = new Adapt_Strategy(n, (int)Math.floor(Math.pow(n, condfactor)), false);
				SAGA_Adapt saga_adapt = new SAGA_Adapt(loss, as,1./Math.floor(Math.pow(n, condfactor)),1.5); 
				saga_adapt.Iterate(n);
				double error = Math.log(Math.abs(loss.getLoss(b_n_star)-loss.getLoss(saga_adapt.getParam()))); 
				errors.add(error);
			}
			result.addresult(names.get(1), errors);
			result.addresult(names.get(0), ns);
	    }
	    result.write2File("outs/slope_one_n");
//		ArrayList<String> series_names = new ArrayList<String>(); 
//		series_names.add("test"); 
//		  List<List<Double>> series = new ArrayList<List<Double>>();
//		  series.add(errors);
//		XYLinesChart chart = new XYLinesChart(series, ns, series_names, "test", "x", "y");
//		chart.setVisible(true); 
		
		
		
		
		
	}
	public static List<DataPoint> generateData(double noise_var, int n, DataPoint b,int d,double conditionFactor){ 
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
	public static DataPoint saga_adapt(List<DataPoint> data, int steps,Config cf,double mu,double conditionNum){ 
		int n = data.size();
		boolean ciritical_seen = false; 
		LinkedList<Integer> ms = new LinkedList<Integer>(); 
		int nextm = (int) Math.floor(n-1); 
		ms.push(nextm); 
		nextm = (int) (nextm-1.0);
		while(nextm>conditionNum){
			ms.push(nextm); 
			nextm = (int) (nextm-1.0); 

		}
		int m = ms.pollFirst();
		double eta_m = 0.1;
		int s = 2*m;
		List<Integer> indices = new ArrayList<Integer>(n);
		for(int i = 0; i < n; ++i) {
			indices.add(i, i);
		}
		Random random = new Random();
		Collections.shuffle(indices, random);
		ArrayList<Integer> subIndices = new ArrayList<Integer>(); 
		for(int j=0;j<m;j++){ 
			subIndices.add(indices.get(j)); 
		}
		List<DataPoint> subData = Adapt_ss.getsubsample(data, subIndices); 
		SAGA_old saga_m = new SAGA_old(subData,cf,cf.lambda,eta_m); 
		for(int i = 0;i<steps;i++){
//			System.out.println("m="+m+",s="+s);
			saga_m.OneIteration(1);
			if(!ciritical_seen && (i)>s && s >0){ 
//				System.out.println("v="+(n/2.0));
				int past_m =m; 
				m = ms.pollFirst(); 
				eta_m = 0.1/(mu*m+0.1);
//				s += (int) Math.floor(-1*Math.log(6)/Math.log(1-1.0/(m)));
				s += 2	; 
				if(s <0){ 
					m = n-1; 
					eta_m = 1.0/n; 
				}
				if(!ciritical_seen && m==n-1){ 
					ciritical_seen = true; 
				}
//				Collections.shuffle(indices, random);
//				subIndices.add(indices.get(m)); 
				for(int k=past_m;k<m;k++){ 
					subData.add(data.get(indices.get(k)));
				}
//				subData.add(data.get(indices.get(m))); 
				SAGA_old saga_new = new SAGA_old(subData,cf,cf.lambda,eta_m); 
				saga_new.setParam(saga_m.getParam());
				saga_m = saga_new; 
			}
		}
		return saga_m.getParam();
	}
}
