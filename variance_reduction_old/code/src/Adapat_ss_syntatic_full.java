import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import opt.config.Config;
import opt.firstorder.SGD;
import opt.firstorder.old.SAGA_old;
import opt.loss.LeastSquares;
import plot.XYLinesChart;
import data.DataPoint;
import data.DensePoint;
import data.Matrix;
import data.Result;
import data.SparsePoint;


public class Adapat_ss_syntatic_full {
	public static void main(String[] args) {
		Config cf = new Config(); 
		cf.lossType = Config.LossType.REGRESSION; 
		cf.lambda = 0; 
		cf.nSamplesPerPass = 1;
		cf.initType = Config.InitType.ZERO;
		int MaxItr = 100000; 
		Random r = new Random(); 
		int d = 20; 
		cf.featureDim = d; 
		int n = 10000; 
		SparsePoint b = new SparsePoint(); 
		for(int i = 0;i < d;i++){ 
			b.set(i, r.nextDouble());
		}
		b =  (SparsePoint) b.normalize();
		double noise_var = 0.1; 
		DensePoint y_t = new DensePoint(n); 
		Matrix X = new Matrix(n,d);
		List<DataPoint> data = new ArrayList<DataPoint>(); 
		for(int i = 0;i <n;i++){ 
			DataPoint Xi = new SparsePoint(); 
			for(int j=0;j<d;j++){ 
				double xij = r.nextGaussian()/((j+1)); 
				Xi.set(j, xij);
				X.set(i, j, xij);
				
			}
			double yi = Xi.scalarProduct(b);
			y_t.set(i,yi+noise_var*r.nextGaussian()); 
			Xi.setLabel(yi+noise_var*r.nextGaussian());
			data.add(Xi); 
		}
		System.out.println("data generating accomblished");
		Matrix xxt = (X.transpose().times(X));
		xxt = xxt.times(1.0/n);
		DataPoint b_n_star = AdaptSAGA_SYN.regression(data,d);
//		System.out.println("distances:"+b_n_star.squaredNormOfDifferenceTo(b));
		double conditionNum = xxt.getConditionNumber();
		double mu = xxt.getMinEigenvalue();
		System.out.println("condition number of the matrix: "+ 1.0/conditionNum);
//		System.out.println("mu="+mu);
		double eta_n = 0.1/(mu*n+0.1);
		SAGA_old saga_n = new SAGA_old(data,cf,cf.lambda,eta_n); 
		LeastSquares loss = new LeastSquares(data, d);
		loss.setLambda(0);
		SGD sgd = new SGD(loss); 
//		sgd.setLearning_rate(Math.pow(2.0/d, 2));;
//		Loss_static loss = new LeastSquares(data, d);
//		loss.setLambda(0);
		List<Double> convergence= new ArrayList<Double>(); 
		List<Double> convergence_n = new ArrayList<Double>(); 
		List<Double> convergence_sgd = new ArrayList<Double>(); 
		List<Double> convergence_opt= new ArrayList<Double>(); 
		List<Double> convergence_n_opt = new ArrayList<Double>(); 
		List<Double> convergence_sgd_opt = new ArrayList<Double>(); 
		
		List<Double> itrs = new ArrayList<Double>(); 
		
		
		boolean ciritical_seen = false; 
		int cirictial_iteration = 0; 
		ArrayList<String> resnames = new ArrayList<String>(); 
		resnames.add("saga_adapt_empirical");
		resnames.add("saga_adapt_expected"); 
		Result res = new Result(resnames); 
		int repeat = 1; 
		for(int c=0;c<repeat;c++){ 
			convergence= new ArrayList<Double>(); 
			convergence_n = new ArrayList<Double>(); 
			convergence_sgd = new ArrayList<Double>(); 
			convergence_opt= new ArrayList<Double>(); 
			convergence_n_opt = new ArrayList<Double>(); 
			convergence_sgd_opt = new ArrayList<Double>(); 
			LinkedList<Integer> ms = new LinkedList<Integer>(); 
			int nextm = (int) Math.floor(n-1); 
			ms.push(nextm); 
			nextm = (int) (nextm-1.0);
			while(nextm>400){
				ms.push(nextm); 
				nextm = (int) (nextm-1.0); 

			}
			int m = ms.pollFirst();
			double eta_m = 0.1;
			int s = 2*m;
			System.out.println("s="+s);
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
			saga_n = new SAGA_old(data, cf, cf.lambda, eta_n);
			for(int i = 0;i<MaxItr;i++){
//				System.out.println("m="+m+",s="+s);
				itrs.add((double) ((i+1)*cf.nSamplesPerPass)); 
				saga_m.OneIteration(cf.nSamplesPerPass);
				saga_n.OneIteration(cf.nSamplesPerPass);
				double epsilon = 1e-20;
//				if(i % recording_step == 1){
//					
//				}
				convergence.add(Math.log(epsilon+saga_m.getParam().squaredNormOfDifferenceTo(b_n_star))/Math.log(2.0));
				convergence_n.add(	Math.log(epsilon+saga_n.getParam().squaredNormOfDifferenceTo(b_n_star))/Math.log(2.0));
				convergence_sgd.add(Math.log(epsilon + sgd.getParam().squaredNormOfDifferenceTo(b_n_star))/Math.log(2.0));
				convergence_opt.add(Math.log(epsilon+saga_m.getParam().squaredNormOfDifferenceTo(b))/Math.log(2.0));
				convergence_n_opt.add(Math.log(epsilon+ saga_n.getParam().squaredNormOfDifferenceTo(b))/Math.log(2.0));
				convergence_sgd_opt.add(Math.log(epsilon+sgd.getParam().squaredNormOfDifferenceTo(b))/Math.log(2.0)); 
				if(!ciritical_seen && (i+1)*cf.nSamplesPerPass>s && s >0){ 
//					System.out.println("v="+(n/2.0));
					int past_m =m; 
					m = ms.pollFirst(); 
					eta_m = 0.1/(mu*m+0.1);
//					s += (int) Math.floor(-1*Math.log(6)/Math.log(1-1.0/(m)));
					s += 2	; 
					if(s <0){ 
						m = n-1; 
						eta_m = 1.0/n; 
					}
					if(!ciritical_seen && m==n-1){ 
						cirictial_iteration = (i*cf.nSamplesPerPass);
						ciritical_seen = true; 
					}
//					Collections.shuffle(indices, random);
//					subIndices.add(indices.get(m)); 
					for(int k=past_m;k<m;k++){ 
						subData.add(data.get(indices.get(k)));
					}
//					subData.add(data.get(indices.get(m))); 
					SAGA_old saga_new = new SAGA_old(subData,cf,cf.lambda,eta_m); 
					saga_new.setParam(saga_m.getParam());
					saga_m = saga_new; 
				}
			}
			res.addresult(resnames.get(0), convergence);
			res.addresult(resnames.get(1), convergence_opt);
		}
	    res.write2File("outs/SAGA_ADAPT_K400");
		System.out.println("Ciritical Iteration:"+cirictial_iteration);
		List<List<Double>> series = new ArrayList<List<Double>>();
		series.add(convergence);
		series.add(convergence_n); 
//		series.add(convergence_sgd);
		List<String> snames = new ArrayList<String>(); 
		snames.add("saga_adapt");
		snames.add("saga");
//		snames.add("SGD"); 
		XYLinesChart chart = new XYLinesChart(series,itrs,snames,"Convergence rate towards empirical optimal","Number of Iterations","Log_2(|w^s - w_n^*|^2)");
		
		int width = 800; /* Width of the image */
        int height = 600; /* Height of the image */ 
        chart.save("outs/empirical_convergence", width, height);
        chart.setVisible(true); 
        
        List<List<Double>> series2 = new ArrayList<List<Double>>();
		series2.add(convergence_opt);
		series2.add(convergence_n_opt); 
//		series2.add(convergence_sgd_opt);
		List<String> snames2 = new ArrayList<String>(); 
		snames2.add("saga_adapt");
		snames2.add("saga");
//		snames2.add("SGD"); 
		XYLinesChart chart2 = new XYLinesChart(series2,itrs,snames2,"Convergence rate towards the true optimal","Number of Iterations","Log_2(|w^s - w^*|^2)");
		
        chart2.save("outs/Optimal_convergence", width, height);
        chart2.setVisible(true);
        
		
	}
}
