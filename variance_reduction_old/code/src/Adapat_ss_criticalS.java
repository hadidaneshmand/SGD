import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import opt.config.Config;
import opt.firstorder.old.SAGA_old;
import plot.XYLinesChart;
import data.DataPoint;
import data.DensePoint;
import data.Matrix;
import data.SparsePoint;


public class Adapat_ss_criticalS {
	public static void main(String[] args) {
		Config cf = new Config(); 
		cf.lossType = Config.LossType.REGRESSION; 
		cf.lambda = 0; 
		cf.nSamplesPerPass = 20;
		cf.initType = Config.InitType.ZERO;
		int MaxItr = 2000; 
		Random r = new Random(); 
		int d = 50; 
		cf.featureDim = d; 
		int n = 30000; 
		

		DensePoint b = new DensePoint(d); 
		for(int i = 0;i < d;i++){ 
			b.set(i, r.nextDouble());
		}
		
		double noise_var = 1; 
		DensePoint y_t = new DensePoint(n); 
		Matrix X = new Matrix(n,d);
		List<DataPoint> data = new ArrayList<DataPoint>(); 
		for(int i = 0;i <n;i++){ 
			double yi = 0; 
			DataPoint Xi = new SparsePoint(); 
			for(int j=0;j<d;j++){ 
				double xij = r.nextGaussian(); 
				Xi.set(j, xij);
				X.set(i, j, xij);
				yi += xij*b.get(j);
			}
			y_t.set(i, yi); 
			Xi.setLabel(yi+noise_var*r.nextGaussian());
			data.add(Xi); 
		}
		Matrix xxt = (X.transpose().times(X));
		DataPoint b_n_star = AdaptSAGA_SYN.regression(data,d);
		System.out.println("distances:"+b_n_star.squaredNormOfDifferenceTo(b));
		double conditionNum = xxt.getConditionNumber();
		double mu = xxt.getMinEigenvalue();
		System.out.println("condition number of the matrix: "+ conditionNum);
		System.out.println("mu="+mu);
		double eta_n = 1.0/n;
		
		List<Double> switching_steps= new ArrayList<Double>(); 
		List<Double> ms = new ArrayList<Double>(); 
		
	    double repeat = 6; 
		
		for(int m = 1000;m<n;m=m+1000){
			System.out.println("m="+m);
			ms.add((double) m); 
			double step_sum = 0; 
			for(int re=0;re<repeat;re++){
				MaxItr = 10*m; 
				double eta_m = 1.0/m;
				ArrayList<Integer> subIndices = new ArrayList<Integer>(); 
				for(int i=0;i<m;i++){ 
					subIndices.add(r.nextInt(n)); 
				}
				List<DataPoint> subData = Adapt_ss.getsubsample(data, subIndices); 
				SAGA_old saga_m = new SAGA_old(subData,cf,cf.lambda,eta_m); 
				DataPoint b_m_star = AdaptSAGA_SYN.regression(subData, d);
				double adjust = 1; 
				double eta_n_k = Math.pow((1-eta_n*adjust),1); 
				double eta_m_k = Math.pow((1-eta_m*adjust), 1);
				double switching_ratio = Math.log10((1-eta_n_k)/(eta_n_k -eta_m_k ));
				System.out.println("switching ratio:" +switching_ratio);
				for(int i=0;i<MaxItr;i++){ 
					saga_m.OneIteration(cf.nSamplesPerPass);
					double epsilon = 1e-20;
					double ratiot = Math.log10(epsilon+saga_m.getParam().squaredNormOfDifferenceTo(b_m_star)/b_m_star.squaredNormOfDifferenceTo(b_n_star));
					if( ratiot<switching_ratio){ 
						step_sum+=((double) i*cf.nSamplesPerPass);
						break;
					}
				}
			}
			switching_steps.add(step_sum/repeat);
		}
		
		List<List<Double>> series = new ArrayList<List<Double>>();
		series.add(switching_steps);
		List<String> snames = new ArrayList<String>(); 
		snames.add("S");
		XYLinesChart chart = new XYLinesChart(series,ms,snames,"Convergence rate","Number of Iterations","S");
		
		int width = 800; /* Width of the image */
        int height = 600; /* Height of the image */ 
        File chartfile = new File("outs/s_syntactic.JPEG"); 
        chart.save(chartfile, width, height);
        chart.setVisible(true);
        
		
	}
}
