import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import opt.config.Config;
import opt.firstorder.SGD;
import opt.loss.LS_IM_AG;
import opt.loss.LeastSquares;
import opt.loss.Loss_static;
import plot.XYLinesChart;
import data.DataPoint;
import data.DensePoint;
import data.Matrix;
import data.SparsePoint;


public class ImportanceSampling {
	public static void main(String[] args) {
		
		Config cf = new Config(); 
		cf.lossType = Config.LossType.REGRESSION; 
		cf.lambda = 0; 
		cf.nSamplesPerPass = 1;
		cf.initType = Config.InitType.ZERO;
		int MaxItr = 1000; 
		Random r = new Random(); 
		int d = 5; 
		cf.featureDim = d; 
		int n = 1000; 
		SparsePoint b = new SparsePoint(); 
		for(int i = 0;i < d;i++){ 
			b.set(i, r.nextDouble());
		}
		b = (SparsePoint) b.normalize(); 
		
		double noise_var = 0.1; 
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
			}
			double norm = Xi.getNorm(); 
			for(int j=0;j<d;j++){
				double xij = 1.0*X.get(i, j)/norm;
				X.set(i, j, xij);
				yi += xij*b.get(j);
			}
			Xi = (DataPoint) Xi.normalize();
			y_t.set(i, yi+noise_var*r.nextGaussian()); 
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
		Loss_static loss = new LeastSquares(data, d);
		loss.setLambda(0);
		Loss_static loss_im = new LS_IM_AG(data, d);
		loss_im.setLambda(0);
		 
		
		List<Double> convergence_sgd = new ArrayList<Double>();
		List<Double> convergence_sgd_im = new ArrayList<Double>(); 
		List<Double> convergence_sgd_opt= new ArrayList<Double>(); 
		List<Double> convergence_im_opt = new ArrayList<Double>(); 
		List<Double> itrs = new ArrayList<Double>(); 
		for(int i=0;i<MaxItr;i++){ 
			convergence_sgd.add(0.0); 
			convergence_sgd_im.add(0.0); 
			convergence_sgd_opt.add(0.0); 
			convergence_im_opt.add(0.0); 
		}
		int rep = 12; 
		for(int k=0;k<rep;k++){
			SGD sgd = new SGD(loss);
			SGD sgd_importance = new SGD(loss_im);
			for(int i = 0;i<MaxItr;i++){
				System.out.println("iteration:"+i);
				itrs.add((double) ((i+1)*cf.nSamplesPerPass)); 
				sgd.Iterate(cf.nSamplesPerPass);
				sgd_importance.Iterate(cf.nSamplesPerPass);
				double epsilon = 1e-20;
				convergence_sgd.set(i,convergence_sgd.get(i)+Math.log(epsilon+sgd.getParam().squaredNormOfDifferenceTo(b_n_star))/Math.log(2.0));
	     		convergence_sgd_im.set(i, convergence_sgd_im.get(i)+Math.log(epsilon+sgd_importance.getParam().squaredNormOfDifferenceTo(b_n_star))/Math.log(2.0));
				convergence_sgd_opt.set(i,convergence_sgd_opt.get(i)+Math.log(epsilon+sgd.getParam().squaredNormOfDifferenceTo(b))/Math.log(2.0));
				convergence_im_opt.set(i,convergence_im_opt.get(i)+Math.log(epsilon+ sgd_importance.getParam().squaredNormOfDifferenceTo(b))/Math.log(2.0));
			}
		}
		for(int i=0;i<MaxItr;i++){ 
			convergence_sgd.set(i,convergence_sgd.get(i)/rep); 
			convergence_sgd_im.set(i,convergence_sgd_im.get(i)/rep); 
			convergence_sgd_opt.set(i,convergence_sgd_opt.get(i)/rep); 
			convergence_im_opt.set(i,convergence_im_opt.get(i)/rep); 
		}
		List<List<Double>> series = new ArrayList<List<Double>>();
		series.add(convergence_sgd);
		series.add(convergence_sgd_im); 
		List<String> snames = new ArrayList<String>(); 
		snames.add("SGD");
		snames.add("Approximate Cosine Sampler");
		XYLinesChart chart = new XYLinesChart(series,itrs,snames,"Convergence rate towards empirical optimal","Number of Iterations","Log_2(|w^s - w_n^*|^2)");
		
		int width = 800; /* Width of the image */
        int height = 600; /* Height of the image */ 
        File chartfile = new File("outs/empirical_importance_cosine_app.JPEG"); 
        chart.save(chartfile, width, height);
        chart.setVisible(true);
        
        List<List<Double>> series2 = new ArrayList<List<Double>>();
		series2.add(convergence_sgd_opt);
		series2.add(convergence_im_opt); 
		List<String> snames2 = new ArrayList<String>(); 
		snames2.add("SGD");
		snames2.add("Approximate Cosine Sampler");
		XYLinesChart chart2 = new XYLinesChart(series2,itrs,snames2,"Convergence rate towards the true optimal","Number of Iterations","Log_2(|w^s - w^*|^2)");
		
        File chartfile2 = new File("outs/optimal_importance_cosine_app.JPEG"); 
        chart2.save(chartfile2, width, height);
        chart2.setVisible(true);
        
		
	}
}
