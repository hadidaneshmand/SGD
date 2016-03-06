package backup;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import opt.Adapt_Strategy;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.SAGA;
import opt.firstorder.SAGA_Adapt;
import opt.firstorder.SAGA_Adapt_WithoutMemory;
import opt.firstorder.SGD;
import opt.loss.LeastSquares;
import opt.loss.Loss;

import org.ejml.simple.SimpleMatrix;

import data.DataPoint;
import data.DensePoint;
import data.Result;
import data.SparsePoint;


public class AdaptSAGA_SYN {
	public static DataPoint regression(List<DataPoint> data,int d ){ 
		DataPoint out = new SparsePoint(); 
		SimpleMatrix y = new SimpleMatrix(data.size(),1);
		SimpleMatrix X = new SimpleMatrix(data.size(),d); 
		for(int i =0 ;i<data.size();i++){ 
			for(int j=0;j<d;j++){ 
				X.set(i, j, data.get(i).get(j));
			}
			y.set(i, data.get(i).getLabel());
		}
		SimpleMatrix xty = X.transpose().mult(y); 
		SimpleMatrix xtx = X.transpose().mult(X);
		SimpleMatrix b_hat = xtx.invert().mult(xty);
		for(int i=0;i<d;i++){ 
			out.set(i, b_hat.get(i));
		}
		return out; 
	}
	public static List<DataPoint> getsubsample(List<DataPoint> in, List<Integer> indices){ 
		List<DataPoint> out = new ArrayList<DataPoint>(); 
		for(int i = 0;i<indices.size();i++){ 
			out.add(in.get(indices.get(i)));
		}
		return out;
	}
	public static void main(String[] args) {
		double lambda = 0; 
		int nSamplesPerPass = 10000;
		int MaxItr = 20; 
		Random r = new Random(); 
		int d = 20; 
		int n = 100000; 
		DataPoint b = new DensePoint(d); 
		for(int i = 0;i < d;i++){ 
			b.set(i, r.nextDouble());
		}
		b = (DataPoint) b.normalize();
		
		double cf = 0.5;
		List<DataPoint> data = AdaptSAGA_DependencyTest.generateData(0.4,n, b, d, cf);
//		DataPoint b_n_star = regression(data,d);
//		System.out.println("distances:"+b_n_star.squaredNormOfDifferenceTo(b));
		double conditionNum = 1.0/Math.pow(n, cf);
		double mu =conditionNum;
		System.out.println("condition number of the matrix: "+ conditionNum);
		System.out.println("mu="+mu);
		double L = 1.5;
		System.out.println("L:"+L);
		ArrayList<String> snames = new ArrayList<String>(); 
		snames.add("saga");
		snames.add("sagaAdapt");
		snames.add("SGD");
		snames.add("sagaMemoryFree");
		snames.add("new_adapt"); 
		snames.add("steps");
		Result res = new Result(snames); 
		int rep = 1; 
		DataPoint b_n = regression(data, d); 
		for(int i=0;i<rep;i++){
			System.out.println("Simul:"+i);
			runExperimentOnece( res, data, mu, L,b, d, MaxItr, nSamplesPerPass); 
		}
		System.out.println("opt_dist:"+b_n.squaredNormOfDifferenceTo(b));
		res.write2File("outs/saga_adapt_sgd_newadapt_small");
//		List<List<Double>> series = new ArrayList<List<Double>>();
//		series.add(conver_saga);
//		series.add(conver_adapt);
//		series.add(conv_gd);
//		series.add(conv_memofree);
////		series.add(conv_gd); 
//		List<String> snames = new ArrayList<String>(); 
//		snames.add("saga");
//		snames.add("adapt_saga");
//		snames.add("gd_adapt");
//		snames.add("MemoryFree");
////		snames.add("GD");
//		XYLinesChart chart = new XYLinesChart(series,t,snames,"Convergence rate","Number of Iterations","Log(Distances to w^_n^*)");
//		int width = 800; /* Width of the image */
//        int height = 600; /* Height of the image */ 
//        File chartfile = new File("outs/converage_syntactic.JPEG"); 
//        chart.save(chartfile, width, height);
//        chart.setVisible(true);
        
	}
	public static void runExperimentOnece(Result res, List<DataPoint> data, double mu, double L,DataPoint b,int d,int MaxItr,int nSamplesPerPass){ 
		int n = data.size(); 
		double eta_n = (0.3/(mu*n+L));
		FirstOrderOpt[] opt_meth = new FirstOrderOpt[5]; 
		SAGA saga = new SAGA(new LeastSquares(data, d),eta_n);
		Adapt_Strategy as_saga = new Adapt_Strategy(n, ((int)( 1.0/mu)), false);
		Adapt_Strategy as_saga_fr = new Adapt_Strategy(n, ((int)( 1.0/mu)), false);
		Adapt_Strategy as_saga_gd = new Adapt_Strategy(n, ((int)( 1.0/mu)), false);
		Adapt_Strategy as_new = new Adapt_Strategy(n, ((int)( 1.0/mu)), false);
		SAGA_Adapt saga_adapt = new SAGA_Adapt(new LeastSquares(data, d), as_saga,mu,L);
		SAGA_Adapt_WithoutMemory saga_memofree = new SAGA_Adapt_WithoutMemory(new LeastSquares(data, d), as_saga_fr,mu,L);
		SGD sgd = new SGD(new LeastSquares(data, d));
		sgd.setLearning_rate(mu);
//		sgd.setConstant_step_size(true);
		SAGA_Adapt ad_new = new SAGA_Adapt(new LeastSquares(data, d), as_new,mu,L);
		Loss loss = new LeastSquares(data, d); 
		opt_meth[0] = saga; 
		opt_meth[1] = saga_adapt; 
		opt_meth[2] = sgd; 
		opt_meth[3] = saga_memofree;
		opt_meth[4] = ad_new;
		double epsilon = 1e-20;
		List<List> convs = new ArrayList<List>(); 
		for(int i=0;i<opt_meth.length;i++){ 
			convs.add(new ArrayList<Double>());
		}
		double loss_star = loss.getLoss(b);
		DataPoint b_n = regression(data, d);
		double loss_n_star = loss.getLoss(b_n);
		System.out.println("opt_loss"+Math.log(Math.abs(loss_n_star-loss_star))/Math.log(2));
		List<Double> bound = new ArrayList<Double>();
		double eps = Double.MIN_VALUE; 
		for(int i=0;i<MaxItr;i++){ 
			
			for(int j=0;j<opt_meth.length;j++){ 
				opt_meth[j].Iterate(nSamplesPerPass);
				convs.get(j).add(Math.log(eps+Math.abs(loss.getLoss(opt_meth[j].getParam())-loss_n_star))/Math.log(2));
			}
		}
		for(int i=0;i<opt_meth.length;i++){ 
			res.addresult(res.getSeriesnames().get(i), convs.get(i));
		}
		
		ArrayList<Double> t = new ArrayList<Double>(); 
		for(int i=0;i<MaxItr;i++){ 
			double normalized = (i+1)*nSamplesPerPass;
			t.add(normalized);
		}
		res.addresult(res.getSeriesnames().get(res.getSeriesnames().size()-1), t);
	}
}
