package backup;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import opt.firstorder.old.SAGA_old;
import plot.XYLinesChart;
import data.DataPoint;
import data.IOTools;
import data.SparsePoint;
/**
 * Implementation of stochastic optimization methods with adaptive sampling
 * 
 * @author Hadi Daneshmand
 *
 */



public class Adapt_ss {
	
	
	public static List<DataPoint> getsubsample(List<DataPoint> in, List<Integer> indices){ 
		List<DataPoint> out = new ArrayList<DataPoint>(); 
		for(int i = 0;i<indices.size();i++){ 
			out.add(in.get(indices.get(i)));
		}
		return out;
	}
  public static void main(String[] args) {
	  System.out.println("Working Directory = " +
              System.getProperty("user.dir"));
	  String configFilename = System.getProperty("user.dir")+"/configs/config_covtype.txt";
		if(args.length > 0) {
			configFilename = args[0];
		}
		opt.config.Config conf = new opt.config.Config();
		conf.parseFile(configFilename);
		
		System.out.println("algType: " + conf.algType.toString());
		System.out.println("eta0: " + conf.eta0);
		System.out.println("datapointSelection " + conf.datapointSelection);
		System.out.println("lambda: " + conf.lambda);
		System.out.println("lossType: " + conf.lossType);
		System.out.println("nTrials: " + conf.nTrials);
		System.out.println("nPasses: " + conf.nPasses);
		System.out.println("nSamplesPerPass: " + conf.nSamplesPerPass);
		System.out.println("objType: " + conf.objType);
		System.out.println("randomizeValidationSet: " + conf.randomizeValidationSet);
		System.out.println("validation_set_ratio: " + conf.validation_set_ratio);	
		System.out.println("svrg_outer_pSamples: " + conf.svrg_outer_pSamples);
		System.out.println("svrg_pSamples: " + conf.svrg_pSamples);
		

		new File(conf.logDir).mkdir();
		
		List<Double> w_m_plus_star = new ArrayList<Double>(); 
		List<Double> w_n_plus_star = new ArrayList<Double>(); 
		List<Double> w_s_conv = new ArrayList<Double>(); 
		List<Double> w_s_conv_n = new ArrayList<Double>();
		List<Double> w_n_w_m = new ArrayList<Double>();
		int maxItr = 2000;
		List<DataPoint> data = IOTools.readDataPointsFromFile(conf.dataPath, conf.startIndex);
		for(int i=0;i<data.size();i++){ 
			DataPoint p = data.get(i); 
			if(p.getLabel() == 2){ 
				p.setLabel(-1);
			}
			data.set(i, p); 
		}
		int n = data.size();
		System.out.println("Loaded " + n + " points from " + conf.dataPath);
		String filename = "datas/optimals_covtype";	
		Optimals ops = Optimals.ParseOptimals(filename);
		conf.nSamplesPerPass = 50;
		
		List<DataPoint> fullData = getsubsample(data, ops.optimal_n.indices); 
		System.out.println("fullDataSize="+fullData.size());
		DataPoint w_n_star = ops.optimal_n.optimalValue; 
		OptimalContainer subOptimal = ops.optimal_ms.get(7).get(0); 
		DataPoint w_m_star = subOptimal.optimalValue; 
		System.out.println("subsample size:"+subOptimal.m);
		List<DataPoint> subData = getsubsample(data, subOptimal.indices);
		System.out.println("subData size:"+subData.size());
		double L = 3;
		System.out.println();
		double eta_m = 1.0/(2*(subOptimal.m*conf.lambda+L ));
		double eta_n = 1.0/(2*(ops.optimal_n.m*conf.lambda+L )); 
		System.out.println("eta_m:"+eta_m);
		System.out.println("eta_n:"+eta_n);
		SAGA_old saga_m = new SAGA_old(subData, conf,conf.lambda,eta_m); 
		saga_m.setOptimal(subOptimal.optimalValue);
//		saga_m.OnePassOverData();
		SAGA_old saga_n = new SAGA_old(fullData,conf,conf.lambda,eta_n);
//		saga_n.OnePassOverData();
		SAGA_old saga_n_conver = new SAGA_old(fullData, conf,conf.lambda, eta_n);
//		saga_n_conver.OnePassOverData();
		double nfact = 1.0/w_n_star.getNorm();
		nfact = nfact*nfact;
		for(int i=0;i<maxItr;i++){ 
			DataPoint temp = saga_m.getParam();
			SparsePoint pre_w = new SparsePoint(); 
			for(int j=0;j<conf.featureDim;j++){
				pre_w.set(j, 0);
			}
			pre_w = (SparsePoint) pre_w.add(temp);
			saga_m.OneIteration(conf.nSamplesPerPass);
			saga_n_conver.OneIteration(conf.nSamplesPerPass);
			w_s_conv_n.add(nfact*saga_n_conver.getParam().squaredNormOfDifferenceTo(w_n_star));
			w_s_conv.add(nfact*saga_m.getParam().squaredNormOfDifferenceTo(w_m_star)); 
			w_m_plus_star.add(nfact*saga_m.getParam().squaredNormOfDifferenceTo(w_n_star)); 
			saga_n.setParam(pre_w); 
			saga_n.OneIteration(conf.nSamplesPerPass);
			w_n_plus_star.add(nfact*saga_n.getParam().squaredNormOfDifferenceTo(w_n_star));
		}
		for(int i=0;i<w_n_plus_star.size();i++){ 
			w_n_w_m.add(w_n_star.squaredNormOfDifferenceTo(w_m_star)*nfact); 
		}
		List<List<Double>> series = new ArrayList<List<Double>>();
		series.add(w_s_conv);
		series.add(w_s_conv_n);
		series.add(w_m_plus_star); 
		series.add(w_n_plus_star);
		series.add(w_n_w_m);
		
		List<String> snames = new ArrayList<String>(); 
		snames.add("|w^t-w_m^*|^2/|w_n^*|^2");
		snames.add("|w^t-w_n^*|^2/|w_n^*|^2");
		snames.add("|w_m_plus-w_n^*|^2/|w_n^*|^2");
		snames.add("|w_n_plus-w_n^*|^2/|w_n^*|^2");
		snames.add("|w_n_star - w_m_star|"); 
		ArrayList<Double> t = new ArrayList<Double>(); 
		for(int i=0;i<w_n_plus_star.size();i++){ 
			double normalized = Math.log((i+1)*conf.nSamplesPerPass);
			t.add(normalized);
		}
		
    	XYLinesChart chart = new XYLinesChart(series,t,snames,"Distances","Number of Samples","Distances");
    	System.out.println("Squared Norm of Optimals:"+ops.optimal_n.optimalValue.squaredNorm());
    	int width = 1200; /* Width of the image */
        int height = 900; /* Height of the image */ 
        File chartfile = new File("outs/distances_covtype.JPEG"); 
        chart.save(chartfile, width, height);
        chart.setVisible(true);
        List<Double> delta = new ArrayList<Double>(); 
        for(int i=0;i<w_m_plus_star.size();i++){
        	delta.add((w_n_plus_star.get(i)-w_m_plus_star.get(i))/nfact);
        }
        
        List<List<Double>> series2 = new ArrayList<List<Double>>(); 
        List<String> names2 = new ArrayList<String>(); 
        names2.add("|w_n_plus-w_n|-|w_m_plus-w_n|");
        series2.add(delta); 
        XYLinesChart chart2 = new XYLinesChart(series2, t, names2, "Delta", "Log(iterations)", "Delta");
        File chartfile2 = new File("outs/delta_covtype.JPEG");
        chart2.save(chartfile2, width, height);
        chart2.setVisible(true);
        
//		List<List<Double>> series = new ArrayList<List<Double>>();
//		for(int i=0;i<ops.optimal_ms.size();i++){
//			int ms = ops.ms.get(i); 
//			int s = 0; 
//			ArrayList<OptimalContainer> opts_m = ops.optimal_ms.get(i); 
//			for(int j=2;j<opts_m.size();j++){
//				
//				List<DataPoint> sampled_set = getsubsample(data,opts_m.get(j).indices);
//				System.out.println("ms="+ms);
//				double mu = 0.1; 
//				double gama_m = 1.0/(mu*ms+1);
//				double gama_n = 1.0/(mu*ops.optimal_n.m+1);
//				
//				SAGA saga = new SAGA(sampled_set, conf,gama_m); 
//				s+=saga.FindCriticalIndices(ops.optimal_n.optimalValue, gama_n,mu);
//			}
//			System.out.println("s="+(1.0*s)/opts_m.size());
//			ss.add((1.0*s)/opts_m.size());
//		}
//		new XYLineChart(ss,"s_plot","Convergence","Iteration","Objective Value").setVisible(true);
  }


  
}
