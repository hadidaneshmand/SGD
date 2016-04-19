import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.StringTokenizer;

import opt.Adapt_Strategy;
import opt.config.Config;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory;
import opt.firstorder.First_Order_Factory_efficient;
import opt.firstorder.SAGA;
import opt.firstorder.SAGA_Adapt;
import opt.firstorder.SGD;
import opt.firstorder.SVRG_Streaming;
import opt.firstorder.SVRG_Streaming_Main;
import opt.loss.LeastSquares_efficient;
import opt.loss.Logistic_Loss_efficient;
import opt.loss.Loss;
import opt.loss.FirstOrderEfficient;
import opt.loss.MissClass_efficient;
import data.DataPoint;
import data.Result;
import data.SparsePoint;


public class variance_in_opt {
	public static DataPoint[] data; 
	public static DataPoint[] test_data = null; 
	public static void readDataPointsFromFile(String filename, int startIndex, int data_size, boolean is_test) {
		int pos = 0; 
		int neg = 0; 
		try {
			BufferedReader fp = new BufferedReader(new FileReader(filename));
			String line;
			int c = 0; 
			while ((line = fp.readLine()) != null && c <data_size) {
				try {
					DataPoint point = new SparsePoint();
					StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
					double label = Double.valueOf(st.nextToken());					// label has to be at the first position of the text row
					if(label == 1){ 
						pos++;
					}
					else if(label == -1 || label == 2 || label == 0){
						label = -1; 
						neg++; 
					}
					point.setLabel(label);

					while (st.hasMoreTokens()) {
						int feature = Integer.valueOf(st.nextToken()) - startIndex;
						double value = Double.valueOf(st.nextToken());
						point.set(feature, value);
					}
					if(is_test){ 
						test_data[c] = point;
					}else{ 
						data[c] = point;
					}
					 
					c++;
				if(c %50000 == 0){ 
					System.out.println("c:"+c);	
				}
				if(c<10){ 
					System.out.println(point.toString());
				}
				} catch (NumberFormatException e) {
					System.out.println("Could not read datapoint number "+c + " since Line "+line+" seems to be not properly formatted: "+e.getMessage());
				} 
			}
			fp.close();
			System.out.println("pos:"+pos+",neg:"+neg);
		} catch (IOException e) {
			System.out.println("Could not read from file " + filename + " due to " + e.getMessage());
		}
		System.out.println("Free memory (bytes): " + 
				  Runtime.getRuntime().freeMemory());
		System.out.println("Total memory (bytes): " + 
				  Runtime.getRuntime().totalMemory());
		
	}
	public static void main(String[] args) {
		String configFilename = null;
		if(args.length > 0) {
			configFilename = args[0];
		}
		else{
			System.out.println("Config file is missed");
			return; 
		}
		opt.config.Config conf = new opt.config.Config();
		conf.parseFile(configFilename);

//		conf.doubling = false; 
//		conf.agressive_step = false; 
//		conf.nPasses = 100; 
//		conf.nSamplesPerPass = 100; 
//		conf.c0 = 1000; 
//		conf.dataPath = "data/covtype"; 
//		conf.featureDim = 54; 
//		conf.logDir = "outs/test"; 
		System.out.println("Total memory (bytes): " + 
				  Runtime.getRuntime().totalMemory());
		System.out.println("agressive step size for saga: " + conf.agressive_step);
		System.out.println("doubling: " + conf.doubling);
		System.out.println("nTrials: " + conf.nTrials);
		System.out.println("nPasses: " + conf.nPasses);
		System.out.println("nSamplesPerPass: " + conf.nSamplesPerPass);
		System.out.println("file: "+conf.dataPath);
		System.out.println("data size:"+conf.c0);
		System.out.println("out dir:"+conf.logDir);
		System.out.println("classification:" +(conf.lossType != Config.LossType.REGRESSION));
		System.out.println("testfile:"+conf.testFile);
		System.out.println("test_ratio:"+conf.train_ratio);
		int train_si = conf.c0; 
		int test_si = -1; 
		if(conf.train_ratio!= -1){
			train_si = (int)(conf.train_ratio*conf.c0); 
			test_si = conf.c0 - train_si; 
			
		}
		data = new DataPoint[conf.c0]; 
		readDataPointsFromFile( conf.dataPath, 1,conf.c0,false);
		FirstOrderEfficient test_loss = null; 
		
		if(conf.testFile != null && !conf.testFile.isEmpty() ){ 
			test_data = new DataPoint[conf.ntest]; 
			readDataPointsFromFile(conf.testFile, 1,conf.ntest, true);
			test_loss = new Logistic_Loss_efficient(test_data, conf.featureDim); 
			test_loss.setLambda(0);
		}
		if(test_si!=-1){
			DataPoint[] train_data = new DataPoint[train_si]; 
			test_data = new DataPoint[test_si];
			ArrayList<Integer> inds = new ArrayList<Integer>(); 
			for(int i=0;i<conf.c0;i++){ 
				inds.add(i); 
			}
			Collections.shuffle(inds);
			for(int i=0;i<conf.c0;i++){ 
				if(i<train_si){
					train_data[i] = data[i]; 
				}
				else{ 
					test_data[i-train_si] = data[i]; 
				}
			}
			System.out.println("t_data="+test_data[0]);
			data = train_data; 
			System.out.println("t_data_after="+test_data[0]);
			test_loss = new Logistic_Loss_efficient(test_data, conf.featureDim); 
			test_loss.setLambda(0);
		}
		
		int numrep = conf.nTrials;
		int nSamplesPerPass = conf.nSamplesPerPass; 
	    int MaxItr = conf.nPasses; 
	    int d = conf.featureDim; 
	    System.out.println("dim:"+d);
	    int n = data.length;
	    double L = 2; 
//		for(int i=0;i<n;i++){ 
//			data.set(i, (DataPoint) data.get(i).multiply(10)); 
//		}
		for(int i=0;i<n;i++){ 
			if(data[i].getNorm()>L){ 
				L = data[i].getNorm();
			}
		}
		System.out.println("L:"+L);
		if(L>60.0){ 
			for(int i=0;i<data.length;i++){ 
				data[i]=(DataPoint)data[i].normalize(); 
			}
			System.out.println("Data is normalized!!");
			L = 1.5;
		}
		
		double[] lambdas = new double[5]; 
		lambdas[0] = 0.1; 
		lambdas[1] = 0.01; 
		lambdas[2] = 0.001; 
		lambdas[3] = 0.0001;
		lambdas[4] = 0.00001; 
//		lambdas[5] = 0.000001; 
		double eta = 0.005;
		System.out.println("step size:"+eta);
		ArrayList<String> names = new ArrayList<String>(); 
		names.add("suboptimality"); 
		names.add("distance2optimal");
		names.add("variance");
		names.add("steps");
		Result result = new Result(names);
		SAGA[] saga_opts = new SAGA[lambdas.length]; 
		FirstOrderEfficient[] losses = new FirstOrderEfficient[lambdas.length]; 
		for(int i=0;i<lambdas.length;i++){
			
			losses[i] = new Logistic_Loss_efficient(data, d);
			losses[i].set_lambda(lambdas[i]);
			double eta_n = 0.3/(L+lambdas[i]*n); 
			saga_opts[i] = new SAGA(losses[i], eta_n);
			saga_opts[i].Iterate((int) (7*n*Math.log(n)));
			System.out.println("saga["+i+"]: optimized!!"); 
		}
		for(int k=0;k<10;k++){
			ArrayList<Double> lambdas_arr = new ArrayList<Double>(); 
			ArrayList<Double> subopts = new ArrayList<Double>();
			ArrayList<Double> dist2opts = new ArrayList<Double>(); 
			ArrayList<Double> variances = new ArrayList<Double>(); 
			for(int i=0;i<lambdas.length;i++){
				double lambda_n = lambdas[i];
				lambdas_arr.add(1.0/lambda_n); 
				SGD sgd = new SGD(losses[i]); 
				sgd.setStepSize(eta);
				sgd.setConstant_step_size(true);
				sgd.Iterate(n*50);
				List<DataPoint> grads = losses[i].getAllStochasticGradients(saga_opts[i].getParam()); 
				double grad_var = 0; 
				for(int j=0;j<grads.size();j++){ 
					grad_var += grads.get(j).squaredNorm(); 
				}
				grad_var = grad_var/grads.size(); 
				double dist2opt = sgd.getParam().squaredNormOfDifferenceTo(saga_opts[i].getParam()); 
				dist2opts.add(dist2opt);
				double subopt = losses[i].computeLoss(sgd.getParam())-losses[i].computeLoss(saga_opts[i].getParam()); 
				subopts.add(subopt);
				variances.add(grad_var); 
				System.out.println("lambda:"+lambda_n+",variance:"+grad_var+",suboptimality:"+subopt+",dist2opt:"+dist2opt);
				System.out.println("-----------------------");
			}
			result.addresult("steps", lambdas_arr);
			result.addresult("variance", variances);
			result.addresult("suboptimality",subopts);
			result.addresult("distance2optimal",dist2opts);
		}
		result.write2File(conf.logDir+"_sgd_constant_2");
	}
}
