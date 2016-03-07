import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.StringTokenizer;

import opt.Adapt_Strategy_Double_Full;
import opt.Adapt_Strategy_GD;
import opt.config.Config;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.GD;
import opt.firstorder.GD_Adapt;
import opt.firstorder.LBFGS;
import opt.firstorder.SAGA;
import opt.loss.Adaptss_loss_efficient;
import opt.loss.LeastSquares_efficient;
import opt.loss.Logistic_Loss_efficient;
import opt.loss.Loss_static_efficient;
import data.DataPoint;
import data.Result;
import data.SparsePoint;


public class gd_adapt {
	public static DataPoint[] data; 
	public static DataPoint[] test_data = null; 
	public static SAGA saga_opt; 
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
		Loss_static_efficient test_loss = null; 
		
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
		}
		L = 1.5;
		double lambda_n = 1.0/Math.sqrt(n);
		if(conf.lambdaType == Config.LambdaType.SMALL){ 
			lambda_n = Math.pow(1.0/n, 3.0/4); 
			conf.logDir+="_small";
		} 
		else if(conf.lambdaType == Config.LambdaType.MEDIUM){ 
			lambda_n = Math.pow(1.0/n, 2.0/3);
			conf.logDir+="_large";
		}
			
		System.out.println("lambda:"+lambda_n);
		double eta_n = 0.3/(L+lambda_n*n); 
		if(conf.agressive_step){ 
			eta_n = 1.0/(3*L);
		}
		Loss_static_efficient loss = new Logistic_Loss_efficient(data, d);
		if(conf.lossType ==  opt.config.Config.LossType.REGRESSION){
			loss = new LeastSquares_efficient(data,d); 
		}
		loss.setLambda(lambda_n);
		double learning_rate = 0.8/L;
		double loss_opt = 0; 
		if(conf.opt_train == -1){
			saga_opt = new SAGA(loss,eta_n); 
			saga_opt.Iterate((int) (n*Math.log(n)*5));
			
//			opt.Iterate(1000);
			System.out.println("After SAGA: Free memory (bytes): " + 
					  Runtime.getRuntime().freeMemory()+ ",Total memory (bytes): " + 
							  Runtime.getRuntime().totalMemory());
			loss_opt = loss.getLoss(saga_opt.getParam()); 
			
//			System.out.println("After calling GC: Free memory (bytes): " + 
//					  Runtime.getRuntime().freeMemory()+ ",Total memory (bytes): " + 
//							  Runtime.getRuntime().totalMemory());
		} 
		else{ 
			loss_opt = conf.opt_train; 
		}
		System.out.println("loss_opt:"+loss_opt);
		FirstOrderOpt[] methods = new FirstOrderOpt[4];
		GD gd = new GD(loss.clone_loss()); 
		gd.setLearning_rate(learning_rate);
		methods[0] = gd; 
		Adapt_Strategy_GD as_gd = new Adapt_Strategy_GD(n,1.0*L/lambda_n);
		Adaptss_loss_efficient adapt_loss = new Adaptss_loss_efficient(loss.clone_loss(), as_gd);
		GD gd_large = new GD(adapt_loss);
		gd_large.setLearning_rate(learning_rate);
		methods[1] = gd_large; 
	    int m_lbfgs = 20; 
		LBFGS lbfgs = new LBFGS(loss.clone_loss(),m_lbfgs); 
		lbfgs.setLearning_rate(learning_rate);
		methods[2] = lbfgs;
		double kappa = (L/lambda_n);
		System.out.println("kappa:"+kappa);
		Adaptss_loss_efficient adaptloss = new Adaptss_loss_efficient(loss, new Adapt_Strategy_Double_Full(n, 2*d, 2));
		LBFGS lbfgs_adapt = new LBFGS(adaptloss, m_lbfgs); 
		lbfgs_adapt.setLearning_rate(learning_rate);
		methods[3] = lbfgs_adapt; 
		ArrayList<String> names = new ArrayList<String>(); 
		names.add("gd");names.add("gd_steps");names.add("dyna-sgd");names.add("dyna-sgd_steps");names.add("lbfgs");names.add("lbfgs_steps"); names.add("dyna-lbfgs");names.add("dyna-lbfgs_steps"); 
		Result result = new Result(names); 
		Result result_test = new Result(names); 
		for(int i=0;i<conf.nTrials;i++){ 
			ArrayList<ArrayList<Double>> arr_results = new ArrayList<ArrayList<Double>>(); 
			for(int j=0;j<names.size();j++){
				arr_results.add(j, new ArrayList<Double>());
				if(j%2==0){
					FirstOrderOpt method = methods[(int)(j/2.0)]; 
					arr_results.get(j).add(loss.getLoss(method.getParam())-loss_opt); 
				}
				else{
					arr_results.get(j).add(0.0); 
				}
			}
			ArrayList<ArrayList<Double>> arr_test = new ArrayList<ArrayList<Double>>(); 
			if(test_loss != null){ 
				for(int j=0;j<names.size();j++){
					arr_test.add(j,new ArrayList<Double>()); 
					if(j%2 ==0){
						FirstOrderOpt method = methods[(int)(j/2.0)]; 
						arr_test.get(j).add(test_loss.getLoss(method.getParam())); 
					}
					else{
						arr_test.get(j).add(0.0);
					}
				}
			}
			for(int j=0;j<methods.length;j++){ 
				FirstOrderOpt method = methods[j].clone_method(); 
				int past_cg = 0; 
				while((method.getNum_computed_gradients()/(1.0*n)) < 5){  
					int new_cg = (int) (method.getNum_computed_gradients()/(0.5*n)) ;
					method.Iterate(1);
					if(new_cg!=past_cg){ 
						double error = loss.getLoss(method.getParam())-loss_opt; 
						System.out.println("======= "+names.get(2*j)+" =======");
						System.out.println("loss["+(method.getNum_computed_gradients()/(1.0*n))+"]="+error);
						arr_results.get(j*2).add(error); 
						arr_results.get(j*2+1).add(method.getNum_computed_gradients()/(1.0*n)); 
						if(test_loss!=null){
							double error_test = test_loss.getLoss(method.getParam()); 
							System.out.println("test_loss["+(method.getNum_computed_gradients()/(1.0*n))+"]="+error_test);
							arr_test.get(j*2).add(error_test); 
							arr_test.get(j*2+1).add(method.getNum_computed_gradients()/(1.0*n));
						}
					}
				   past_cg = new_cg; 
				}
			}
			for(int j=0;j<names.size();j++){ 
				result.addresult(names.get(j), arr_results.get(j));
			}
			if(test_loss!=null){
				for(int j=0;j<names.size();j++){ 
					result_test.addresult(names.get(j), arr_test.get(j));
				}
			}
		}
		result.write2File(conf.logDir+"_lbfgs");
		result_test.write2File(conf.logDir+"_lbfgs_test");
	}
}
