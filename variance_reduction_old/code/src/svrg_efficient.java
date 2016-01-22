import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
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
import opt.loss.Loss_static_efficient;
import data.DataPoint;
import data.Result;
import data.SparsePoint;


public class svrg_efficient {
	public static DataPoint[] data; 
	public static SAGA saga_opt; 
	public static void readDataPointsFromFile(String filename, int startIndex, int data_size) {
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
					data[c] = point; 
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
		data = new DataPoint[conf.c0]; 
		readDataPointsFromFile( conf.dataPath, 1,conf.c0);
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
		}
		L = 1.5;
		double lambda_n = 1.0/Math.sqrt(n);
		double eta_n = 0.3/(L+lambda_n*n); 
		if(conf.agressive_step){ 
			eta_n = 1.0/(3*L);
		}
		Loss_static_efficient loss = new Logistic_Loss_efficient(data, d);
		if(conf.lossType ==  opt.config.Config.LossType.REGRESSION){
			loss = new LeastSquares_efficient(data,d); 
		}
		loss.setLambda(lambda_n);
		First_Order_Factory_efficient.methods_in = new FirstOrderOpt[2];
		
		double loss_opt = 0; 
		if(conf.T0 == -1){
			saga_opt = new SAGA(loss,eta_n); 
			saga_opt.Iterate((int) (2*n*Math.log(n)));//TODO 
//			opt.Iterate(1000);
			System.out.println("After SAGA: Free memory (bytes): " + 
					  Runtime.getRuntime().freeMemory()+ ",Total memory (bytes): " + 
							  Runtime.getRuntime().totalMemory());
			loss_opt = loss.getLoss(saga_opt.getParam()); 
			saga_opt = null; 
			System.gc(); 
			System.out.println("After calling GC: Free memory (bytes): " + 
					  Runtime.getRuntime().freeMemory()+ ",Total memory (bytes): " + 
							  Runtime.getRuntime().totalMemory());
		} 
		else{ 
			loss_opt = conf.T0; 
		}
		
		System.out.println("loss_opt:"+loss_opt);
		int b = 3; 
		int p = 2; 
		double kappa = L/lambda_n; 
		System.out.println("kapa:"+kappa);
		double eta = 1.0/(5*Math.pow(b, p+1));
		if(n>= 100000){
			eta = 1.0/(10*Math.pow(b, p+1)); 
		}
		System.out.println("eta:"+eta);
		int k_0 = (int) kappa;
		System.out.println("k_0:"+k_0);
		int m = (int) (kappa/(eta*5)); 
		System.out.println("m:"+m);
		SVRG_Streaming svrg = new SVRG_Streaming(loss.clone_loss(),eta, k_0, b,m); 
		SVRG_Streaming_Main svrg_main = new SVRG_Streaming_Main(loss.clone_loss(),eta, k_0, b,m); 
		First_Order_Factory_efficient.methods_in[0] = svrg; 
		First_Order_Factory_efficient.methods_in[1] = svrg_main; 
		First_Order_Factory_efficient.RunExperiment(numrep,loss, MaxItr, nSamplesPerPass, loss_opt,null,0,conf.logDir);
	}
}
