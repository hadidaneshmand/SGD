import java.util.ArrayList;
import java.util.List;

import opt.Adapt_Strategy;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory;
import opt.firstorder.SAGA;
import opt.firstorder.SAGA_Adapt;
import opt.firstorder.SGD;
import opt.firstorder.SVRG_Streaming;
import opt.loss.Logistic_Loss;
import opt.loss.Loss_static;
import opt.loss.LeastSquares;
import data.DataPoint;
import data.IOTools;
import data.Result;


public class sgd_saga_adapt {
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
		System.out.println("agressive step size for saga: " + conf.agressive_step);
		System.out.println("doubling: " + conf.doubling);
		System.out.println("nTrials: " + conf.nTrials);
		System.out.println("nPasses: " + conf.nPasses);
		System.out.println("nSamplesPerPass: " + conf.nSamplesPerPass);
		System.out.println("file: "+conf.dataPath);
		List<DataPoint> data = IOTools.readDataPointsFromFile( conf.dataPath, 1);
		int numrep = conf.nTrials;
		int nSamplesPerPass = conf.nSamplesPerPass; 
	    int MaxItr = conf.nPasses; 
	    int d = conf.featureDim; 
	    int n = data.size();
	    double L = 2; 
//		for(int i=0;i<n;i++){ 
//			data.set(i, (DataPoint) data.get(i).multiply(10)); 
//		}
		for(int i=0;i<n;i++){ 
			if(data.get(i).getNorm()>L){ 
				L = data.get(i).getNorm();
			}
		}
		System.out.println("L:"+L);
		if(L>10.0){ 
			for(int i=0;i<data.size();i++){ 
				data.set(i,(DataPoint)data.get(i).normalize()); 
			}
		}
		double lambda_n = 1.0/Math.sqrt(n);
		double eta_n = 0.3/(L+lambda_n*n); 
		if(conf.agressive_step){ 
			eta_n = 1.0/(3*L);
		}
		Loss_static loss = new Logistic_Loss(data, d);
		if(conf.lossType ==  opt.config.Config.LossType.REGRESSION){
			loss = new LeastSquares(data,d); 
		}
		loss.setLambda(lambda_n);
		SGD sgd = new SGD(loss);
		sgd.setLearning_rate(lambda_n);
		FirstOrderOpt[] methods = new FirstOrderOpt[5];
		SAGA opt = new SAGA(loss,eta_n); 
		opt.Iterate((int) (n*Math.log(n)));
		methods[0] = sgd;
		methods[1] = new SAGA(loss,eta_n);
		double loss_opt = loss.getLoss(opt.getParam()); 
		System.out.println("loss_opt:"+loss_opt);
		Adapt_Strategy as = new Adapt_Strategy(n, (int) (L/lambda_n), false);
		SAGA_Adapt saga_a = new SAGA_Adapt(loss.clone_loss(), as,lambda_n,L);
		Adapt_Strategy as_doubl = new Adapt_Strategy(n, (int) (L/lambda_n), true);
		methods[2] = saga_a;
		methods[3] = new SAGA_Adapt(loss.clone_loss(), as_doubl, lambda_n, L);
		int b = 3; 
		int p = 2; 
		double kappa = L/lambda_n; 
		System.out.println("kapa:"+kappa);
		double eta = 1.0/(5*Math.pow(b, p+1));
		System.out.println("eta:"+eta);
		int k_0 = (int) kappa;
		System.out.println("k_0:"+k_0);
		int m = (int) (kappa/eta); 
		System.out.println("m:"+m);
		SVRG_Streaming svrg = new SVRG_Streaming(loss.clone_loss(),eta, k_0, b,m); 
		methods[4] = svrg; 
		ArrayList<String> names = new ArrayList<String>(); 
		Result res = First_Order_Factory.RunExperiment(numrep,loss, methods, MaxItr, nSamplesPerPass, loss_opt);
        res.write2File(conf.logDir);
	}
}
