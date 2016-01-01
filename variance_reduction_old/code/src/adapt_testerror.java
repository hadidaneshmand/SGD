import java.util.ArrayList;
import java.util.List;

import opt.Adapt_Strategy;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory;
import opt.firstorder.SAGA;
import opt.firstorder.SAGA_Adapt;
import opt.firstorder.SGD;
import opt.firstorder.SVRG_Streaming;
import opt.loss.LeastSquares;
import opt.loss.LogisticRegression;
import opt.loss.Logistic_Loss;
import opt.loss.Loss_static;
import opt.loss.MissClass;
import data.DataPoint;
import data.IOTools;
import data.Result;


public class adapt_testerror {
	public static List<DataPoint> getsubsample(List<DataPoint> in, List<Integer> indices){ 
		List<DataPoint> out = new ArrayList<DataPoint>(); 
		for(int i = 0;i<indices.size();i++){ 
			out.add(in.get(indices.get(i)));
		}
		return out;
	}
	public static void main(String[] args) {
		List<DataPoint> data = IOTools.readDataPointsFromFile( "data/ijcnn1", 1);
		List<DataPoint> data_test = IOTools.readDataPointsFromFile("data/ijcnn1.t", 1);
		System.out.println("n="+data.size());
		int nSamplesPerPass = 1000;
		int MaxItr = 500; 
		int d = 24;
		Loss_static loss_test = new MissClass(data_test, d); 
		loss_test.setLambda(0.0);
		int n = data.size();
		double L = 2; 
		for(int i=0;i<n;i++){ 
			if(data.get(i).getNorm()>L){ 
				L = data.get(i).getNorm();
			}
		}
		System.out.println("L:" + L);
		double lambda_n = 1.0/n;
		double eta_n = 0.3/(L+lambda_n*n); 
		Loss_static loss = new Logistic_Loss(data, d);
		loss.setLambda(lambda_n);
		SGD sgd = new SGD(loss);
		sgd.setLearning_rate(0.05);
		sgd.setConstant_step_size(true);
		FirstOrderOpt[] methods = new FirstOrderOpt[4];
		methods[0] = sgd;
		methods[1] = new SAGA(loss,eta_n);
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
//		SVRG_Streaming svrg = new SVRG_Streaming(loss.clone_loss(),eta, k_0, b,m); 
//		methods[4] = svrg; 
		int numrep = 2;
		Result res = First_Order_Factory.RunExperiment(numrep,loss, methods, MaxItr, nSamplesPerPass, 0);
        res.write2File("outs/ijcnn1_streaming_test");
		
	}
}
