import java.util.List;
import java.util.Random;

import backup.AdaptSAGA_DependencyTest;
import backup.AdaptSAGA_SYN;
import opt.Adapt_Strategy;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory;
import opt.firstorder.SAGA;
import opt.firstorder.SAGAMemo;
import opt.firstorder.SAGA_Adapt;
import opt.firstorder.SGD;
import opt.firstorder.SVRG_Streaming;
import opt.loss.LeastSquares;
import opt.loss.Loss_static;
import data.DataPoint;
import data.DensePoint;
import data.Result;


public class sgd_saga_const {
	public static void main(String[] args) {
		Random r = new Random(); 
		int d = 100; 
		int n = 10000; 
		DataPoint b = new DensePoint(d); 
		for(int i = 0;i < d;i++){ 
			b.set(i, r.nextDouble());
		}
		b = (DataPoint) b.normalize();
		double cf = 0.5;
		List<DataPoint> data = AdaptSAGA_DependencyTest.generateData(0.4,n, b, d, cf);
		double mu = 1.0/Math.pow(n, cf);
		double L = 1; 
		for(int i=0;i<n;i++){ 
			if(data.get(i).getNorm()>L){
				L = data.get(i).getNorm();
			}
		}
		System.out.println("L="+L);
		double eta_n = 1.0/Math.sqrt(n); 
		Loss_static loss = new LeastSquares(data, d);
		loss.setLambda(0);
		SGD sgd = new SGD(loss.clone_loss());
		sgd.setLearning_rate(eta_n);
		sgd.setConstant_step_size(true);
		SAGA saga = new SAGA( loss.clone_loss() , eta_n);
		Adapt_Strategy as = new Adapt_Strategy(n, 100, false); 
		int b_s = 3; 
		int p = 2; 
		double kappa = L/mu; 
		System.out.println("kapa:"+kappa);
		double eta = 1.0/(5*Math.pow(b_s, p+1));
		System.out.println("eta:"+eta);
		int k_0 = (int) kappa;
		System.out.println("k_0:"+k_0);
		int m = (int) (kappa/eta); 
		System.out.println("m:"+m);
		SVRG_Streaming svrg = new SVRG_Streaming(loss.clone_loss(),eta, k_0, b_s,m); 
		SAGA_Adapt saga_adapt = new SAGA_Adapt(loss.clone_loss(), as.clone_strategy(), mu, L);
		FirstOrderOpt[] methods = new FirstOrderOpt[4]; 
		methods[0] = sgd; 
		methods[1] = saga; 
		methods[2] = svrg;
		methods[3] = saga_adapt; 
		DataPoint b_n_start = AdaptSAGA_SYN.regression(data, d); 
		Result res = First_Order_Factory.RunExperiment(4, loss.clone_loss(), methods, 50, 1000, loss.getLoss(b_n_start));
		res.write2File("outs/saga_sgd_const");
	}
}
