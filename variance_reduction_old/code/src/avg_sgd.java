import data.DataPoint;
import data.DensePoint_efficient;
import opt.Adapt_Strategy;
import opt.firstorder.Accelarted;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory_efficient;
import opt.firstorder.GD;
import opt.firstorder.LBFGS_external;
import opt.firstorder.Nesterov2;
import opt.firstorder.SAGA;
import opt.firstorder.SAGA_Adapt;
import opt.firstorder.SAGA_Adapt_Lambda;
import opt.firstorder.SGD;
import opt.firstorder.SGD_AVG;
import opt.loss.LeastSquares_efficient;
import opt.loss.Logistic_Loss_efficient;
import opt.loss.Loss;


public class avg_sgd {
	public static void main(String[] args) {
		Input.initialize(args);
		int len = 5; 
		Logistic_Loss_efficient train_loss = new Logistic_Loss_efficient(Input.data, Input.d);
		int n = Input.data.length; 
		train_loss.setLambda(1.0/n);
//		FirstOrderOpt[] methods = new FirstOrderOpt[len]; 
//		for(int i=0;i<len;i++){
//			double lambda = Math.pow(10, -1.0*(i+1));
//			Loss loss_i = train_loss.clone_loss(); 
//			loss_i.set_lambda(lambda);
//			methods[i] = new LBFGS_external(loss_i, 100);
//			methods[i].setParam((DataPoint) DensePoint_efficient.one(loss_i.getDimension()).multiply(3.0));
//			methods[i].setName("lambda:"+lambda);
//		}
		FirstOrderOpt[] methods = new FirstOrderOpt[2];
//		GD gd = new GD(train_loss.clone_loss()); 
//	    gd.setStepSize(0.01);
//		Nesterov2 acc = new Nesterov2(train_loss.clone_loss(),gd.clone_accelarable()); 
//		acc.setStepSize(0.01);
		double stepsize = 0.3/(1+Input.L);
		SAGA saga = new SAGA(train_loss.clone_loss(), stepsize);
		SAGA_Adapt_Lambda dynasaga = new SAGA_Adapt_Lambda(train_loss.clone_loss(), new Adapt_Strategy(n, (int) (2*Input.d), true), 1.0/n, Input.L);
		methods[1] = saga; 
		methods[0] = dynasaga; 
		Loss test_loss = new Logistic_Loss_efficient(Input.test_data, Input.d); 
	    First_Order_Factory_efficient.methods_in = methods; 
	    First_Order_Factory_efficient.run_experiment(5, train_loss.clone_loss(), 30, 2000, -1, test_loss,  Input.config.logDir+"_lss",Input.L,false);
	}
}
