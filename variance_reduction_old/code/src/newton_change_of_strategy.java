import opt.Adapt_Strategy_Double_Full;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory_efficient;
import opt.firstorder.Newton;
import opt.firstorder.NewtonDataDriven;
import opt.loss.Dyna_regularizer_loss_e;
import opt.loss.Locality;
import opt.loss.Logistic_Loss_efficient;
import opt.loss.Loss;
import opt.loss.SecondOrderLoss;
import data.DataPoint;
import data.DensePoint_efficient;


public class newton_change_of_strategy {
	public static void main(String[] args) {
		Input.initialize(args);
		Loss loss = Input.loss_train.clone_loss(); 
		Logistic_Loss_efficient.buildHessians(Input.data, Input.d);
		int n = loss.getDataSize();
		int d = loss.getDimension(); 
		loss.set_lambda(1.0/n);
		DataPoint initParam = (DataPoint) DensePoint_efficient.one(Input.loss_train.getDimension()).multiply(3.0); 
		int num_strategy = 7; 
		Dyna_regularizer_loss_e[] dyna_losses = new Dyna_regularizer_loss_e[num_strategy];
		for(int i=0;i<num_strategy;i++){ 
			double incrementFactor = 1+Math.pow(2, i-2); 
			Adapt_Strategy_Double_Full strategy = new Adapt_Strategy_Double_Full(loss.getDataSize(), 3*d, 1, 1); 
			strategy.setIncrement_factor(incrementFactor);
			dyna_losses[i] = new Dyna_regularizer_loss_e(loss.clone_loss(),strategy);
		}
		FirstOrderOpt[] methods_in = new FirstOrderOpt[num_strategy];
		for(int i = 0 ;i < num_strategy ; i++){ 
			methods_in[i] = new Newton(dyna_losses[i]); 
			methods_in[i].setName("c:"+(1+Math.pow(2, i-2)));
		}
//		Adapt_Strategy_Double_Full strategy = new Adapt_Strategy_Double_Full(loss.getDataSize(), 2*d, 1, 1); 
//		methods_in[num_strategy] = new Newton(new Locality(loss.clone_loss(), strategy, 0.1));
//		methods_in[num_strategy].setName("data-driven");
//		methods_in[num_strategy].setParam(initParam);
		First_Order_Factory_efficient.methods_in = methods_in; 
		First_Order_Factory_efficient.experiment_with_iterations_complexity(4, loss.clone_loss(), 8, -1.0, Input.loss_test, Input.config.logDir+"_newton_strategy", Input.L, false,n);
		methods_in = new FirstOrderOpt[num_strategy];
		for(int i = 0 ;i < num_strategy ; i++){ 
			Adapt_Strategy_Double_Full strategy = new Adapt_Strategy_Double_Full(loss.getDataSize(), 3*d, 1, 1); 
			methods_in[i] = new NewtonDataDriven((SecondOrderLoss) loss.clone_loss(),strategy,(Math.pow(2, i-2))); 
			methods_in[i].setName("c:"+(1+Math.pow(2, i-2)));
		}
		First_Order_Factory_efficient.methods_in = methods_in;
		First_Order_Factory_efficient.experiment_with_iterations_complexity(4, loss.clone_loss(), 8, -1.0, Input.loss_test, Input.config.logDir+"_newtondd_strategy", Input.L, false,n);
	}
}
