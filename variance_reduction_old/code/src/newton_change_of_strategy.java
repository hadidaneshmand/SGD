import opt.Adapt_Strategy_Double_Full;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory_efficient;
import opt.firstorder.Newton;
import opt.loss.Dyna_regularizer_loss_e;
import opt.loss.Loss;
import data.DataPoint;
import data.DensePoint_efficient;


public class newton_change_of_strategy {
	public static void main(String[] args) {
		Input.initialize(args);
		Loss loss = Input.loss_train.clone_loss(); 
//		Logistic_Loss_efficient.buildHessians(Input.data, Input.d);
		int n = loss.getDataSize();
		int d = loss.getDimension(); 
		loss.set_lambda(1.0/n);
		DataPoint initParam = (DataPoint) DensePoint_efficient.one(Input.loss_train.getDimension()).multiply(3.0); 
		int num_strategy = 4; 
		Dyna_regularizer_loss_e[] dyna_losses = new Dyna_regularizer_loss_e[num_strategy]; 
		for(int i=0;i<num_strategy;i++){ 
			double incrementFactor = 5 + 5*i; 
			Adapt_Strategy_Double_Full strategy = new Adapt_Strategy_Double_Full(loss.getDataSize(), d, 1, 1); 
			strategy.setIncrement_factor(incrementFactor);
			dyna_losses[i] = new Dyna_regularizer_loss_e(loss.clone_loss(),strategy );
		}
		FirstOrderOpt[] methods_in = new FirstOrderOpt[num_strategy];
		for(int i = 0 ;i < num_strategy ; i++){ 
			methods_in[i] = new Newton(dyna_losses[i]); 
			methods_in[i].setParam(initParam);
			methods_in[i].setName("c:"+(5 + 5*i));
		}
		
		First_Order_Factory_efficient.methods_in = methods_in; 
		First_Order_Factory_efficient.experiment_with_iterations_complexity(1, loss.clone_loss(), 8, -1.0, Input.loss_test, Input.config.logDir+"_newton_data", Input.L, false,n);
	}
}
