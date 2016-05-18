import opt.Adapt_Strategy_Double_Full;
import opt.SampleSizeStrategy;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory_efficient;
import opt.firstorder.Newton;
import opt.firstorder.NewtonData6;
import opt.firstorder.NewtonDataDriven;
import opt.firstorder.NewtonTangent;
import opt.loss.Dyna_regularizer_loss_e;
import opt.loss.Loss;
import opt.loss.SecondOrderLoss;

public class newton_test6{ 
	public static void main(String[] args) {
		Input.initialize(args);
		Loss loss = Input.loss_train.clone_loss(); 
//		Logistic_Loss_efficient.buildHessians(Input.data, Input.d);
		int n = loss.getDataSize();
		int d = loss.getDimension(); 
		loss.set_lambda(1.0/n);
		FirstOrderOpt[] methods_in = new FirstOrderOpt[2]; 
		Adapt_Strategy_Double_Full strategy = new Adapt_Strategy_Double_Full(loss.getDataSize(), 3*d, 1, 1);
		methods_in[1] = new NewtonDataDriven((SecondOrderLoss) loss.clone_loss(),strategy.clone_strategy(),2.0);
		methods_in[0] = new NewtonData6((SecondOrderLoss) loss.clone_loss(),strategy.clone_strategy(),2.0); 
		First_Order_Factory_efficient.methods_in = methods_in; 
		First_Order_Factory_efficient.experiment_with_iterations_complexity(1, loss.clone_loss(), 20,-1.0 , Input.loss_test, Input.config.logDir+"_newton6", Input.L, false,n);
	}
}