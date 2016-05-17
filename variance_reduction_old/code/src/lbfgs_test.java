import opt.Adapt_Strategy_Double_Full;
import opt.SampleSizeStrategy;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory_efficient;
import opt.firstorder.LBFGS_my;
import opt.loss.Dyna_regularizer_loss_e;
import opt.loss.Loss;




public class lbfgs_test {
	public static void main(String[] args) {
		Input.initialize(args);
		Loss loss = Input.loss_train.clone_loss(); 
		int n = loss.getDataSize();
		int d = loss.getDimension(); 
		int m = 100; 
		loss.set_lambda(1.0/n);
		FirstOrderOpt[] methods_in = new FirstOrderOpt[2]; 
		SampleSizeStrategy strategy = new Adapt_Strategy_Double_Full(loss.getDataSize(), 400, 3, 10);
		Dyna_regularizer_loss_e ssreg_loss_for_lbfg = new Dyna_regularizer_loss_e(loss.clone_loss(),strategy); 
		methods_in[1] = new LBFGS_my(loss.clone_loss(), m); 
		methods_in[0] = new LBFGS_my(ssreg_loss_for_lbfg, m); 
		First_Order_Factory_efficient.methods_in = methods_in; 
		First_Order_Factory_efficient.saga_for_opt = true; 
		First_Order_Factory_efficient.experiment_with_iterations_complexity(3, loss.clone_loss(), 15, -1.0, Input.loss_test, Input.config.logDir+"_lbfgs", Input.L, false,n);
	}
}
