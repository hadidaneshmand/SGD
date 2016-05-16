import data.DataPoint;
import data.DensePoint_efficient;
import opt.Adapt_Strategy_Double_Full;
import opt.SampleSizeStrategy;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory_efficient;
import opt.firstorder.LBFGS_external;
import opt.firstorder.LBFGS_my;
import opt.firstorder.Newton;
import opt.firstorder.NewtonDataDriven;
import opt.firstorder.SAGA;
import opt.loss.Dyna_regularizer_loss_e;
import opt.loss.Loss;
import opt.loss.SecondOrderLoss;

public class newton_temp{ 
	public static void main(String[] args) {
		Input.initialize(args);
		Loss loss = Input.loss_train.clone_loss(); 
//		Logistic_Loss_efficient.buildHessians(Input.data, Input.d);
		int n = loss.getDataSize();
		int d = loss.getDimension(); 
		int m = 100; 
		loss.set_lambda(1.0/n);
		FirstOrderOpt[] methods_in = new FirstOrderOpt[1]; 
		SampleSizeStrategy strategy = new Adapt_Strategy_Double_Full(loss.getDataSize(), 3*d, 1, 1);
		Dyna_regularizer_loss_e adapt_reg_loss = new Dyna_regularizer_loss_e(loss.clone_loss(), strategy.clone_strategy());
		Dyna_regularizer_loss_e ssreg_loss_for_lbfg = new Dyna_regularizer_loss_e(loss.clone_loss(),new Adapt_Strategy_Double_Full(loss.getDataSize(), 3*d, 3, 12)); 
		methods_in[0] = new NewtonDataDriven((SecondOrderLoss) loss.clone_loss(),strategy,2.0);
//		for(int i=0;i<methods_in.length;i++){
//			methods_in[i].setParam(initParam);
//		}
		First_Order_Factory_efficient.methods_in = methods_in; 
		First_Order_Factory_efficient.experiment_with_iterations_complexity(3, loss.clone_loss(), 100, -1.0 , Input.loss_test, Input.config.logDir+"_newton_datadriven", Input.L, false,n);
	}
}