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

public class newton_test{ 
	public static void main(String[] args) {
		Input.initialize(args);
		Loss loss = Input.loss_train.clone_loss(); 
//		Logistic_Loss_efficient.buildHessians(Input.data, Input.d);
		int n = loss.getDataSize();
		int d = loss.getDimension(); 
		int m = 100; 
		loss.set_lambda(1.0/n);
		DataPoint initParam = (DataPoint) DensePoint_efficient.one(Input.loss_train.getDimension()).multiply(3.0); 
		FirstOrderOpt[] methods_in = new FirstOrderOpt[8]; 
		SampleSizeStrategy strategy = new Adapt_Strategy_Double_Full(loss.getDataSize(), 3*d, 1, 1);
		Dyna_regularizer_loss_e adapt_reg_loss = new Dyna_regularizer_loss_e(loss.clone_loss(), strategy.clone_strategy());
		Dyna_regularizer_loss_e ssreg_loss_for_lbfg = new Dyna_regularizer_loss_e(loss.clone_loss(),new Adapt_Strategy_Double_Full(loss.getDataSize(), 3*d, 3, 12)); 
		methods_in[1] = new Newton((SecondOrderLoss) loss.clone_loss()); 
		methods_in[0] = new Newton((SecondOrderLoss) adapt_reg_loss);
		methods_in[2] = new LBFGS_my(loss.clone_loss(), m); 
		methods_in[3] = new LBFGS_my(ssreg_loss_for_lbfg, m); 
		methods_in[4] = new SAGA(loss.clone_loss(), 0.3/(Input.L+1));
		methods_in[5] = new NewtonDataDriven((SecondOrderLoss) loss.clone_loss(),strategy,2.0);
		methods_in[6] = new LBFGS_external(ssreg_loss_for_lbfg, m);
		methods_in[7] = new LBFGS_external(loss.clone_loss(), m); 
//		for(int i=0;i<methods_in.length;i++){
//			methods_in[i].setParam(initParam);
//		}
		First_Order_Factory_efficient.methods_in = methods_in; 
		First_Order_Factory_efficient.experiment_with_iterations_complexity(3, loss.clone_loss(), 100, -1.0, Input.loss_test, Input.config.logDir+"_newton_datadriven", Input.L, false,n);
	}
}