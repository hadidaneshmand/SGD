import opt.Adapt_Strategy_Double_Full;
import opt.SampleSizeStrategy;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory_efficient;
import opt.firstorder.LBFGS_my;
import opt.firstorder.Newton;
import opt.loss.Dyna_regularizer_loss_e;
import opt.loss.Dyna_samplesize_loss_e;
import opt.loss.Logistic_Loss_efficient;
import opt.loss.Loss;
import opt.loss.SecondOrderLoss;
import data.DataPoint;
import data.DensePoint_efficient;

public class newton_test_time {
	public static void main(String[] args) {
		Input.initialize(args);
		Input.loss_train.set_lambda(0.0001);
		Loss loss = Input.loss_train.clone_loss(); 
		Logistic_Loss_efficient.buildHessians(Input.data, Input.d);
		int n = loss.getDataSize();
		int d = loss.getDimension(); 
		int m = 100; 
		loss.set_lambda(1.0/n);
		DataPoint initParam = (DataPoint) DensePoint_efficient.one(Input.loss_train.getDimension()).multiply(3.0); 
		FirstOrderOpt[] methods_in = new FirstOrderOpt[3]; 
		SampleSizeStrategy strategy = new Adapt_Strategy_Double_Full(loss.getDataSize(), 3*d, 1, 1);
		Dyna_regularizer_loss_e adapt_reg_loss = new Dyna_regularizer_loss_e(loss.clone_loss(), strategy.clone_strategy());
		Dyna_regularizer_loss_e ssreg_loss_for_lbfg = new Dyna_regularizer_loss_e(loss.clone_loss(),strategy.clone_strategy());
		Dyna_samplesize_loss_e adapt_ss_loss = new Dyna_samplesize_loss_e(loss.clone_loss(), strategy);
		methods_in[0] = new Newton((SecondOrderLoss) loss.clone_loss()); 
		methods_in[1] = new Newton((SecondOrderLoss) loss.clone_loss()); 
		methods_in[2] = new Newton((SecondOrderLoss) adapt_reg_loss);
//		methods_in[2] = new LBFGS_my(loss.clone_loss(), m); 
//		methods_in[3] = new LBFGS_my(ssreg_loss_for_lbfg, m); 
		for(int i=0;i<methods_in.length;i++){
			methods_in[i].setParam(initParam);
		}
		First_Order_Factory_efficient.methods_in = methods_in; 
		First_Order_Factory_efficient.experiment_with_iterations_complexity(1, loss.clone_loss(), 60, -1, Input.loss_test, Input.config.logDir+"_newton_thomas", Input.L, false,n);
	}
}
