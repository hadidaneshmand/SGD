import data.DataPoint;
import data.DensePoint_efficient;
import opt.Adapt_Strategy_Double_Full;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory_efficient;
import opt.firstorder.LBFGS_external;
import opt.firstorder.Newton;
import opt.loss.Dyna_regularizer_loss_e;
import opt.loss.Dyna_samplesize_loss_e;
import opt.loss.Loss;
import opt.loss.SecondOrderLoss;

public class newton_test{ 
	public static void main(String[] args) {
		Input.initialize(args);
		Input.loss_train.set_lambda(0.0001);
		Loss loss = Input.loss_train.clone_loss(); 
		int n = loss.getDataSize();
		int d = loss.getDimension(); 
		loss.set_lambda(1.0/n);
		DataPoint initParam = (DataPoint) DensePoint_efficient.one(Input.loss_train.getDimension()).multiply(3.0); 
		FirstOrderOpt[] methods_in = new FirstOrderOpt[2]; 
		Dyna_samplesize_loss_e adapt_reg_loss = new Dyna_regularizer_loss_e(loss.clone_loss(), new Adapt_Strategy_Double_Full(loss.getDataSize(), 3*d, 1, 12));
		methods_in[1] = new Newton((SecondOrderLoss) loss.clone_loss()); 
		methods_in[0] = new Newton((SecondOrderLoss) adapt_reg_loss);
		for(int i=0;i<methods_in.length;i++){
			methods_in[i].setParam(initParam);
		}
		First_Order_Factory_efficient.methods_in = methods_in; 
		First_Order_Factory_efficient.experiment_with_iterations_complexity(1, loss.clone_loss(), 15, -1, Input.loss_test, Input.config.logDir+"_newton", Input.L, false,n);
	}
}