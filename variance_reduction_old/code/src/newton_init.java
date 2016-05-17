import opt.Adapt_Strategy_Double_Full;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory_efficient;
import opt.firstorder.Newton;
import opt.firstorder.NewtonDataDriven;
import opt.loss.Dyna_regularizer_loss_e;
import opt.loss.Logistic_Loss_efficient;
import opt.loss.Loss;
import opt.loss.SecondOrderLoss;
import data.DataPoint;
import data.DensePoint_efficient;


public class newton_init {
	public static void main(String[] args) {
		Input.initialize(args);
		Loss loss = Input.loss_train.clone_loss(); 
//		Logistic_Loss_efficient.buildHessians(Input.data, Input.d);
		int n = loss.getDataSize();
		int d = loss.getDimension(); 
		loss.set_lambda(1.0/n); 
		int num_strategy = 3; 
		FirstOrderOpt[] methods_in = new FirstOrderOpt[num_strategy];
		for(int i = 0 ;i < num_strategy ; i++){ 
			methods_in[i] = new Newton((SecondOrderLoss) loss.clone_loss()); 
			methods_in[i].setName("newton:"+i);
			if(i==1){
				methods_in[i].setParam((DataPoint)DensePoint_efficient.one(loss.getDimension()).multiply(3.0));
			}
			if(i==2){
				methods_in[i].setParam((DataPoint)DensePoint_efficient.one(loss.getDimension()).multiply(10.0));
			}
		}
//		Adapt_Strategy_Double_Full strategy = new Adapt_Strategy_Double_Full(loss.getDataSize(), 2*d, 1, 1); 
//		methods_in[num_strategy] = new Newton(new Locality(loss.clone_loss(), strategy, 0.1));
//		methods_in[num_strategy].setName("data-driven");
//		methods_in[num_strategy].setParam(initParam);
		First_Order_Factory_efficient.methods_in = methods_in; 
		First_Order_Factory_efficient.experiment_with_iterations_complexity(4, loss.clone_loss(), 30, -1.0, Input.loss_test, Input.config.logDir+"_newton_init", Input.L, false,n);
		methods_in = new FirstOrderOpt[num_strategy];
		for(int i = 0 ;i < num_strategy ; i++){ 
			methods_in[i] = new NewtonDataDriven((SecondOrderLoss) loss.clone_loss(),new Adapt_Strategy_Double_Full(loss.getDataSize(), 3*d, 1, 1),1); 
			methods_in[i].setName("dyna-newton:"+i);
			if(i==1){
				methods_in[i].setParam((DataPoint)DensePoint_efficient.one(loss.getDimension()).multiply(3.0));
			}
			if(i==2){
				methods_in[i].setParam((DataPoint)DensePoint_efficient.one(loss.getDimension()).multiply(10.0));
			}
		}
		First_Order_Factory_efficient.methods_in = methods_in;
		First_Order_Factory_efficient.experiment_with_iterations_complexity(4, loss.clone_loss(), 30, -1.0, Input.loss_test, Input.config.logDir+"_newtondd_init", Input.L, false,n);
	}
}
