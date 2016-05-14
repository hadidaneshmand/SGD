import opt.firstorder.EulerNewton;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.First_Order_Factory_efficient;
import opt.loss.Loss;
import opt.loss.SecondOrderLoss;
import data.DataPoint;
import data.DensePoint_efficient;

public class eulernewton_test {
	public static void main(String[] args) {
		Input.initialize(args);
		Loss loss = Input.loss_train.clone_loss(); 
		int n = loss.getDataSize();
		loss.set_lambda(0.01);
		DataPoint initParam = (DataPoint) DensePoint_efficient.one(Input.loss_train.getDimension()).multiply(0.0); 
		FirstOrderOpt[] methods_in = new FirstOrderOpt[1]; 
		methods_in[0] = new EulerNewton((SecondOrderLoss) loss);
		for(int i=0;i<methods_in.length;i++){
			methods_in[i].setParam(initParam);
		}
		First_Order_Factory_efficient.methods_in = methods_in; 
		First_Order_Factory_efficient.experiment_with_iterations_complexity(1, loss.clone_loss(), 20, 0, Input.loss_test, Input.config.logDir+"_eulernewton", Input.L, false,n);
	}
}
