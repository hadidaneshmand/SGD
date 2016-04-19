package opt.firstorder;


import data.DataPoint;
import data.DensePoint_efficient;
import opt.loss.Loss;

public class SGD_AVG extends SGD {
	DataPoint w_a; 
	public SGD_AVG(Loss loss) {
		super(loss);
		w_a = DensePoint_efficient.zero(loss.getDimension());
		setConstant_step_size(true);
	}
	
	@Override
	public void setName() {
		name = "avg-sgd";
	}
	
	
	@Override
	public DataPoint getParam() {
		return w_a;
	}
	
	@Override
	public FirstOrderOpt clone_method() {
		SGD_AVG out = new SGD_AVG(loss.clone_loss()); 
		out.name = this.name; 
		out.step_size = step_size; 
		out.num_computed_gradients = this.num_computed_gradients; 
		out.w = clone_w(); 
		out.constant_step_size = constant_step_size; 
		out.eta = eta; 
		out.total_iterations = total_iterations; 
		return out;
	}
	@Override
	public void iterate_once() {
		super.iterate_once();
		w_a = (DataPoint) w_a.multiply((total_iterations-1.0)/(total_iterations)).add(w.multiply(1.0/(total_iterations)));
	}
	

}
