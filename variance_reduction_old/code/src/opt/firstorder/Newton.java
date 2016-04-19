package opt.firstorder;


import data.DataPoint;
import opt.loss.SecondOrderLoss;

public class Newton extends FirstOrderOpt{
    boolean first_itr = true;
	public Newton(SecondOrderLoss loss) {
		super(loss);
		setStepSize(0.5);
	}

	@Override
	public void setName() {
		this.name = "newton-"+loss.getType(); 
	}

	@Override
	public void Iterate(int stepNum) {
		for(int i=0;i<stepNum;i++){ 
			num_computed_gradients+=loss.getDataSize(); 
			Iterate_once();
		}
	}
	int c = 0; 
	public void Iterate_once(){ 
//		if(first_itr){ 
//			System.out.println("stepsize"+getStepSize());
//			w = (DataPoint) w.add(loss.getAverageGradient(w).multiply(-1*getStepSize()));
//			c++; 
//			if(c>2){
//				first_itr = false; 
//			}
//			return; 
//		}
		DataPoint delta = loss.getAverageGradient(w).times(((SecondOrderLoss)loss).getHessian(w).invert()); 
		delta = (DataPoint) delta.multiply(-1.0); 
		double step_size = backtracking_line_search(delta); 
//		double  step_size = 1; 
//		double step_size = exact_line_search(delta); 
//		double step_size = 0.01;
		w = (DataPoint) w.add(delta.multiply(step_size));
	}

	@Override
	public FirstOrderOpt clone_method() {
		Newton method = new Newton((SecondOrderLoss) loss.clone_loss()); 
		method.w = clone_w(); 
		method.num_computed_gradients = this.num_computed_gradients; 
		method.name = this.name; 
		return method;
	}

}
