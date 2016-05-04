package opt.firstorder;


import data.DataPoint;
import opt.loss.SecondOrderLoss;
import opt.loss.adaptive_loss;

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

	int c = 0; 
	int initialSample = -1; 
	public void iterate_once(){ 
//		if(first_itr){ 
//			System.out.println("stepsize"+getStepSize());
//			w = (DataPoint) w.add(loss.getAverageGradient(w).multiply(-1*getStepSize()));
//			c++; 
//			if(c>2){
//				first_itr = false; 
//			}
//			return; 
//		}
		if(initialSample == -1){ 
			initialSample = loss.getDataSize(); 
		}
		DataPoint delta = loss.getAverageGradient(w).times(((SecondOrderLoss)loss).getHessian(w).inverse()); 
		delta = (DataPoint) delta.multiply(-1.0);
		double step_size = 1.0;
		if(loss.getDataSize()<=initialSample){
			step_size = backtracking_line_search(delta); 
		}
//		double  step_size = 1; 
//		double step_size = exact_line_search(delta); 
//		double step_size = 0.01;
		w = (DataPoint) w.add(delta.multiply(step_size));
		if(loss instanceof adaptive_loss){ 
			((adaptive_loss)loss).tack(); 
		}
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
