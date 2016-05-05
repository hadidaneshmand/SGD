package opt.firstorder;


import Jama.Matrix;
import data.DataPoint;
import opt.loss.SecondOrderLoss;
import opt.loss.adaptive_loss;

public class Newton extends FirstOrderOpt{
    boolean first_itr = true;
    private double lastLocalNorm;
	public Newton(SecondOrderLoss loss) {
		super(loss);
		setStepSize(0.5);
		lastLocalNorm = Double.MAX_VALUE; 
	}

	@Override
	public void setName() {
		this.name = "newton-"+loss.getType(); 
	}

	int c = 0; 
	int initialSample = -1; 
	boolean firststep = true;
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
		Matrix H_inv = ((SecondOrderLoss)loss).getHessian(w).inverse();
		DataPoint grad = loss.getAverageGradient(w);
		DataPoint delta = grad.times(H_inv); 
		delta = (DataPoint) delta.multiply(-1.0);
		double step_size = 1.0;
		if(loss.getDataSize()<=initialSample){
			step_size = backtracking_line_search(delta); 
		}
//		double  step_size = 1; 
//		double step_size = exact_line_search(delta); 
//		double step_size = 0.01;
		w = (DataPoint) w.add(delta.multiply(step_size));
		double localnorm = grad.scalarProduct(grad.times(H_inv));
		lastLocalNorm = localnorm; 
		if(loss instanceof adaptive_loss){ 
			if(firststep){
				
				if(localnorm<1.0/30){ 
					((adaptive_loss)loss).tack(); 
					firststep = false;
				}
				
			}
			else{
				((adaptive_loss)loss).tack(); 
			}
			
		}
	}

	@Override
	public FirstOrderOpt clone_method() {
		Newton method = new Newton((SecondOrderLoss) loss.clone_loss()); 
		method.w = clone_w(); 
		method.num_computed_gradients = this.num_computed_gradients; 
		method.name = this.name; 
		method.lastLocalNorm = this.lastLocalNorm; 
		return method;
	}

	public double getLastLocalNorm() {
		return lastLocalNorm;
	}

	public void setLastLocalNorm(double lastLocalNorm) {
		this.lastLocalNorm = lastLocalNorm;
	}

}
