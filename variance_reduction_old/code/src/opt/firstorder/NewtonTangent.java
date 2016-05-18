package opt.firstorder;

import Jama.Matrix;
import data.DataPoint;
import opt.loss.SecondOrderLoss;
import opt.loss.adaptive_loss;

public class NewtonTangent extends Newton{
	private double tangent_stepsize; 
	public NewtonTangent(SecondOrderLoss loss) {
		super(loss);
		tangent_stepsize = 0.05;
	}
	
	@Override
	public void iterate_once() {
		if(initialSample == -1){ 
			initialSample = loss.getDataSize(); 
		}
		Matrix H_inv = ((SecondOrderLoss)loss).getHessian(w).inverse();
		DataPoint grad = loss.getAverageGradient(w);
		
		
		double localnorm = grad.scalarProduct(grad.times(H_inv));
		DataPoint delta = grad.times(H_inv); 
		delta = (DataPoint) delta.multiply(-1.0);
		double step_size = 1.0;
		if(localnorm>0.06 && initialSample == loss.getDataSize()){
			step_size = 1.0/(1.0+Math.sqrt(localnorm));
		}
			
		System.out.println("step size:"+step_size);
		w = (DataPoint) w.add(delta.multiply(step_size));
		lastLocalNorm = localnorm; 
		System.out.println("local norm:"+lastLocalNorm);
		if(loss instanceof adaptive_loss){ 
			if(firststep){
				if(localnorm<0.06){ 
					((adaptive_loss)loss).tack(w); 
					firststep = false;
				}
				
			}
			else{
				// predictor step :
				DataPoint delta_tangent = w.times(H_inv);
				w = (DataPoint) w.add(delta_tangent.multiply(-1*tangent_stepsize)); 
				((adaptive_loss)loss).tack(w);
			
			}
			
		}
	}

	public double getTangent_stepsize() {
		return tangent_stepsize;
	}

	public void setTangent_stepsize(double tangent_stepsize) {
		this.tangent_stepsize = tangent_stepsize;
	}
	
	@Override
	public void setName() {
		this.name = "euler-newton";
	}

}
