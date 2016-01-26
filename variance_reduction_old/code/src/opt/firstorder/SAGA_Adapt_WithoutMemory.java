package opt.firstorder;

import opt.Adapt_Strategy;
import opt.loss.Loss;
import data.DataPoint;
import data.DensePoint;
import data.DensePoint_efficient;

public class SAGA_Adapt_WithoutMemory extends SAGA_Adapt {
	
	public String getName() {return "AdaptMemoFree";};
	public SAGA_Adapt_WithoutMemory(Loss loss, Adapt_Strategy as,double mu, double L) {
		super(loss, as,mu,L);
	}
	@Override
	public void Iterate(int stepNum) {
		for (int i = 0; i < stepNum; ++i) {
			int pastSize = as.getSubsamplesi(); 
			int index = as.Tack();
			int newsize = as.getSubsamplesi(); 
			if(newsize>pastSize){ 
				EraseTheMemory();
			}
			setLearning_rate(0.3/(L+as.getSubsamplesi()*mu));
			// Compute stochastic gradient for p
			DataPoint gp = loss.getStochasticGradient(index, w);
			// Compute SAGA gradient
			DataPoint g = VarianceCorrection( index, gp);
			updateMemory(gp, index);
			// gradient step
			w = (DataPoint) w.subtract(g.multiply(getLearning_rate()));
							
		}

	}
	public void EraseTheMemory(){ 
		// allocate memory to store one gradient per datapoint
		phi = new DataPoint[loss.getDataSize()];
	    nGradients	 = 0; //number of gradients stored so far
		// average gradient
		avg_phi = new DensePoint_efficient(loss.getDimension());
		for(int i=0;i<loss.getDimension();i++){ 
			avg_phi.set(i, 0.0);
		}
	}
	@Override
	public FirstOrderOpt clone_method() {
		SAGA_Adapt_WithoutMemory out = new SAGA_Adapt_WithoutMemory(loss.clone_loss(), as.clone_strategy(),mu,L);
		out.phi = new DensePoint[loss.getDataSize()];
		for(int i=0;i<loss.getDataSize();i++){
			out.phi[i] = phi[i];
		}
		out.avg_phi = new DensePoint_efficient(loss.getDimension());
		for(int i=0;i<loss.getDimension();i++){ 
			out.avg_phi.set(i, avg_phi.get(i));
		}
		out.w = cloneParam();
		return out;
	}

}
