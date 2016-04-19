package opt.firstorder;

import opt.Adapt_Strategy;
import opt.loss.Loss;
import data.DataPoint;
import data.DensePoint;
import data.DensePoint_efficient;

public class SAGA_Adapt_WithoutMemory extends SAGA_Adapt {
	@Override
	public void setName() {
		name =  "AdaptMemoFree"; 
	}
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
			setStepSize(0.3/(L+as.getSubsamplesi()*mu));
			// Compute stochastic gradient for p
			DataPoint gp = getLoss().getStochasticGradient(index, w);
			// Compute SAGA gradient
			DataPoint g = VarianceCorrection( index, gp);
			updateMemory(gp, index);
			// gradient step
			w = (DataPoint) w.subtract(g.multiply(getStepSize()));
							
		}

	}
	public void EraseTheMemory(){ 
		// allocate memory to store one gradient per datapoint
		phi = new DataPoint[getLoss().getDataSize()];
	    nGradients	 = 0; //number of gradients stored so far
		// average gradient
		avg_phi = new DensePoint_efficient(getLoss().getDimension());
		for(int i=0;i<getLoss().getDimension();i++){ 
			avg_phi.set(i, 0.0);
		}
	}
	@Override
	public FirstOrderOpt clone_method() {
		SAGA_Adapt_WithoutMemory out = new SAGA_Adapt_WithoutMemory(getLoss().clone_loss(), as.clone_strategy(),mu,L);
		out.phi = new DensePoint[getLoss().getDataSize()];
		for(int i=0;i<getLoss().getDataSize();i++){
			out.phi[i] = phi[i];
		}
		out.avg_phi = new DensePoint_efficient(getLoss().getDimension());
		for(int i=0;i<getLoss().getDimension();i++){ 
			out.avg_phi.set(i, avg_phi.get(i));
		}
		out.w = clone_w();
		return out;
	}

}
