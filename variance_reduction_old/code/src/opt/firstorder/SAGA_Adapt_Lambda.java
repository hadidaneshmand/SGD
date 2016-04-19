package opt.firstorder;

import data.DataPoint;
import opt.Adapt_Strategy;
import opt.loss.Loss;

public class SAGA_Adapt_Lambda extends SAGA_Adapt {

	public SAGA_Adapt_Lambda(Loss loss, Adapt_Strategy as, double mu, double L) {
		super(loss, as, mu, L);
		
	}
	public void Iterate(int stepNum) {
		for (int i = 0; i < stepNum; ++i) {
			int index = as.Tack();
//			System.out.println("samplesize:"+as.getSubsamplesi()+",index:"+index);
			getLoss().set_lambda(1.0/as.getSubsamplesi());
			setStepSize(0.3/(L+1));
			
			// Compute stochastic gradient for p
			DataPoint gp = getLoss().getStochasticGradient(index, w);
			// Compute SAGA gradient
			DataPoint g = VarianceCorrection( index, gp);
			updateMemory(gp, index);
			// gradient step
			w = (DataPoint) w.subtract(g.multiply(getStepSize()));
							
		}

	}
	
	@Override
	public void setName() {
		name = "AdaptLambda"; 
	}
	
	
	

}
