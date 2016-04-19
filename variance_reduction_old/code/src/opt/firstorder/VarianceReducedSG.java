package opt.firstorder;


import java.util.Calendar;
import java.util.Random;

import opt.utils;
import opt.loss.Loss;
import data.DataPoint;

public abstract class  VarianceReducedSG extends FirstOrderOpt implements Accelarable{

	public VarianceReducedSG(Loss loss, double learning_rate) {
		super(loss);
		this.setStepSize(learning_rate); 
	}

	@Override
	public void Iterate(int stepNum) {
		int n = getLoss().getDataSize();
		
		for (int i = 0; i < stepNum; ++i) {
			
			// gradient step
			w = (DataPoint) w.subtract(getGradient(w).multiply(getStepSize()));
//			if(i % 20000 == 1){ 
//				System.out.println("VR Iteration: "+i +", Free memory (bytes): " + 
//				  Runtime.getRuntime().freeMemory()+ ", Total memory (bytes): " + 
//						  Runtime.getRuntime().totalMemory());
//			}
		}

	}
	
	public abstract DataPoint VarianceCorrection(int index, DataPoint gp);
	public abstract void updateMemory(DataPoint stochasticGradient,int index);
	
	@Override
	public DataPoint getGradient(DataPoint w) {
		int index = utils.getInstance().getGenerator().nextInt(loss.getDataSize()); 
		// Compute stochastic gradient for p
		DataPoint gp = getLoss().getStochasticGradient(index, w);
		// Compute SAGA gradient
		DataPoint g = VarianceCorrection( index, gp);
		updateMemory(gp, index);
		return g;
	}
	@Override
	public double computationalComplexity() {
		return 1;
	}
	@Override
	public Accelarable clone_accelarable() {
		return (Accelarable) clone_method();
	}
}
