package opt.firstorder;


import java.util.Calendar;
import java.util.Random;

import opt.utils;
import opt.loss.Loss;
import data.DataPoint;

public abstract class  VarianceReducedSG extends FirstOrderOpt {

	public VarianceReducedSG(Loss loss, double learning_rate) {
		super(loss);
		this.setLearning_rate(learning_rate); 
	}

	@Override
	public void Iterate(int stepNum) {
		int n = loss.getDataSize();
		
		for (int i = 0; i < stepNum; ++i) {
			int index = utils.getInstance().getGenerator().nextInt(n); 
			// Compute stochastic gradient for p
			DataPoint gp = loss.getStochasticGradient(index, w);
			// Compute SAGA gradient
			DataPoint g = VarianceCorrection( index, gp);
			updateMemory(gp, index);
			// gradient step
			w = (DataPoint) w.subtract(g.multiply(getLearning_rate()));
//			if(i % 5000 == 1){ 
//				System.out.println("VR Iteration: "+i +", Free memory (bytes): " + 
//				  Runtime.getRuntime().freeMemory()+ ", Total memory (bytes): " + 
//						  Runtime.getRuntime().totalMemory());
//			}
		}

	}
	
	public abstract DataPoint VarianceCorrection(int index, DataPoint gp);
	public abstract void updateMemory(DataPoint stochasticGradient,int index);
}
