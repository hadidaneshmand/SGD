package opt.firstorder;


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
		Random random = utils.getGenerator();
		for (int i = 0; i < stepNum; ++i) {
			int index = random.nextInt(n); 
			// Compute stochastic gradient for p
			DataPoint gp = loss.getStochasticGradient(index, w);
			// Compute SAGA gradient
			DataPoint g = VarianceCorrection( index, gp);
			updateMemory(gp, index);
			// gradient step
			w = (DataPoint) w.subtract(g.multiply(getLearning_rate()));
							
		}

	}
	
	public abstract DataPoint VarianceCorrection(int index, DataPoint gp);
	public abstract void updateMemory(DataPoint stochasticGradient,int index);
}
