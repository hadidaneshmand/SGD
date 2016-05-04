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
	public void iterate_once() {
		w = (DataPoint) w.subtract(getGradient(w).multiply(getStepSize()));
	}
	@Override
	public void one_pass() {
		Iterate(loss.getDataSize());;
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
	@Override
	public void update_iterations(double time) {
		this.time += time; 
		num_computed_gradients+=1; 
	}
}
