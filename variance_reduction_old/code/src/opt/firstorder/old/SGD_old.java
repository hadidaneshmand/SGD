package opt.firstorder.old;

import java.util.List;
import java.util.Random;

import opt.utils;
import opt.config.Config;
import data.DataPoint;

public class SGD_old extends GradientDescent {
	int total_iterations; 
	
	public SGD_old(List<DataPoint> data, Config conf, double lambda) {
		super(data, conf, lambda);
	    this.eta = 1; 
	    this.total_iterations = 0; 
	}
	
	@Override
	public void optimize(int iteration_num) {
		Random generator = utils.getInstance().getGenerator();
		for (int k = 0; k < iteration_num; ++k) {
			total_iterations++;
			DataPoint p = data.get(generator.nextInt(data.size()));
			DataPoint mu = computeStochasticGradient(p);
			if (mu != null) {
				w = (DataPoint) w.subtract(mu.multiply(eta/total_iterations));
			}
			
		}
		
	}
}
