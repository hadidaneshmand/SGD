package opt.firstorder;



import javax.management.RuntimeErrorException;

import opt.loss.Loss;
import data.DataPoint;

public class SGD extends FirstOrderOpt {

	private int total_iterations;
	private boolean constant_step_size; 
	private double eta = 0.1;
	public SGD(Loss loss) {
		super(loss);
		this.setLearning_rate(0.9); 
		this.total_iterations = 0;
		setConstant_step_size(false);
	}

	@Override
	public void Iterate(int stepNum) {
		for (int k = 0; k < stepNum; ++k) {
			total_iterations++;
			DataPoint mu = null; 
			mu = loss.getStochasticGradient(w);
			if (mu != null) {
				double gamma = 0; 
				if(isConstant_step_size()){ 
					gamma = getLearning_rate();
				}
				else{ 
					gamma = (1.0*eta)/(eta+getLearning_rate()*total_iterations);
				}
				w = (DataPoint) w.subtract(mu.multiply(gamma));
			}
			else{
				throw new RuntimeErrorException(new Error("the gradient is null in SGD iterate")); 
			}
			
		}

	}


	public boolean isConstant_step_size() {
		return constant_step_size;
	}

	public void setConstant_step_size(boolean constant_step_size) {
		this.constant_step_size = constant_step_size;
	}

	@Override
	public String getName() {
		if(isConstant_step_size()){
		return "SGD:"+getLearning_rate();
		}
		if(eta!=0.1){
			return "SGD:"+getEta();
		}
		return "SGD";
	}

	@Override
	public FirstOrderOpt clone_method() {
		SGD sgd = new SGD(loss.clone_loss()); 
		sgd.constant_step_size = constant_step_size; 
		sgd.learning_rate = this.learning_rate; 
		sgd.total_iterations = this.total_iterations; 
		sgd.w = this.cloneParam(); 
		return sgd;
	}

	public double getEta() {
		return eta;
	}

	public void setEta(double gamma) {
		this.eta = gamma;
	}

}
