package opt.firstorder;



import javax.management.RuntimeErrorException;

import opt.loss.Loss;
import data.DataPoint;

public class SGD extends FirstOrderOpt {

	protected int total_iterations;
	protected boolean constant_step_size; 
	protected double eta = 0.1;
	public SGD(Loss loss) {
		super(loss);
		this.setStepSize(0.9); 
		this.total_iterations = 0;
		setConstant_step_size(false);
	}

	@Override
	public void Iterate(int stepNum) {
		for (int k = 0; k < stepNum; ++k) {
			iterate_once();
		}
	}

	public void iterate_once(){ 
		total_iterations++;
		DataPoint mu = null; 
		mu = getLoss().getStochasticGradient(w);
		if (mu != null) {
			double gamma = 0; 
			if(isConstant_step_size()){ 
				gamma = getStepSize();
			}
			else{ 
				gamma = (1.0*eta)/(eta+getStepSize()*total_iterations);
			}
			w = (DataPoint) w.subtract(mu.multiply(gamma));
		}
		else{
			throw new RuntimeErrorException(new Error("the gradient is null in SGD iterate")); 
		}
	}

	public boolean isConstant_step_size() {
		return constant_step_size;
	}

	public void setConstant_step_size(boolean constant_step_size) {
		this.constant_step_size = constant_step_size;
	}
	
	public String getName() {
		if(isConstant_step_size()){ 
			return name+":"+getStepSize(); 
		}
		return name;
	}
	
	@Override
	public void setName() {
		name = "SGD"; 
	}

	@Override
	public FirstOrderOpt clone_method() {
		SGD sgd = new SGD(getLoss().clone_loss()); 
		sgd.setName(this.name); 
		sgd.constant_step_size = constant_step_size; 
		sgd.step_size = this.step_size; 
		sgd.total_iterations = this.total_iterations; 
		sgd.w = this.clone_w(); 
		return sgd;
	}

	public double getEta() {
		return eta;
	}

	public void setEta(double gamma) {
		this.eta = gamma;
	}

}
