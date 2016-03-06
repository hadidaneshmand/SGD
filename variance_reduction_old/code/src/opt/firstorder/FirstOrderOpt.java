package opt.firstorder;

import opt.loss.Loss;
import data.DataPoint;
import data.DensePoint;

public abstract class FirstOrderOpt {
	protected Loss loss; 
	protected DataPoint w; 
	protected double learning_rate = 0.8; 
	protected double num_computed_gradients = 0;
	public FirstOrderOpt(Loss loss) {
		this.loss = loss;
		w = new DensePoint(loss.getDimension());
		for (int i =0;i<loss.getDimension();i++) {
			w.set(i, 0.0);
		}
	}
	
	public abstract void Iterate(int stepNum);
	public abstract String getName();
	public void setParam(DataPoint w){ 
		this.w = w; 
	}
	public DataPoint getParam(){ 
		return this.w; 
	}
	public double getLearning_rate() {
		return learning_rate;
	}
	public void setLearning_rate(double learning_rate) {
		this.learning_rate = learning_rate;
	}
	public abstract FirstOrderOpt clone_method();
	public DataPoint cloneParam(){ 
		DataPoint w_past = new DensePoint(loss.getDimension());
		for(int i=0;i<loss.getDimension();i++){ 
			w_past.set(i, w.get(i));
		}
		return w_past;
	}

	public double getNum_computed_gradients() {
		return num_computed_gradients;
	}

	public void setNum_computed_gradients(double num_computed_gradients) {
		this.num_computed_gradients = num_computed_gradients;
	}
	
}
