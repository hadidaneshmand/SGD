package opt.firstorder;

import java.util.ArrayList;

import opt.Adapt_Strategy;
import opt.SampleSizeStrategy;
import opt.loss.Loss;
import data.DataPoint;

public class GD_Adapt extends GD {
	SampleSizeStrategy as; 
	public GD_Adapt(Loss loss, SampleSizeStrategy as) {
		super(loss);
		this.as = as;
		setLearning_rate(0.1);
	}
	
	@Override
	public void Iterate(int stepNum) {
	    for(int i=0;i<stepNum;i++){
	    	    as.Tack(); 
	    		ArrayList<Integer> indices = (ArrayList<Integer>) as.getSubInd();
	    		num_computed_gradients+=indices.size();
	    		DataPoint g = loss.getStochasticGradient(indices, w);
//	    		System.out.println("size:"+indices.size());
	    		w = (DataPoint) w.subtract(g.multiply(learning_rate));
	    }
		  

	}
	@Override
	public FirstOrderOpt clone_method() {
		GD_Adapt out = new GD_Adapt(loss.clone_loss(), as.clone_strategy());
		out.setParam(this.cloneParam());
		out.setLearning_rate(this.learning_rate);
		out.num_computed_gradients = this.num_computed_gradients; 
		return out;
	}
	@Override
	public String getName() {
		return "dyna-gd";
	}
	

}
