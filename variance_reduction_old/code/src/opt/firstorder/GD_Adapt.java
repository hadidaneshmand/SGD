package opt.firstorder;

import java.util.ArrayList;

import opt.Adapt_Strategy;
import opt.loss.Loss;
import data.DataPoint;

public class GD_Adapt extends GD {
	Adapt_Strategy as; 
	public GD_Adapt(Loss loss, Adapt_Strategy as) {
		super(loss);
		this.as = as;
		setLearning_rate(0.1);
	}
	
	@Override
	public void Iterate(int stepNum) {
	    for(int i=0;i<stepNum;i++){
	    	int pastsi = as.getSubsamplesi(); 
	    	as.Tack(); 
	    	int newsi = as.getSubsamplesi(); 
	    	if(newsi>pastsi){
	    		System.out.println("Yes");
	    		ArrayList<Integer> indices = as.getSubInd();
	    		DataPoint g = loss.getStochasticGradient(indices, w);
	    		System.out.println("size:"+indices.size());
	    		System.out.println("l:"+learning_rate);
	    		w = (DataPoint) w.subtract(g.multiply(learning_rate));
	    	}
	    }
		  

	}
	

}
