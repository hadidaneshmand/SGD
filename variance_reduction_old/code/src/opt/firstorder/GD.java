package opt.firstorder;

import opt.loss.Loss;
import data.DataPoint;

public class GD extends FirstOrderOpt {

	public GD(Loss loss) {
		super(loss);
	}

	@Override
	public void Iterate(int stepNum) {
		for(int i=0;i<stepNum;i++){ 
			    DataPoint g = loss.getAverageGradient(w);
			    num_computed_gradients+=loss.getDataSize();
				w = (DataPoint) w.add(g.multiply(-1*learning_rate));
		}
	}

	@Override
	public String getName() {
		return "GD";
	}

	@Override
	public FirstOrderOpt clone_method() {
		GD newobj= new GD(loss.clone_loss()); 
		newobj.setParam(this.cloneParam());
		newobj.setLearning_rate(this.learning_rate);
		newobj.num_computed_gradients = this.num_computed_gradients; 
		return newobj;
	}


}
