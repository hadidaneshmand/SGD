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
				w = (DataPoint) w.add(g.multiply(-1*learning_rate));
		}
	}

	@Override
	public String getName() {
		return "Gradient Descent";
	}

	@Override
	public FirstOrderOpt clone_method() {
		GD newobj= new GD(loss.clone_loss()); 
		newobj.setParam(this.cloneParam());
		return newobj;
	}


}
