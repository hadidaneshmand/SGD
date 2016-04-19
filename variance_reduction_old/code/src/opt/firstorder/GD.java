package opt.firstorder;

import opt.loss.Loss;
import data.DataPoint;

public class GD extends FirstOrderOpt implements Accelarable{

	public GD(Loss loss) {
		super(loss);
	}

	@Override
	public void Iterate(int stepNum) {
		for(int i=0;i<stepNum;i++){ 
			    DataPoint g = getLoss().getAverageGradient(w);
			    num_computed_gradients+=getLoss().getDataSize();
				w = (DataPoint) w.add(g.multiply(-1*step_size));
		}
	}

	

	@Override
	public FirstOrderOpt clone_method() {
		GD newobj= new GD(getLoss().clone_loss()); 
		newobj.setParam(this.clone_w());
		newobj.setStepSize(this.step_size);
		newobj.num_computed_gradients = this.num_computed_gradients; 
		return newobj;
	}

	@Override
	public void setName() {
		System.out.println("losstype"+getLoss().getType());
		name = getLoss().getType()+"-gd";
	}

	@Override
	public DataPoint getGradient(DataPoint wi) {
		return loss.getAverageGradient(wi);
	}
	

	@Override
	public double computationalComplexity() {
		return loss.getDataSize();
	}

	@Override
	public Accelarable clone_accelarable() {
		return (Accelarable) clone_method();
	}


}
