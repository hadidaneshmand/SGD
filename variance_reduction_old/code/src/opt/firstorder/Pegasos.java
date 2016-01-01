package opt.firstorder;

import opt.loss.Loss;
import data.DataPoint;

public class Pegasos extends FirstOrderOpt {
	private int T;
	private double L; 
	public Pegasos(Loss loss) {
		super(loss);
		setT(0); 
		L = 1; 
	}

	@Override
	public void Iterate(int stepNum) {
		for(int i=0;i<stepNum;i++){ 
			setT(getT() + 1); 
			double eta_t = 0.1/(0.1+loss.getLambda()*getLearning_rate());
			w = (DataPoint) w.multiply(1-eta_t*loss.getLambda()).add(loss.getStochasticGradient(w).multiply(-1*eta_t));
		}
	}

	public int getT() {
		return T;
	}

	public void setT(int t) {
		T = t;
	}

	public double getL() {
		return L;
	}

	public void setL(double l) {
		L = l;
	}

	@Override
	public String getName() {
		return "Pegasos";
	}

	@Override
	public FirstOrderOpt clone_method() {
		Pegasos pout = new Pegasos(loss.clone_loss()); 
		pout.setL(L);
		pout.setT(T);
		pout.setParam(this.cloneParam());
		return pout;
	}

}
