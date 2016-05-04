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
	public void one_pass() {
		Iterate(loss.getDataSize());
	}
	@Override
	public void iterate_once() {
		setT(getT() + 1); 
		double eta_t = 0.1/(0.1+getLoss().getLambda()*getStepSize());
		w = (DataPoint) w.multiply(1-eta_t*getLoss().getLambda()).add(getLoss().getStochasticGradient(w).multiply(-1*eta_t));
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
	public void setName() {
		name = "pegasos"; 
	}

	@Override
	public FirstOrderOpt clone_method() {
		Pegasos pout = new Pegasos(getLoss().clone_loss()); 
		pout.setL(L);
		pout.setT(T);
		pout.setParam(this.clone_w());
		return pout;
	}

}
