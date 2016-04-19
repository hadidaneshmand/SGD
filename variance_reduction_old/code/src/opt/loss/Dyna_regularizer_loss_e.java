package opt.loss;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import data.DataPoint;
import opt.SampleSizeStrategy;

public class Dyna_regularizer_loss_e extends Dyna_samplesize_loss_e {
	public Dyna_regularizer_loss_e(Loss loss, SampleSizeStrategy as) {
		super(loss, as);
		set_lambda(1.0/as.getSubsamplesi());
	}
	@Override
	public DataPoint getAverageGradient(DataPoint w) {
		DataPoint out = super.getAverageGradient(w);
		loss.set_lambda(1.0/as.getSubsamplesi());
		return out;
	}
	
	@Override
	public Loss clone_loss() {
		Dyna_regularizer_loss_e out = new Dyna_regularizer_loss_e(loss.clone_loss(), as.clone_strategy());
		return out;
	}
	@Override
	public DataPoint getStochasticGradient(DataPoint w) {
		DataPoint out = super.getStochasticGradient(w);
		loss.set_lambda(1.0/as.getSubsamplesi());
		return out;
	}
	@Override
	public SimpleMatrix getHessian(DataPoint w) {
		as.Tack(); 
		return super.getHessian(w);
	}
	@Override
	public SimpleMatrix getHessian(DataPoint w, ArrayList<Integer> inds) {
		as.Tack();
		return super.getHessian(w, inds);
	}
	@Override
	public String getType() {
		return "dyna-reg";
	}
	

}
