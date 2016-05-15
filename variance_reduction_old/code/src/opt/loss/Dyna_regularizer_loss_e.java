package opt.loss;

import data.DataPoint;
import opt.SampleSizeStrategy;

public class Dyna_regularizer_loss_e extends Dyna_samplesize_loss_e {
	private double R = 1.0; 
	public Dyna_regularizer_loss_e(Loss loss, SampleSizeStrategy as) {
		super(loss, as);
		set_lambda(R/as.getSubsamplesi());
	}
	
	@Override
	public Loss clone_loss() {
		Dyna_regularizer_loss_e out = new Dyna_regularizer_loss_e(loss.clone_loss(), as.clone_strategy());
		out.setR(R);
		return out;
	}
	
	@Override
	public void tack(DataPoint w) {
		super.tack(w);
		loss.set_lambda(R/as.getSubsamplesi());
		System.out.println("reg");
	}
	

	@Override
	public String getType() {
		return "dyna-reg";
	}

	public double getR() {
		return R;
	}

	public void setR(double r) {
		R = r;
		set_lambda(R/as.getSubsamplesi());
	}
	
	

}
