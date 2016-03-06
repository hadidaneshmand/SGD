package opt.loss;

import java.util.ArrayList;
import java.util.List;

import opt.Adapt_Strategy;
import opt.SampleSizeStrategy;
import data.DataPoint;

public class Adaptss_loss_efficient implements Loss {
	Loss loss; 
	SampleSizeStrategy as; 
	public Adaptss_loss_efficient(Loss loss, SampleSizeStrategy as) {
		this.loss = loss; 
		this.as = as; 
	}
	@Override
	public DataPoint getStochasticGradient(int index, DataPoint w) {
		return loss.getStochasticGradient(index,w);
	}
	@Override
	public DataPoint getStochasticGradient(DataPoint w) {
		return loss.getStochasticGradient(as.Tack(),w); //TODO
	}
	@Override
	public double getLoss(DataPoint w) {
		return loss.getLoss(w);
	}
	
	@Override
	public int getDimension() {
		return loss.getDimension();
	}
	@Override
	public int getDataSize() {
		return as.getSubsamplesi();
	}
	
	@Override
	public double getLambda() {
		return loss.getLambda();
	}
	@Override
	public DataPoint getAverageGradient(DataPoint w) {
		as.Tack(); 
		return loss.getStochasticGradient(as.getSubInd(), w);
	}
	@Override
	public DataPoint getStochasticGradient(List<Integer> indices,
			DataPoint w) {
		return loss.getStochasticGradient(indices, w);
	}
	@Override
	public Loss clone_loss() {
		Adaptss_loss_efficient out = new Adaptss_loss_efficient(loss.clone_loss(), as.clone_strategy());
		return out;
	}
	@Override
	public void set_lambda(double lambda) {
		loss.set_lambda(lambda);
	}

	
}
