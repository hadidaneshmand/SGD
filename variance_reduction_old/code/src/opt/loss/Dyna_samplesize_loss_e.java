package opt.loss;

import java.util.ArrayList;
import java.util.List;

import org.ejml.simple.SimpleMatrix;

import opt.Adapt_Strategy;
import opt.SampleSizeStrategy;
import opt.utils;
import data.DataPoint;

public class Dyna_samplesize_loss_e implements adaptive_loss, SecondOrderLoss {
	protected Loss loss; 
	protected SampleSizeStrategy as; 
	protected int computed_sgd; 
	protected int computed_gd; 
	protected int computed_hessian; 
	protected int computed_subHessian; 
	
	
	public Dyna_samplesize_loss_e(Loss loss, SampleSizeStrategy as) {
		this.loss = loss; 
		this.as = as; 
	}
	@Override
	public DataPoint getStochasticGradient(int index, DataPoint w) {
		return loss.getStochasticGradient(index,w);
	}
	@Override
	public DataPoint getStochasticGradient(DataPoint w) {
		computed_sgd++;
		int randInd = as.getSubInd().get(utils.getInstance().getGenerator().nextInt(as.getSubsamplesi())); 
		return loss.getStochasticGradient(randInd,w); //TODO
	}
	@Override
	public double computeLoss(DataPoint w) {
		return loss.computeLoss(w);
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
		computed_gd++; 
		computed_sgd+=as.getSubsamplesi(); 
		return loss.getStochasticGradient(as.getSubInd(), w);
	}
	@Override
	public DataPoint getStochasticGradient(List<Integer> indices,
			DataPoint w) {
		return loss.getStochasticGradient(indices, w);
	}
	@Override
	public Loss clone_loss() {
		Dyna_samplesize_loss_e out = new Dyna_samplesize_loss_e(loss.clone_loss(), as.clone_strategy());
		computed_gd = this.computed_gd; 
		computed_hessian = this.computed_hessian; 
		computed_sgd = this.computed_sgd; 
		computed_subHessian = this.computed_subHessian; 
		return out;
	}
	@Override
	public void set_lambda(double lambda) {
		loss.set_lambda(lambda);
	}
	@Override
	public String getType() {
		return "dyna";
	}
	@Override
	public SimpleMatrix getHessian(DataPoint w) {
		computed_hessian++; 
		computed_subHessian+=loss.getDataSize();
		return ((SecondOrderLoss) loss).getHessian(w);
	}

	@Override
	public SimpleMatrix getHessian(DataPoint w, ArrayList<Integer> inds) {
		computed_subHessian++; 
		return ((SecondOrderLoss) loss).getHessian(w,inds);
	}

	@Override
	public SimpleMatrix getHessian_exlusive_regularizer(DataPoint w, int ind) {
		return ((SecondOrderLoss) loss).getHessian_exlusive_regularizer(w, ind);
	}
	@Override
	public double getMaxNorm() {
		return loss.getMaxNorm();
	}
	@Override
	public void tack() {
		as.Tack(); 
	}
	
}
