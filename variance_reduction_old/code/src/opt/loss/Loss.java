package opt.loss;

import java.util.List;

import data.DataPoint;

public interface Loss {
	public DataPoint getStochasticGradient(int index,DataPoint w);
	public DataPoint getStochasticGradient(DataPoint w);
	public  double computeLoss(DataPoint w);
	public  double computeLoss(List<Integer> indices,DataPoint w);
	public  double computeLoss(int index,DataPoint w);
	public int getDimension();
	public int getDataSize();
	public double getLambda();
	public DataPoint getAverageGradient(DataPoint w);
	public DataPoint getStochasticGradient(List<Integer> indices, DataPoint w);
	public Loss clone_loss();
	public void set_lambda(double lambda);
	public String getType();
	public double getMaxNorm(); 
}
