package opt.loss;

import java.util.ArrayList;

import data.DataPoint;

public interface Loss {
	public DataPoint getStochasticGradient(int index,DataPoint w);
	public DataPoint getStochasticGradient(DataPoint w);
	public abstract double getLoss(DataPoint w);
	public int getDimension();
	public int getDataSize();
	public double getLambda();
	public DataPoint getAverageGradient(DataPoint w);
	public DataPoint getStochasticGradient(ArrayList<Integer> indices, DataPoint w);
	public Loss clone_loss();
	public void set_lambda(double lambda);
}
