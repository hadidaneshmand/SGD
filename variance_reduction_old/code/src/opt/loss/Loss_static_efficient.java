package opt.loss;

import java.util.ArrayList;
import java.util.List;

import opt.utils;
import data.DataPoint;
import data.DensePoint;

public abstract class Loss_static_efficient implements Loss {
	protected DataPoint[] data;
	protected double lambda;
	private int dimension;
	
	
	public Loss_static_efficient(DataPoint[] data,int dimension) {
		this.setDimension(dimension); 
		this.data = data; 
		setLambda(0);
	}
	
	public abstract DataPoint getStochasticGradient(int index,DataPoint w);
	public DataPoint getStochasticGradient(DataPoint w){
		return getStochasticGradient(utils.getInstance().getGenerator().nextInt(data.length), w);
	}
	public DataPoint getStochasticGradient(ArrayList<Integer> indices, DataPoint w){
		DataPoint g = new DensePoint(getDimension());
		for(int i=0;i<dimension;i++){ 
			g.set(i, 0);
		}
		for(int i=0;i<indices.size();i++){ 
			DataPoint gi = getStochasticGradient(indices.get(i),w);
			g = (DataPoint) g.add(gi);
		}
		return (DataPoint) g.multiply(1.0/indices.size()); 
	}
	public DataPoint getAverageGradient(DataPoint w){
		DataPoint g = new DensePoint(getDimension());
		for(int i=0;i<dimension;i++){ 
			g.set(i, 0);
		}
		for (int i=0;i<data.length;i++) {
			DataPoint gi = getStochasticGradient(i,w);
			g = (DataPoint) g.add(gi);
		}
		g = (DataPoint) g.multiply(1.0 /data.length);
		return g;
	}
	public List<DataPoint> getAllStochasticGradients(DataPoint w){ 
		ArrayList<DataPoint> gds = new ArrayList<DataPoint>(); 
		for(int i=0;i<data.length;i++){ 
			gds.add(getStochasticGradient(i, w)); 
		}
		return gds;
	}
	public abstract double getLoss(DataPoint w);
	public double getLambda() {
		return lambda;
	}
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	
	public int getDataSize(){ 
		return this.data.length; 
	}

	public int getDimension() {
		return dimension;
	}

	public void setDimension(int dimension) {
		this.dimension = dimension;
	}
	
	@Override
	public void set_lambda(double lambda) {
		this.lambda = lambda;
	}
	
}
