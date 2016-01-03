package opt.loss;

import java.util.ArrayList;
import java.util.List;

import opt.utils;
import data.DataPoint;
import data.DensePoint;

public abstract class Loss_static implements Loss {
	protected List<DataPoint> data;
	protected double lambda;
	private int dimension;
	
	public Loss_static(List<DataPoint> data,int dimension) {
		this.setDimension(dimension); 
		this.setData(data);
		setLambda(0);
	}
	
	public abstract DataPoint getStochasticGradient(int index,DataPoint w);
	public DataPoint getStochasticGradient(DataPoint w){
		return getStochasticGradient(utils.getInstance().getGenerator().nextInt(data.size()), w);
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
		for (int i=0;i<getData().size();i++) {
			DataPoint gi = getStochasticGradient(i,w);
			g = (DataPoint) g.add(gi);
		}
		g = (DataPoint) g.multiply(1.0 / getData().size());
		return g;
	}
	public List<DataPoint> getAllStochasticGradients(DataPoint w){ 
		ArrayList<DataPoint> gds = new ArrayList<DataPoint>(); 
		for(int i=0;i<data.size();i++){ 
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

	public List<DataPoint> getData() {
		return data;
	}

	public void setData(List<DataPoint> data) {
		this.data = data;
	} 
	public int getDataSize(){ 
		return this.data.size(); 
	}

	public int getDimension() {
		return dimension;
	}

	public void setDimension(int dimension) {
		this.dimension = dimension;
	}
	
}
