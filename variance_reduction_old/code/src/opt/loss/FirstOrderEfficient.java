package opt.loss;

import java.util.ArrayList;
import java.util.List;

import opt.utils;
import data.DataPoint;
import data.DensePoint_efficient;

public abstract class FirstOrderEfficient implements Loss{
	private DataPoint[] data;
	protected double lambda;
	private int dimension;
	
	@Override
	public double computeLoss(List<Integer> indices, DataPoint w) {
		double out = 0.0; 
		for(int i=0;i<indices.size();i++){ 
			out += computeLoss(i, w); 
		}
		out/=indices.size(); 
		out += 0.5*lambda*w.squaredNorm(); 
		return out;
	}
	
	public FirstOrderEfficient(DataPoint[] data,int dimension) {
		this.setDimension(dimension); 
		this.setData(data); 
		setLambda(0);
	}
	
	public abstract DataPoint getStochasticGradient(int index,DataPoint w);
	public DataPoint getStochasticGradient(DataPoint w){
		return getStochasticGradient(utils.getInstance().getGenerator().nextInt(getData().length), w);
	}
	public DataPoint getStochasticGradient(List<Integer> indices, DataPoint w){
		DataPoint g = getSumOfGradient(indices, w);
		return (DataPoint) g.multiply(1.0/indices.size()); 
	}
	
	@Override
	public DataPoint getSumOfGradient(List<Integer> indices, DataPoint w) {
		System.out.println();
		DataPoint g = DensePoint_efficient.zero(dimension);
		for(int i=0;i<indices.size();i++){ 
			DataPoint gi = getStochasticGradient(indices.get(i),w);
			g = (DataPoint) g.add(gi);
		}
		return g;
	}
	public DataPoint getAverageGradient(DataPoint w){
		DataPoint g =  DensePoint_efficient.zero(dimension);
		for(int i=0;i<dimension;i++){ 
			g.set(i, 0);
		}
		for (int i=0;i<getData().length;i++) {
			DataPoint gi = getStochasticGradient(i,w);
			g = (DataPoint) g.add(gi);
		}
		g = (DataPoint) g.multiply(1.0 /getData().length);
		return g;
	}
	public List<DataPoint> getAllStochasticGradients(DataPoint w){ 
		ArrayList<DataPoint> gds = new ArrayList<DataPoint>(); 
		for(int i=0;i<getData().length;i++){ 
			gds.add(getStochasticGradient(i, w)); 
		}
		return gds;
	}
	public double computeLoss(DataPoint w){
		double out = 0.0; 
		for(int i=0;i<data.length;i++){
			out+= computeLoss(i, w); 
		}
		out/=data.length; 
		out+=0.5*lambda*w.squaredNorm(); 
		return out; 
	}
	public double getLambda() {
		return lambda;
	}
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	
	public int getDataSize(){ 
		return this.getData().length; 
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
	@Override
	public String getType() {
		return "static";
	}

	public DataPoint[] getData() {
		return data;
	}

	public void setData(DataPoint[] data) {
		this.data = data;
	}
	double LL = -1; 
	@Override
	public double getMaxNorm() {
		if(LL ==-1){ 
			for(int i=0;i<getDataSize();i++){ 
				double dnorm = data[i].getNorm(); 
				if(LL< dnorm){ 
					LL = dnorm; 
				}
			}
		}
		return LL;
	}
	
	
}
