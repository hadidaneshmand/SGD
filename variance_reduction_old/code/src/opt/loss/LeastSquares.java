package opt.loss;

import java.util.Iterator;
import java.util.List;

import data.DataPoint;

public class LeastSquares extends Loss_static{

	public LeastSquares(List<DataPoint> data, int dimension) {
		super(data, dimension);
	}

	@Override
	public DataPoint getStochasticGradient(int index, DataPoint w) {
		// use squared loss: (w^T*x - y)^2 + (lambda/2 * ||w||^2)
		DataPoint p = getData().get(index); 
		DataPoint g = null;
		double y = p.getLabel();
		g = (DataPoint) p.multiply(2 * (w.scalarProduct(p) - y));
		g = (DataPoint) g.add(w.multiply(lambda)); // add regularizer			
		return g;
	}

	
	public Loss clone_loss(){ 
		LeastSquares out = new LeastSquares(data, getDimension());
		out.setLambda(getLambda());
		return out;
	}

	@Override
	public double computeLoss(int index, DataPoint w) {
		if(getData() == null) {
			return -1;
		}
		DataPoint p = data.get(index);
		double y = p.getLabel();
		double t = (w.scalarProduct(p) - y);
		return t*t;
	}

	

}
