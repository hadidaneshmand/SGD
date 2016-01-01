package opt.loss;

import java.util.Iterator;

import data.DataPoint;

public class LeastSquares_efficient extends Loss_static_efficient {

	public LeastSquares_efficient(DataPoint[] data, int dimension) {
		super(data, dimension);
	}

	@Override
	public Loss clone_loss() {
		LeastSquares_efficient out = new LeastSquares_efficient(data, getDimension()); 
		out.setLambda(lambda);
		return out;
	}

	@Override
	public DataPoint getStochasticGradient(int index, DataPoint w) {
		// use squared loss: (w^T*x - y)^2 + (lambda/2 * ||w||^2)
		DataPoint p = data[index]; 
		DataPoint g = null;
		double y = p.getLabel();
		g = (DataPoint) p.multiply(2 * (w.scalarProduct(p) - y));
		g = (DataPoint) g.add(w.multiply(lambda)); // add regularizer			
		return g;
	}

	@Override
	public double getLoss(DataPoint w) {
		double loss = 0;
		if(data == null) {
			return -1;
		}
		// use squared loss: (1/n) * (w^T*x - y)^2
		for (int i=0;i<data.length;i++) {
			DataPoint p = data[i]; 
			double y = p.getLabel();
			double t = (w.scalarProduct(p) - y);
			loss += t*t;
		}
		loss /= data.length;
		loss += (lambda/2.0)*w.squaredNorm();
		return loss;
	}

}
