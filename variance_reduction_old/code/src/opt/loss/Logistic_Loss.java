package opt.loss;

import java.util.Iterator;
import java.util.List;

import data.DataPoint;

public class Logistic_Loss extends Loss_static{
   
	public Logistic_Loss(List<DataPoint> data, int dimension) {
		super(data, dimension);
	}
	

	@Override
	public DataPoint getStochasticGradient(int index, DataPoint w) {
		// loss =  log[1 + exp(−ywTx)] + (lambda/2 * ||w||^2)
		DataPoint g = null;
		DataPoint p = data.get(index); 
		double prod = p.scalarProduct(w);
		int y = (int) p.getLabel();
		prod = Math.exp(-1*prod *y);
		g = (DataPoint) p.multiply(-1*y);
		g = (DataPoint) g.multiply(prod/(1+prod));
		g = (DataPoint) g.add(w.multiply(lambda)); // add regularization term lambda*w (cost function is n*lambda*|w|^2) TODO ask about this !!!!
		return g; 
	}

	@Override
	public double getLoss(DataPoint w) {
		double loss = 0;
		for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
			DataPoint p = (DataPoint) iter.next();
			int y = (int) p.getLabel();
			loss += Math.log(1 + Math.exp(-1*y*p.scalarProduct(w))); 
		}
		loss /= data.size();
		loss += w.squaredNorm()*lambda/2;
		return loss;
	}


	@Override
	public Loss clone_loss() {
		Logistic_Loss lg = new Logistic_Loss(data, getDimension()); 
		lg.setLambda(getLambda());
		return lg;
	}

}
