package opt.loss;

import java.util.Iterator;
import java.util.List;

import data.DataPoint;

public class LogisticRegression extends Logistic_Loss{

	public LogisticRegression(List<DataPoint> data, int dimension) {
		super(data, dimension);
	}

	@Override
	public DataPoint getStochasticGradient(int index,DataPoint w) {
		// loss =  log[1 + exp(−ywTx)] + (lambda/2 * ||w||^2)
		DataPoint g = null;
		DataPoint p = getData().get(index);
		double prod = p.scalarProduct(w);
		int y = (int) p.getLabel();
		prod = Math.exp(-1*prod *y);
		g = (DataPoint) p.multiply(-y);
		g = (DataPoint) g.multiply(prod/(1+prod));
		g = (DataPoint) g.add(w.multiply(lambda)); // add regularization term lambda*w (cost function is n*lambda*|w|^2)TODO ask about this !!!!
		return g;
				
	}

	@Override
	public double computeLoss(DataPoint w) {
		double loss = 0;
		if(getData() == null) {
			return -1;
		}
		// loss =  \sum_x log[1 + exp(−ywTx)]/n + (lambda/2 * ||w||^2)
		for (Iterator<DataPoint> iter = getData().iterator(); iter.hasNext();) {
			DataPoint p = (DataPoint) iter.next();
			int y = (int) p.getLabel();
			loss += Math.log(1 + Math.exp(-1*y*p.scalarProduct(w))); 
		}
		loss /= getData().size();
		loss += Math.pow(w.getNorm(),2)*lambda/2;
		return loss; 
	}

	

}
