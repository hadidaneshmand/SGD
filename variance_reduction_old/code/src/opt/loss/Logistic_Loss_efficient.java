package opt.loss;


import data.DataPoint;

public class Logistic_Loss_efficient extends Loss_static_efficient {

	public Logistic_Loss_efficient(DataPoint[] data, int dimension) {
		super(data, dimension);
	}

	@Override
	public Loss clone_loss() {
		Logistic_Loss_efficient out = new Logistic_Loss_efficient(data, getDimension());
		out.setLambda(lambda);
		return null;
	}

	@Override
	public DataPoint getStochasticGradient(int index, DataPoint w) {
		// loss =  log[1 + exp(−ywTx)] + (lambda/2 * ||w||^2)
		DataPoint g = null;
		DataPoint p = data[index]; 
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
		for (int i=0;i<data.length;i++) {
			DataPoint p = data[i];
			int y = (int) p.getLabel();
			loss += Math.log(1 + Math.exp(-1*y*p.scalarProduct(w))); 
		}
		loss /= data.length;
		loss += w.squaredNorm()*lambda/2;
		return loss;
	}

}
