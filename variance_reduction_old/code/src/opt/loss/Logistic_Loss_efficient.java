package opt.loss;


import org.ejml.simple.SimpleMatrix;

import data.DataPoint;
import data.Matrix;

public class Logistic_Loss_efficient extends SecondOrderEfficientLoss {
	public Logistic_Loss_efficient(DataPoint[] data, int dimension) {
		super(data, dimension);
	}

	@Override
	public Loss clone_loss() {
		Logistic_Loss_efficient out = new Logistic_Loss_efficient(getData(), getDimension());
		out.setLambda(lambda);
		return out;
	}

	@Override
	public DataPoint getStochasticGradient(int index, DataPoint w) {
		// loss =  log[1 + exp(−ywTx)] + (lambda/2 * ||w||^2)
		DataPoint g = null;
		DataPoint p = getData()[index]; 
		double prod = p.scalarProduct(w);
		int y = (int) p.getLabel();
		prod = Math.exp(-1*prod *y);
		g = (DataPoint) p.multiply(-1*y);
		g = (DataPoint) g.multiply(prod/(1+prod));
		g = (DataPoint) g.add(w.multiply(lambda)); // add regularization term lambda*w (cost function is n*lambda*|w|^2) TODO ask about this !!!!
		return g; 	
	}

	@Override
	public double computeLoss(DataPoint w) {
		double loss = 0;
		for (int i=0;i<getData().length;i++) {
			DataPoint p = getData()[i];
			int y = (int) p.getLabel();
			loss += Math.log(1 + Math.exp(-1*y*p.scalarProduct(w)))/getDataSize(); 
			
		}
//		loss /= getData().length;
		loss += w.squaredNorm()*lambda/2;
		return loss;
	}

//	@Override
//	public SimpleMatrix getHessian(DataPoint w) {
//		SimpleMatrix out = new SimpleMatrix(getDimension(),getDimension()); 
//	
//		for(int i=0;i<getDataSize();i++){ 
//			DataPoint di = getData()[i];
//			SimpleMatrix hi = (di).crossProduct_sm(di, getDimension()); 
//			double prod = -1*di.scalarProduct(w)*di.getLabel(); 
//			double g = Math.exp(prod)/(Math.pow(1+Math.exp(prod),2)); 
//			if(i == 0){
//				System.out.println("g:"+g);
//			}
//			hi = hi.scale(g);
//			out = out.plus(hi); 
//		}
//		out = out.scale(1.0/getDataSize());
//	    SimpleMatrix I = SimpleMatrix.identity(getDimension()); 
//	    I = I.scale(lambda);
//	    System.out.println("I[2,2]="+I.get(2, 2));
//	    out = out.plus(I); 
//	    System.out.println("H[2,2]="+out.get(2, 2));
//		
//		return out;
//	}

	@Override
	public SimpleMatrix getHessian_exlusive_regularizer(DataPoint w, int ind) {
		DataPoint di = getData()[ind];
		SimpleMatrix hi = (di).crossProduct_sm(di, getDimension()); 
		double prod = -1*di.scalarProduct(w)*di.getLabel(); 
		double g = Math.exp(prod)/(Math.pow(1+Math.exp(prod),2)); 
		hi = hi.scale(g);
		return hi;
	}

}
