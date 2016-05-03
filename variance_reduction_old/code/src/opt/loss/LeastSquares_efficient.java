package opt.loss;


import Jama.Matrix;
import data.DataPoint;

public class LeastSquares_efficient extends SecondOrderEfficientLoss {

	public LeastSquares_efficient(DataPoint[] data, int dimension) {
		super(data, dimension);
	}

	@Override
	public Loss clone_loss() {
		LeastSquares_efficient out = new LeastSquares_efficient(getData(), getDimension()); 
		out.setLambda(lambda);
		return out;
	}

	@Override
	public DataPoint getStochasticGradient(int index, DataPoint w) {
		// use squared loss: (w^T*x - y)^2 + (lambda/2 * ||w||^2)
		DataPoint p = getData()[index]; 
		double y = p.getLabel();
		DataPoint g = (DataPoint) p.multiply(2 * (w.scalarProduct(p) - y));
//		System.out.println("least_norm:"+w.getNorm());
		g = (DataPoint) g.add(w.multiply(lambda)); // add regularizer		
		
		return g;
	}

	@Override
	public double computeLoss(DataPoint w) {
		double loss = 0;
		if(getData() == null) {
			return -1;
		}
		// use squared loss: (1/n) * (w^T*x - y)^2
		for (int i=0;i<getData().length;i++) {
			DataPoint p = getData()[i]; 
			double y = p.getLabel();
			double t = (w.scalarProduct(p) - y);
			loss += t*t;
		}
		loss /= getData().length;
		loss += (lambda/2.0)*w.squaredNorm();
		return loss;
	}

//	@Override
//	public SimpleMatrix getHessian(DataPoint w) {
//		SimpleMatrix A = new SimpleMatrix(getDataSize(),getDimension()); 
//		for(int i=0;i<getDataSize();i++){
//			for(int j=0;j<getDimension();j++){ 
//				A.set(i, j, getData()[i].get(j));
//			}
//		}
//		SimpleMatrix out = A.transpose().mult(A);
//		out = out.scale(2.0/getDataSize()); 
//		
//		SimpleMatrix I = SimpleMatrix.identity(getDimension());
//		out = out.plus(I.scale(lambda)); 
//		return out;
//	}

	@Override
	public Matrix getHessian_exlusive_regularizer(DataPoint w, int ind) {
		DataPoint p = getData()[ind]; 
		return  p.crossProduct_sm(p, getDimension()).times(2.0);
	}

}
