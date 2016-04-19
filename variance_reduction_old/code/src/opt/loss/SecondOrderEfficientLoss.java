package opt.loss;



import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import data.DataPoint;

public abstract class SecondOrderEfficientLoss extends FirstOrderEfficient implements SecondOrderLoss {

	public SecondOrderEfficientLoss(DataPoint[] data, int dimension) {
		super(data, dimension);
	}
	@Override
	public SimpleMatrix getHessian(DataPoint w) {
		SimpleMatrix out = new SimpleMatrix(getDimension(),getDimension()); 
		for(int i=0;i<getDataSize();i++){
			out = out.plus(getHessian_exlusive_regularizer(w, i)); 
		}
		out = out.scale(1.0/getDataSize());
	    SimpleMatrix I = SimpleMatrix.identity(getDimension()); 
	    I = I.scale(lambda);
	    out = out.plus(I); 
		return out;
	}
	
	@Override
	public SimpleMatrix getHessian(DataPoint w, ArrayList<Integer> inds) {
		SimpleMatrix out = new SimpleMatrix(getDimension(),getDimension()); 
		for(int i=0;i<inds.size();i++){
			out = out.plus(getHessian_exlusive_regularizer(w, inds.get(i))); 
		}
		out = out.scale(1.0/inds.size());
	    SimpleMatrix I = SimpleMatrix.identity(getDimension()); 
	    I = I.scale(lambda);
	    out = out.plus(I); 
		return out;
	}

	

}
