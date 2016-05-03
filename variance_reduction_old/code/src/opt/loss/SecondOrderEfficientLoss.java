package opt.loss;



import java.util.ArrayList;
import Jama.Matrix;
import data.DataPoint;

public abstract class SecondOrderEfficientLoss extends FirstOrderEfficient implements SecondOrderLoss {

	public SecondOrderEfficientLoss(DataPoint[] data, int dimension) {
		super(data, dimension);
	}
	@Override
	public Matrix getHessian(DataPoint w) {
		Matrix out = new Matrix(getDimension(),getDimension()); 
		for(int i=0;i<getDataSize();i++){
			out = out.plus(getHessian_exlusive_regularizer(w, i)); 
		}
		out = out.times(1.0/getDataSize());
	    Matrix I = Matrix.identity(getDimension(),getDimension()); 
	    I = I.times(lambda);
	    out = out.plus(I); 
		return out;
	}
	
	@Override
	public Matrix getHessian(DataPoint w, ArrayList<Integer> inds) {
		Matrix out = new Matrix(getDimension(),getDimension()); 
		for(int i=0;i<inds.size();i++){
			out = out.plus(getHessian_exlusive_regularizer(w, inds.get(i))); 
		}
		out = out.times(1.0/inds.size());
	    Matrix I = Matrix.identity(getDimension(),getDimension()); 
	    I = I.times(lambda);
	    out = out.plus(I); 
		return out;
	}

	

}
