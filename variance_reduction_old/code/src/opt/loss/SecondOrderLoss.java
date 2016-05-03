package opt.loss;


import java.util.ArrayList;

import Jama.Matrix;
import data.DataPoint;

public interface SecondOrderLoss extends Loss{
	public Matrix getHessian(DataPoint w); 
	public Matrix getHessian(DataPoint w, ArrayList<Integer> inds); 
	public Matrix getHessian_exlusive_regularizer(DataPoint w, int ind); 
}
