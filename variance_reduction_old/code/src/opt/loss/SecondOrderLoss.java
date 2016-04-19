package opt.loss;


import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import data.DataPoint;

public interface SecondOrderLoss extends Loss{
	public SimpleMatrix getHessian(DataPoint w); 
	public SimpleMatrix getHessian(DataPoint w, ArrayList<Integer> inds); 
	public SimpleMatrix getHessian_exlusive_regularizer(DataPoint w, int ind); 
}
