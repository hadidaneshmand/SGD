package opt.firstorder;

import Jama.Matrix;
import Jama.QRDecomposition;
import opt.loss.Loss;
import opt.loss.SecondOrderLoss;


public class EulerNewton extends FirstOrderOpt {
	double mu; 
	public EulerNewton(Loss loss,double mu) {
		super(loss);
		setStepSize(0.001);
		this.mu = mu; 
	}

	@Override
	public void setName() {
		this.name = "euler-newton"; 
	}

	@Override
	public void iterate_once() {
		// Making Jacoobian Matrix 
		SecondOrderLoss sloss = (SecondOrderLoss) loss; 
		int d  = loss.getDimension(); 
		Matrix hessian = sloss.getHessian(w);
		Matrix wmat = new Matrix(d,1);
		for(int i=0;i<d;i++){
			
			wmat.set(i, 0, w.get(i));
		}
		Matrix A = new Matrix(d, d+1); 
		A.setMatrix(0,d-1,0,d-1, hessian);
		A.setMatrix(0,d-1,d,d, wmat);
		Matrix AT = A.transpose(); 
		QRDecomposition QR = new QRDecomposition(AT); 
		Matrix Q = QR.getQ(); 
		Matrix R = QR.getR(); 
		double sigma = Math.signum(Q.det()*R.det()); 
		System.out.println("sigma:"+sigma);
		// Jacoobian Moor Inverse 
		Matrix Rinv = new Matrix(d+1, d); 
		Rinv.setMatrix(0, d-1, 0, d-1, R.transpose().inverse());
		Matrix A_inv = Q.times(Rinv); 
		// t(a) 
		Matrix ta = Q.getMatrix(0, d, d, d);
		ta = ta.times(sigma); 
		// Euler Newton Updates 
		Matrix u = new Matrix(d+1,d); 
		
		
	}

	@Override
	public FirstOrderOpt clone_method() {
		EulerNewton en = new EulerNewton(loss, mu); 
		en.setParam(this.clone_w());
		en.setNum_computed_gradients(num_computed_gradients);
		en.step_size = en.step_size; 
		en.alpha_line_search = en.alpha_line_search; 
		en.c_1 = en.c_1; 
		en.c_2 = en.c_2; 
		en.max_itr = en.max_itr; 
		en.beta = en.beta; 
		return en;
	}
	

}
