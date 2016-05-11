package opt.firstorder;

import data.DataPoint;
import data.DensePoint_efficient;
import Jama.Matrix;
import Jama.QRDecomposition;
import opt.loss.Loss;
import opt.loss.SecondOrderLoss;


public class EulerNewton extends Newton {
	public EulerNewton(SecondOrderLoss loss) {
		super(loss);
		setStepSize(0.001);
		loss.set_lambda(0.01);
	}

	@Override
	public void setName() {
		this.name = "euler-newton"; 
	}

	@Override
	public void iterate_once() {
		if(getLastLocalNorm()>0.00000001){ 
			super.iterate_once() ;
			return;
		}
		// Making Jacoobian Matrix 
		SecondOrderLoss sloss = (SecondOrderLoss) loss; 
		int d  = loss.getDimension(); 
		Matrix hessian = sloss.getHessian(w);
		System.out.println("stepsize:"+step_size);
		Matrix A = new Matrix(d, d+1); 
		A.setMatrix(0,d-1,0,d-1, hessian);
		System.out.println("set Matrix test:A:"+A.get(4, 3)+","+hessian.get(4, 3));
		for(int i=0;i<d;i++){
			A.set(i, d, w.get(i));
		}
		Matrix AT = new Matrix(d+1, d+1); 
		AT.setMatrix(0, d, 0, d-1, A.transpose());
		System.out.println("at:"+AT.get(2, d));
		AT.set(d, d, loss.getLambda());
		QRDecomposition QR = new QRDecomposition(AT); 
		Matrix Q = QR.getQ(); 
		Matrix R = QR.getR();
		R = R.getMatrix(0, d-1, 0, d-1);
//		System.out.println("R[2:0]:"+R.getMatrix(2, 2,0,d-1));
//		System.out.println("dimR:"+R.getRowDimension()+"*"+R.getColumnDimension());
//		System.out.println("dimQ:"+Q.getRowDimension()+"*"+Q.getColumnDimension());
		double sigma = Math.signum(Q.det()*R.det()); 
		//############## TEST ######################
//		QRDecomposition QR_new = new QRDecomposition(A.transpose()); 
//		Matrix R_new = QR_new.getR(); 
//		System.out.println("|Q-Q'|:"+R_new.minus(R).normF());
		//############## TEST ######################
		System.out.println("sigma:"+sigma);
		
		// Jacoobian Moor Inverse 
		Matrix Rinv = new Matrix(d+1, d); 
		Rinv.setMatrix(0, d-1, 0, d-1, R.transpose().inverse());
		Matrix A_inv = Q.times(Rinv); 
		// t(a) 
		Matrix ta = Q.getMatrix(0, d, d, d);
		ta = ta.times(sigma); 
		
		// Euler Newton Updates 
		Matrix u = new Matrix(d+1,1); 
		for(int i = 0 ;i< d;i++){ 
			u.set(i, 0, w.get(i));
		}
		u.set(d, 0, loss.getLambda());
		Matrix v = u.plus(ta.times(step_size));
		
		// compute matrix H
		DataPoint w_new = mat2data(v, d); 
		double lambda_new = v.get(d, 0); 
		System.out.println("lambda by v update:"+lambda_new);
		System.out.println("distance v to w:"+ w_new.squaredNormOfDifferenceTo(w));
		loss.set_lambda(lambda_new);
		DataPoint H_d = loss.getAverageGradient(w_new); 
		Matrix H = new Matrix(d,1); 
		for(int i=0;i<d;i++){ 
			H.set(i, 0, H_d.get(i));
		}
		u = v.plus(A_inv.times(H).times(-1.0)); 
		w = mat2data(u, d); 
		loss.set_lambda(u.get(d, 0));
		System.out.println("euler-loss:"+loss.computeLoss(w));
	}
	private DataPoint mat2data(Matrix in, int d){ 
		DataPoint w_new = new DensePoint_efficient(d); 
		for(int i=0;i<d;i++){
			w_new.set(i, in.get(i, 0));
		}
		return w_new;
	}
	

	@Override
	public FirstOrderOpt clone_method() {
		EulerNewton en = new EulerNewton((SecondOrderLoss) loss); 
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
