package opt.firstorder;

import data.DataPoint;
import data.DensePoint_efficient;
import opt.externalcodes.LBFGS;
import opt.externalcodes.LBFGS.ExceptionWithIflag;
import opt.loss.Loss;
import opt.loss.adaptive_loss;

public class LBFGS_external extends FirstOrderOpt{
	int m;
	private int[] iprint;
	private double eps;
	private double xtol;
	private int[] iflag; 
	private double[] diag; 

	public LBFGS_external(Loss loss, int m) {
		super(loss);
		this.m = m; 
		iprint = new int[2]; 
		iprint[0] = 1; 
		iprint[1] = 0; 
		eps = 1.0E-10; 
		xtol = 1.0E-12; 
		iflag = new int[1]; 
		iflag[0] = 0; 
		diag = new double [ loss.getDimension() ];
	}

	@Override
	public void setName() {
		this.name = "lbfgs-"+loss.getType(); 
	}

	@Override
	public void Iterate(int stepNum) {
		for(int i=0;i<stepNum;i++){ 
			num_computed_gradients += loss.getDataSize(); 
			Iterate_once();
		}
	}
	
	public void Iterate_once(){ 
		int d = loss.getDimension(); 
		double[] wx = new double[d]; 
		double[] gx = new double[d]; 
		for(int i=0;i<d;i++){ 
			wx[i] = w.get(i); 
		}
		DataPoint g = loss.getAverageGradient(w); 
		for(int i=0;i<d;i++){ 
			gx[i] = g.get(i); 
		}
		try {
			LBFGS.lbfgs(loss.getDimension(), m, wx, loss.computeLoss(w),gx, false, diag, iprint, eps, xtol, iflag);
//			System.out.println("hessian_min_eigh");
		} catch (ExceptionWithIflag e) {
			e.printStackTrace();
		}
		for(int i=0;i<d;i++){ 
			w.set(i, wx[i]);
		}
		if(loss instanceof adaptive_loss){ 
			((adaptive_loss) loss).tack();
		}
	}

	@Override
	public FirstOrderOpt clone_method() {
		return this;
	}
	

}
