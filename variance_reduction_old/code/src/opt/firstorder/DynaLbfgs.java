package opt.firstorder;

import opt.loss.Loss;

public class DynaLbfgs extends LBFGS_my {
	
	public DynaLbfgs(Loss loss, int m) {
		super(loss, m);
	}
	
	@Override
	public void setName() {
		this.name = "dyna-lbfgs"; 
	}
	@Override
	public void iterate_once() {
		int lastSize = loss.getDataSize(); 
		super.iterate_once();
		int newSize = loss.getDataSize(); 
		if(newSize!=lastSize){ 
			ys.clear();
			ss.clear(); 
			rhos.clear(); 
			em = 0; 
		}
	}
	
}
