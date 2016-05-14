package opt;

import java.util.ArrayList;
import java.util.List;

public class Adapt_Strategy_GD extends IndexStrategy {
	double kappa;
	int ii; 
	public Adapt_Strategy_GD(int n, double kappa) {
		super(n);
		this.kappa = kappa; 
		this.ss = 16; 
		ii = 0; 
	}
	

	@Override
	public int Tack() {
		ii+=ss; 
		if(ss<kappa){ 
			if(ii>1.0*Math.pow(kappa, 1)*Math.log(ss)){
				ii = 0; 
				ss = ss*2; 
			}
		}
		else{
			ss = (int) (1.0*ss*(kappa/(kappa-1)));
		}
//		System.out.println("ss:"+ss+",kappa:"+kappa+",ii:"+ii+",left:"+(Math.pow(kappa, 1)*Math.log(ss)));
		return ss;
	}

	

	@Override
	public SampleSizeStrategy clone_strategy() {
		Adapt_Strategy_GD out = new Adapt_Strategy_GD(n, kappa);
		out.inds = (ArrayList<Integer>) this.inds.clone();
		out.ss = this.ss; 
		return out;
	}
	
	@Override
	public String getName() {
		return "agd";
	}


	

}
