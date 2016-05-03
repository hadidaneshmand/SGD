import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import opt.firstorder.FirstOrderOpt;
import opt.firstorder.SAGA;
import opt.loss.FirstOrderEfficient;
import opt.loss.Logistic_Loss_efficient;
import opt.loss.Loss;
import data.DataPoint;
import data.Result;


public class scaling {
	public static void main(String[] args) {
		Input.initialize(args);
		int n = Input.loss_train.getDataSize();
		double pivot_l = 1.0/(500*(3+1));;
		int num_lambda_update = 20; 
		
		ArrayList<String> names = new ArrayList<String>(); 
		names.add("loss"); 
		names.add("omega"); 
		Result result = new Result(names);
		SAGA[] saga_opts = new SAGA[num_lambda_update]; 
		FirstOrderEfficient[] losses = new FirstOrderEfficient[num_lambda_update]; 
		Loss pivot_loss = Input.loss_train.clone_loss(); 
		Loss reg_free = Input.loss_train.clone_loss(); 
		reg_free.set_lambda(0.0);
		pivot_loss.set_lambda(pivot_l);
		FirstOrderOpt saga = new SAGA(pivot_loss, 0.3/(Input.L+n*pivot_l)); 
		saga.Iterate((int) (10*n*Math.log(n)));
		ArrayList<Integer> inds = new ArrayList<Integer>(); 
		for(int i=0;i<n;i++){ 
			inds.add(i); 
		}
		DataPoint w = saga.getParam();
			Collections.shuffle(inds);
			List<Double> loss = new ArrayList<Double>(); 
			List<Double> omega = new ArrayList<Double>();
			for(int i=0;i<num_lambda_update;i++){
				double alpha = 1 + 1.0*i/num_lambda_update;
				DataPoint w_alpha = (DataPoint) w.multiply(alpha); 
				System.out.println("alpha["+i+"]:"+alpha);
				double loss_i = reg_free.computeLoss(w_alpha);
				System.out.println("loss["+i+"]:"+loss_i);
				loss.add(loss_i);
				double omega_i = w_alpha.squaredNorm(); 
				System.out.println("omega["+i+"]:"+omega_i);
				omega.add(omega_i);  
			}
			result.addresult(names.get(0), loss);
			result.addresult(names.get(1),omega);
		result.write2File(Input.config.logDir+"_scaling");
	}
}
