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


public class lambda_in_obj {
	public static void main(String[] args) {
		Input.initialize(args);
		int n = Input.loss_train.getDataSize();
		double pivot_l = 1.0/n;
		int num_lambda_update = 20; 
		
		ArrayList<String> names = new ArrayList<String>(); 
		names.add("loss"); 
		names.add("omega"); 
		names.add("distances"); 
		names.add("lambda");
		Result result = new Result(names);
		SAGA[] saga_opts = new SAGA[num_lambda_update]; 
		FirstOrderEfficient[] losses = new FirstOrderEfficient[num_lambda_update]; 
		
		
		Loss pivot_loss = Input.loss_train.clone_loss(); 
		Loss reg_free = Input.loss_train.clone_loss(); 
		reg_free.set_lambda(0.0);
		pivot_loss.set_lambda(pivot_l);
		FirstOrderOpt saga = new SAGA(pivot_loss, 0.3/(Input.L+n*pivot_l)); 
		saga.Iterate((int) (10*n*Math.log(n)));
		double pivot_val = pivot_loss.computeLoss(saga.getParam());
		ArrayList<Integer> inds = new ArrayList<Integer>(); 
		for(int i=0;i<n;i++){ 
			inds.add(i); 
		}
		DataPoint[] fulldata = ((FirstOrderEfficient) Input.loss_train.clone_loss()).getData(); 
		for(int k=0;k<1;k++){
			Collections.shuffle(inds);
			List<Double> lambda = new ArrayList<Double>(); 
			List<Double> loss = new ArrayList<Double>(); 
			List<Double> omega = new ArrayList<Double>();
			List<Double> distances = new ArrayList<Double>();
			for(int i=0;i<num_lambda_update;i++){
				double mu =1.0/(500*(i+1));
				int subsi = n; 
				DataPoint[] data = Input.data;
				double lambda_i = mu;
				losses[i] = new Logistic_Loss_efficient(data, Input.d);
				losses[i].set_lambda(lambda_i);
				double eta_n = 0.3/(Input.L+lambda_i*subsi); 
				saga_opts[i] = new SAGA(losses[i], eta_n);
				saga_opts[i].Iterate((int) (10*subsi*Math.log(subsi)));
				if(i>-1){
					System.out.println("lambda["+i+"]:"+lambda_i);
					lambda.add((double) mu);
					double loss_i = reg_free.computeLoss(saga_opts[i].getParam());
					System.out.println("loss["+i+"]:"+loss_i);
					loss.add(loss_i); 
					double omega_i = saga_opts[i].getParam().squaredNorm(); 
					System.out.println("omega["+i+"]:"+omega_i);
					omega.add(omega_i); 
					double distance_i = saga_opts[i].getParam().squaredNormOfDifferenceTo(saga.getParam()); 
					System.out.println("distances["+i+"]:"+distance_i);
					distances.add(distance_i); 
				}
			}
			result.addresult(names.get(0), loss);
			result.addresult(names.get(1),omega);
			result.addresult(names.get(2),distances);
			result.addresult(names.get(3),lambda);
		}
		result.write2File(Input.config.logDir+"_pareto");
	}
}
