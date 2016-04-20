import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import opt.utils;
import opt.firstorder.FirstOrderOpt;
import opt.firstorder.SAGA;
import opt.firstorder.SGD;
import opt.loss.Logistic_Loss_efficient;
import opt.loss.Loss;
import opt.loss.FirstOrderEfficient;
import data.DataPoint;
import data.Result;


public class lambda_in_obj {
	public static void main(String[] args) {
		Input.initialize(args);
		int n = Input.loss_train.getDataSize();
		double pivot_l = 1.0/n;
		int num_lambda_update = 7; 
		
		ArrayList<String> names = new ArrayList<String>(); 
		names.add("suboptimality"); 
		names.add("steps");
		Result result = new Result(names);
		SAGA[] saga_opts = new SAGA[num_lambda_update]; 
		FirstOrderEfficient[] losses = new FirstOrderEfficient[num_lambda_update]; 
		
		
		Loss pivot_loss = Input.loss_train.clone_loss(); 
		pivot_loss.set_lambda(pivot_l);
		FirstOrderOpt saga = new SAGA(pivot_loss, 0.3/(Input.L+n*pivot_l)); 
		saga.Iterate((int) (10*n*Math.log(n)));
		double pivot_val = pivot_loss.computeLoss(saga.getParam());
		ArrayList<Integer> inds = new ArrayList<Integer>(); 
		for(int i=0;i<n;i++){ 
			inds.add(i); 
		}
		DataPoint[] fulldata = ((FirstOrderEfficient) Input.loss_train.clone_loss()).getData(); 
		for(int k=0;k<4;k++){
			Collections.shuffle(inds);
			List<Double> ls = new ArrayList<Double>(); 
			List<Double> opt = new ArrayList<Double>(); 
			for(int i=0;i<num_lambda_update;i++){
				double coeff = Math.pow(1.5, i);
				int subsi = (int) ((int) 2000*coeff);
				System.out.println("sample size:"+subsi);
				DataPoint[] data = new DataPoint[subsi]; 
				for(int j=0;j<subsi;j++){ 
					data[j] = fulldata[inds.get(j)]; 
				}
				double lambda_i = 1.0/subsi;
				losses[i] = new Logistic_Loss_efficient(data, Input.d);
				losses[i].set_lambda(lambda_i);
//				losses[i].setData(data);
				double eta_n = 0.3/(Input.L+lambda_i*subsi); 
				saga_opts[i] = new SAGA(losses[i], eta_n);
				saga_opts[i].Iterate((int) (10*n*Math.log(n)));
				if(i>-1){
					ls.add((double) subsi);
					System.out.println("loss["+i+"]:"+saga_opts[i].getLoss().computeLoss(saga_opts[i].getParam()));
					double loss_opt = pivot_loss.computeLoss(saga_opts[i].getParam())-pivot_val;
					System.out.println("saga["+i+"]:"+loss_opt); 
					System.out.println("norm2:"+saga_opts[i].getParam().squaredNorm());
					System.out.println("distances:"+saga_opts[i].getParam().squaredNormOfDifferenceTo(saga.getParam()));
					opt.add(loss_opt); 
				}
			}
			result.addresult(names.get(0), opt);
			result.addresult(names.get(1),ls);
		}
		
		result.write2File(Input.config.logDir+"_dyna_lambda");
	}
}
