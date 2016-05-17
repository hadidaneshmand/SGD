package opt.firstorder;

import java.util.ArrayList;

import Jama.Matrix;
import data.DataPoint;
import opt.SampleSizeStrategy;
import opt.loss.SecondOrderLoss;

public class NewtonDataDriven extends Newton {
	double c; 
	SampleSizeStrategy as; 
	SecondOrderLoss loss_without_lambda; 
	public NewtonDataDriven(SecondOrderLoss loss, SampleSizeStrategy as, double c) {
		super(loss);
		this.c = c; 
		this.as = as; 
		loss_without_lambda = (SecondOrderLoss) loss.clone_loss(); 
		loss_without_lambda.set_lambda(0.0);
	}
	
	Matrix Last_H_inv; 
	DataPoint storedgrad; 
	boolean use_storage = false; 
	
	@Override
	public int getDataSize() {
		return as.getSubsamplesi();
	}
	@Override
	public void update_iterations(double time) {
		this.time+= time;
		num_computed_gradients+= as.getSubsamplesi(); 
	}
	public void iterate_once(){ 
		if(initialSample == -1){ 
			initialSample = loss.getDataSize(); 
		}
		loss.set_lambda(1.0/as.getSubsamplesi());
		Matrix H_inv = ((SecondOrderLoss)loss).getHessian(w,(ArrayList<Integer>) as.getSubInd()).inverse();
		Last_H_inv = H_inv;
		DataPoint grad = null;
		if(use_storage){
			grad = storedgrad; 
			use_storage = false; 
		}
		else{
			grad = loss.getStochasticGradient(as.getSubInd(), w); 
		}
		double localnorm = grad.scalarProduct(grad.times(H_inv));
		DataPoint delta = grad.times(H_inv); 
		delta = (DataPoint) delta.multiply(-1.0);
		double step_size = 1.0;
		if(localnorm>0.03 && initialSample == loss.getDataSize()){
			step_size = backtracking_line_search(delta); 
		}
			
		System.out.println("step size:"+step_size);
		w = (DataPoint) w.add(delta.multiply(step_size));
		lastLocalNorm = localnorm; 
		System.out.println("local norm:"+lastLocalNorm);
			if(firststep){
				if(localnorm<0.03){ 
					 changeSampleSize();
				}
				
			}
			else{
				if(localnorm < 0.03){ 
					 changeSampleSize();
				}
			}
	}
	
	private void changeSampleSize() {
		System.out.println("++++++++++++++++++++");
		System.out.println("change sample size");
		if(as.getSubsamplesi() == loss.getDataSize()){ 
			return; 
		}
		DataPoint sum_grad = loss_without_lambda.getSumOfGradient(as.getSubInd(), w); 
		sum_grad = (DataPoint) sum_grad.add(w); 
		double cc = c; 
		int ss = as.getSubsamplesi();
		int newss = 0; 
		for(int i=0;i<10;i++){
			System.out.println("++++++++++++++++++++");
			newss = (int) (cc*ss +ss);
			if(newss == ss){
				break; 
			}
			newss = Math.min(loss.getDataSize(), newss); 
			DataPoint sum_grad_n = (DataPoint) sum_grad.add(loss_without_lambda.getSumOfGradient(as.getAllInds().subList(ss, newss), w));
			DataPoint grad = (DataPoint) sum_grad_n.multiply(1.0/newss); 
			DataPoint H_inv_grad = grad.times(Last_H_inv); 
			double first_taylor_term = grad.scalarProduct(H_inv_grad); 
			double second_taylor_term = (1.0/ss-1.0/newss)*H_inv_grad.squaredNorm();  
			double approximate_lambda_nu = first_taylor_term+second_taylor_term; 
			System.out.println("approximate lambda:"+approximate_lambda_nu);
			if(approximate_lambda_nu < 0.03){
				storedgrad = grad; 
				use_storage = true; 
				break;
			}
			else{
				cc = cc*0.7; 
				System.out.println("new increment factor cc:"+cc);
			}
		}
		as.setSampleSize(newss);
	}

	@Override
	public void setName() {
		this.name = "DataDrivenNewton"; 
	}

	@Override
	public FirstOrderOpt clone_method() {
		NewtonDataDriven method = new NewtonDataDriven((SecondOrderLoss) loss.clone_loss(),as,c); 
		method.w = clone_w(); 
		method.num_computed_gradients = this.num_computed_gradients; 
		method.name = this.name; 
		method.lastLocalNorm = this.lastLocalNorm; 
		return method;
	}
	


}
