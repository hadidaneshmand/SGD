package opt.firstorder;

import java.util.ArrayList;
import java.util.List;

import opt.loss.Loss;
import data.Result;

public class First_Order_Factory {
	
	public static Result RunExperiment(int numbexp, Loss loss,FirstOrderOpt[] methods_in, int maxItr,int step, double opt_value){ 
		ArrayList<String> names = new ArrayList<String>(); 
		for(int i=0;i<methods_in.length;i++){ 
			names.add(methods_in[i].getName()); 
		}
		names.add("steps"); 
		Result out = new Result(names); 
		for(int j=0;j<numbexp;j++){
			System.out.println("run:"+j);
			FirstOrderOpt[]	methods = new FirstOrderOpt[methods_in.length];
			for(int i=0;i<methods.length;i++){ 
				methods[i] = methods_in[i].clone_method();
			}
			List<List> convs = new ArrayList<List>(); 
			for(int i=0;i<methods.length+1;i++){
				convs.add(new ArrayList<Double>()); 
			}
			for(int k=0;k<maxItr;k++){
				for(int i =0;i<methods.length;i++){ 
					methods[i].Iterate(step);
					double error = Math.abs(loss.getLoss(methods[i].getParam())-opt_value); 
					error = Math.log(error)/Math.log(2); 
					convs.get(i).add(error); 
				}
				convs.get(convs.size()-1).add((k+1)*step);
			}
			for(int i=0;i<names.size();i++){ 
				out.addresult(names.get(i), convs.get(i));
			}
			System.gc();
		}
		return out;
	}
}
