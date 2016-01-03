package opt.firstorder;

import java.util.ArrayList;
import java.util.List;

import opt.loss.Loss;
import data.Result;

public class First_Order_Factory_efficient {
	public static FirstOrderOpt[] methods_in; 
	public static FirstOrderOpt[] methods; 
	public static Result RunExperiment(int numbexp, Loss loss, int maxItr,int step, double opt_value){ 
		ArrayList<String> names = new ArrayList<String>(); 
		for(int i=0;i<methods_in.length;i++){ 
			names.add(methods_in[i].getName()); 
		}
		names.add("steps"); 
		Result out = new Result(names); 
		for(int j=0;j<numbexp;j++){
			System.out.println("run:"+j);
			methods = new FirstOrderOpt[methods_in.length];
			for(int i=0;i<methods.length;i++){ 
				methods[i] = methods_in[i].clone_method();
			}
			List<List> convs = new ArrayList<List>(); 
			for(int i=0;i<methods.length+1;i++){
				convs.add(new ArrayList<Double>()); 
			}
			for(int i=0;i<methods.length;i++){ 
				double error = Math.abs(loss.getLoss(methods[i].getParam())-opt_value); 
				error = Math.log(error)/Math.log(2); 
				convs.get(i).add(error);
			}
			convs.get(convs.size()-1).add(0);
			for(int k=0;k<maxItr;k++){
				System.out.println("pass:"+k+"-----------------");
				for(int i =0;i<methods.length;i++){ 
					System.out.println("method name:"+methods[i].getName());
					methods[i].Iterate(step);
					System.out.println("finished iterations");
					double error = Math.abs(loss.getLoss(methods[i].getParam())-opt_value); 
					System.out.println("error computed");
					error = Math.log(error)/Math.log(2); 
					convs.get(i).add(error); 
					System.out.println("-------------------");
				}
				convs.get(convs.size()-1).add((k+1)*step);
			}
			for(int i=0;i<names.size();i++){ 
				out.addresult(names.get(i), convs.get(i));
			}
			System.out.println("Free memory (bytes): " + 
					  Runtime.getRuntime().freeMemory());
			methods = null; 
			System.gc();
			System.out.println("Free memory after GC (bytes): " + 
					  Runtime.getRuntime().freeMemory());
		}
		return out;
	}
}
