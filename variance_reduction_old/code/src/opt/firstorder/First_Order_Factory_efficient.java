package opt.firstorder;

import java.util.ArrayList;
import java.util.List;

import opt.loss.Loss;
import opt.loss.MissClass;
import opt.loss.MissClass_efficient;
import data.DataPoint;
import data.DensePoint;
import data.Result;

public class First_Order_Factory_efficient {
	public static FirstOrderOpt[] methods_in; 
	public static FirstOrderOpt method; 
	public static Result out; 
	public static Result out_tes; 
	static List<List> convs;
	static List<List> convs_test; 
	public static void RunExperiment(int numbexp, Loss loss, int maxItr,int step, double opt_value, Loss test_loss, double loss_test_opt,String out_file){ 
		
		ArrayList<String> names = new ArrayList<String>(); 
		for(int i=0;i<methods_in.length;i++){ 
			names.add(methods_in[i].getName()); 
		}
		names.add("steps"); 
		out = new Result(names); 
		out_tes = new Result(names);
		for(int j=0;j<numbexp;j++){
			System.out.println("run:"+j);
			
			convs = new ArrayList<List>(); 
			for(int i=0;i<methods_in.length+1;i++){
				convs.add(new ArrayList<Double>()); 
			}
			System.out.println("loss zero:"+loss.getLoss(DensePoint.zero(loss.getDimension())));
			for(int i=0;i<methods_in.length;i++){ 
				double error = Math.abs(loss.getLoss(DensePoint.zero(loss.getDimension()))-opt_value); 
				error = Math.log(error)/Math.log(2); 
				convs.get(i).add(error);
			}
			convs.get(convs.size()-1).add(0);
			if(test_loss != null){
				convs_test = new ArrayList<List>(); 
				for(int i=0;i<methods_in.length+1;i++){
					convs_test.add(new ArrayList<Double>()); 
				}
				
				for(int i=0;i<methods_in.length;i++){ 
					double error = Math.abs(test_loss.getLoss(DensePoint.zero(test_loss.getDimension()))- loss_test_opt); 
					error = Math.log(error)/Math.log(2); 
					convs_test.get(i).add(error);
				}
				System.out.println("test:"+test_loss.getLoss(DensePoint.zero(loss.getDimension())));
				convs_test.get(convs_test.size()-1).add(0);
			}
			
			for(int i =0;i<methods_in.length;i++){
				method = methods_in[i].clone_method();
				for(int k=0;k<maxItr;k++){
					System.out.println("pass:"+k+"-----------------");
					
						 
						System.out.println("method name:"+method.getName());
						method.Iterate(step);
						System.out.println("finished iterations");
						double error = (loss.getLoss(method.getParam())-opt_value); 
						System.out.println("error:"+error);
						error = Math.log(error)/Math.log(2); 
						convs.get(i).add(error); 
						System.out.println("-------------------");
						if(test_loss!= null){ 
							double test_error =(test_loss.getLoss(method.getParam()));
							System.out.println("test_error:"+test_error);
							convs_test.get(i).add(test_error);
						}
				}
				
			}
			for(int k=0;k<maxItr;k++){ 
				convs.get(convs.size()-1).add((k+1)*step);
				if(test_loss!=null){ 
					convs_test.get(convs_test.size()-1).add((k+1)*step);
				}
			}
			for(int i=0;i<names.size();i++){ 
				out.addresult(names.get(i), convs.get(i));
			}
			if(test_loss!=null){ 
				for(int i=0;i<names.size();i++){
					out_tes.addresult(names.get(i), convs_test.get(i));
				}
			}
			System.out.println("Free memory (bytes): " + 
					  Runtime.getRuntime().freeMemory());
			System.gc();
			System.out.println("Free memory after GC (bytes): " + 
					  Runtime.getRuntime().freeMemory());
		}
		out.write2File(out_file);
		if(test_loss!=null){
			out_tes.write2File(out_file+"_test");
		}
	}
}
