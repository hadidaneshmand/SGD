package opt.firstorder;

import java.util.ArrayList;
import java.util.List;

import opt.Adapt_Strategy_Double_Full;
import opt.SampleSizeStrategy;
import opt.loss.Dyna_regularizer_loss_e;
import opt.loss.Loss;
import opt.loss.SecondOrderLoss;
import data.DataPoint;
import data.Result;

public class First_Order_Factory_efficient {
	public static FirstOrderOpt[] methods_in; 
	public static FirstOrderOpt method; 
	public static Result out; 
	public static Result out_tes; 
	public static double[] opt_values; 
	public static DataPoint[] opt_params; 
	public static FirstOrderOpt method_for_opt;
	public static int frequency_opt = -1; 
	public static boolean saga_for_opt = false; 
	public static double compute_opt_once(Loss loss,double L){
		int n = loss.getDataSize(); 
		double lambda_n = loss.getLambda(); 
		System.out.println("lambda:"+loss.getLambda());
		double eta_n = 0.3/(L+lambda_n*n); 
		double loss_opt = -1; 
		if(saga_for_opt){ 
		    method_for_opt = new SAGA(loss.clone_loss(),eta_n);
		    method_for_opt.Iterate(loss.getDataSize()*100);
		}
		else{
			method_for_opt = new Newton((SecondOrderLoss) loss.clone_loss()); 
			System.out.println("######## Computing Pivot Optimal #########");
			System.out.println("localnorm:"+((Newton) method_for_opt).getLastLocalNorm());
			while(((Newton) method_for_opt).getLastLocalNorm() > Math.pow(10, -20)){
				method_for_opt.Iterate(1);
				System.out.println("localnorm:"+((Newton) method_for_opt).getLastLocalNorm());
			}
		}
		loss_opt = loss.computeLoss(method_for_opt.getParam()); 
		return loss_opt; 
	}
	public static void compute_opts(Loss loss,double L){ 
		opt_values = new double[methods_in.length]; 
		opt_params = new DataPoint[methods_in.length]; 
		if(loss!=null){ 
			double opt = compute_opt_once(loss,L); 
			for(int i=0;i<methods_in.length;i++){ 
				opt_values[i] = opt; 
			}
		}
		else{ 
			if(frequency_opt == -1){ 
				for(int i=0;i<methods_in.length;i++){ 
					Loss loss_i = methods_in[i].getLoss(); 
					opt_values[i] = compute_opt_once(loss_i,L); 
					opt_params[i] = method_for_opt.getParam(); 
					System.out.println("opt["+i+"]:"+methods_in[i].getLoss().computeLoss(opt_params[i]));
				}
			}
			else{
				int ii =0; 
				while(ii<methods_in.length){ 
					double opt_value = compute_opt_once(methods_in[ii].getLoss(),L); 
					for(int i=0;i<frequency_opt;i++){ 
						opt_values[ii] = opt_value; 
						opt_params[ii] = method_for_opt.getParam(); 
						ii++; 
					}
				}
			}
		}
		for(int i=0;i<methods_in.length;i++){ 
			System.out.println("opt_value["+i+"]="+opt_values[i]);
		}
	}
	public static double train_error(double opt_value){ 
		System.out.println("method_loss:"+method.getLoss().computeLoss(method.getParam())+",opt_value:"+opt_value);
		double error = method.getLoss().computeLoss(method.getParam())-opt_value;
//		error = Math.log(error)/Math.log(2);
		return error; 
	}
	public static double test_error(Loss test_loss){ 
		double error = test_loss.computeLoss(method.getParam());
		return error; 
	}
	public static void run_experiment(int numbexp, Loss loss, int maxItr,int step, double opt_value, Loss test_loss, String out_file,double L){ 
		run_experiment(numbexp, loss, maxItr,step, opt_value, test_loss, out_file, L,false);
	}
	public static void run_experiment(int numbexp, Loss loss, int maxItr,int step, double opt_value, Loss test_loss, String out_file,double L,boolean report_solutionspace ){ 
		if(opt_value == -1){ 
			compute_opts(loss,L);
		}
		else{
		    opt_values = new double[methods_in.length]; 
		    for(int i =0 ;i< methods_in.length;i++){ 
		    	opt_values[i] = opt_value; 
		    }
		}
		ArrayList<String> names = new ArrayList<String>(); 
		for(int i=0;i<methods_in.length;i++){ 
			names.add(methods_in[i].getName()); 
		}
		names.add("steps"); 
		out = new Result(names); 
		out_tes = new Result(names);
		for(int j=0;j<numbexp;j++){
			System.out.println("run:"+j);
			
			List<List> convs = new ArrayList<List>(); 
			List<List>	convs_test = new ArrayList<List>();
			for(int i=0;i<methods_in.length+1;i++){
				convs.add(new ArrayList<Double>()); 
				if(test_loss!=null){
					convs_test.add(new ArrayList<Double>());
				}
			}
			
			for(int i =0;i<methods_in.length;i++){
				method = methods_in[i].clone_method();
				for(int k=0;k<maxItr;k++){
					System.out.println("pass:"+k+"-----------------");
						System.out.println("method name:"+method.getName());
						if(k>0){
							method.Iterate(step);
						}
						System.out.println("finished iterations");
						double error = computeError(method.getParam(), method.getLoss(), i, report_solutionspace);  
						System.out.println("error:"+error);
						convs.get(i).add(error); 
						System.out.println("-------------------");
						if(test_loss!= null){ 
							double test_error =test_error(test_loss);
							System.out.println("test_error:"+test_error);
							convs_test.get(i).add(test_error);
						}
				}
				
			}
			for(int k=0;k<maxItr;k++){ 
				convs.get(convs.size()-1).add((k)*step);
				if(test_loss!=null){ 
					convs_test.get(convs_test.size()-1).add((k)*step);
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
	public static void experiment_with_iterations_complexity(int numbexp, Loss loss, int maxItr, double opt_value, Loss test_loss, String out_file,double L, boolean report_solutionspace,int n){
		if(opt_value == -1){ 
			compute_opts(loss, L);
		}
		else{
		    opt_values = new double[methods_in.length]; 
		    for(int i =0 ;i< methods_in.length;i++){ 
		    	opt_values[i] = opt_value; 
		    }
		}
		ArrayList<String> names = new ArrayList<String>(); 
		for(int i=0;i<3*methods_in.length;i++){
		    if( i % 3 == 0){
		    	int j = i/3; 
		    	names.add(methods_in[j].getName()); 
		    }
		    else if (i % 3 == 1){ 
		    	names.add("steps"+i); 
		    }
		    else{
		    	names.add("time"+i); 
		    }
		}
		Result result = new Result(names); 
		Result result_test = new Result(names); 
		for(int i=0;i<1;i++){ 
			ArrayList<ArrayList<Double>> arr_results = new ArrayList<ArrayList<Double>>(); 
			for(int j=0;j<names.size();j++){
				arr_results.add(j, new ArrayList<Double>());
				if(j%3==0){
					int method_ind = (int)(j/3.0); 
					FirstOrderOpt method = methods_in[method_ind]; 
					System.out.println("methodparam:"+method.getParam().getNorm());
					arr_results.get(j).add(computeError(method.getParam(), loss, method_ind, report_solutionspace)); 
				}
				else{
					arr_results.get(j).add(0.0); 
				}
			}
			ArrayList<ArrayList<Double>> arr_test = new ArrayList<ArrayList<Double>>(); 
			if(test_loss != null){ 
				for(int j=0;j<names.size();j++){
					arr_test.add(j,new ArrayList<Double>()); 
					if(j%3 ==0){
						FirstOrderOpt method = methods_in[(int)(j/3.0)]; 
						arr_test.get(j).add(test_loss.computeLoss(method.getParam())); 
					}
					else{
						arr_test.get(j).add(0.0);
					}
				}
			}
			for(int j=0;j<methods_in.length;j++){ 
				FirstOrderOpt method = methods_in[j].clone_method(); 
				
				while(method.getNum_computed_gradients()/(1.0*n)<= maxItr){  
					System.out.println("======= "+names.get(3*j)+" =======");
					System.out.println("datasetsize:"+method.getDataSize());
					System.out.println("lambda:"+method.getLoss().getLambda());
					method.one_pass();
					
					System.out.println("time:"+method.getTime());
//					double error = loss.computeLoss(method.getParam())-loss_opt; 
					double error = computeError(method.getParam(), loss, j, report_solutionspace); 
					
					double iter = method.getNum_computed_gradients()/(1.0*n);
					if(error < 0){ 
						error = Math.pow(10, -14); 
					}
					System.out.println("loss["+iter+"]="+error);
					arr_results.get(j*3).add(error); 
					arr_results.get(j*3+1).add(method.getNum_computed_gradients()/(1.0*n)); 
					arr_results.get(j*3+2).add(method.getTime()); 
					if(test_loss!=null){
						double error_test = test_loss.computeLoss(method.getParam()); 
						System.out.println("test_loss["+iter+"]="+error_test);
						arr_test.get(j*3).add(error_test); 
						arr_test.get(j*3+1).add(method.getNum_computed_gradients()/(1.0*n));
						arr_test.get(j*3+2).add(method.getTime()); 
					}
					if(error<=Math.pow(10, -14)){
						break;
					}
				}
			}
			for(int j=0;j<names.size();j++){ 
				result.addresult(names.get(j), arr_results.get(j));
			}
			if(test_loss!=null){
				for(int j=0;j<names.size();j++){ 
					result_test.addresult(names.get(j), arr_test.get(j));
				}
			}
		}
		result.write2File(out_file);
		result_test.write2File(out_file+"_test");
	}

	public static double computeError(DataPoint param, Loss loss, int index, boolean report_solutionspace){
		double error = 0; 
		if(report_solutionspace){
			error = param.squaredNormOfDifferenceTo(opt_params[index]); 
		}
		else{ 
			error = loss.computeLoss(param)-opt_values[index]; 
		}
		return error; 
	}
}
