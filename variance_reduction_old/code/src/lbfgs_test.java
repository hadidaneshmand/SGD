
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import opt.Adapt_Strategy_Double_Full;
import opt.firstorder.GD;
import opt.firstorder.SAGA;
import opt.firstorder.LBFGS_my;
import opt.loss.Dyna_samplesize_loss_e;
import opt.loss.Logistic_Loss_efficient;
import plot.XYLinesChart;
import data.DataPoint;
import data.SparsePoint;


public class lbfgs_test {
	public static DataPoint[] data; 
	public static DataPoint[] test_data = null; 
	public static SAGA saga_opt; 
	public static void readDataPointsFromFile(String filename, int startIndex, int data_size, boolean is_test) {
		int pos = 0; 
		int neg = 0; 
		try {
			BufferedReader fp = new BufferedReader(new FileReader(filename));
			String line;
			int c = 0; 
			while ((line = fp.readLine()) != null && c <data_size) {
				try {
					DataPoint point = new SparsePoint();
					StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
					double label = Double.valueOf(st.nextToken());					// label has to be at the first position of the text row
					if(label == 1){ 
						pos++;
					}
					else if(label == -1 || label == 2 || label == 0){
						label = -1; 
						neg++; 
					}
					point.setLabel(label);

					while (st.hasMoreTokens()) {
						int feature = Integer.valueOf(st.nextToken()) - startIndex;
						double value = Double.valueOf(st.nextToken());
						point.set(feature, value);
					}
					if(is_test){ 
						test_data[c] = point;
					}else{ 
						data[c] = point;
					}
					 
					c++;
				if(c %50000 == 0){ 
					System.out.println("c:"+c);	
				}
				if(c<10){ 
					System.out.println(point.toString());
				}
				} catch (NumberFormatException e) {
					System.out.println("Could not read datapoint number "+c + " since Line "+line+" seems to be not properly formatted: "+e.getMessage());
				} 
			}
			fp.close();
			System.out.println("pos:"+pos+",neg:"+neg);
		} catch (IOException e) {
			System.out.println("Could not read from file " + filename + " due to " + e.getMessage());
		}
		System.out.println("Free memory (bytes): " + 
				  Runtime.getRuntime().freeMemory());
		System.out.println("Total memory (bytes): " + 
				  Runtime.getRuntime().totalMemory());
		
	}
	public static void main(String[] args) {
		int n =  50000; 
		int d =  54; 
		int T = 40; 
		double lambda = 1.0/Math.sqrt(n);
		data = new DataPoint[n]; 
		readDataPointsFromFile("data/covtype", 1,n,false);
		System.out.println("n:"+data.length);
		List<Double> convergence_lbfgs = new ArrayList<Double>(); 
		List<Double> convergence_gd = new ArrayList<Double>(); 
		List<Double> convergence_saga = new ArrayList<Double>(); 
		List<Double> t = new ArrayList<Double>(); 
		Logistic_Loss_efficient loss = new Logistic_Loss_efficient(data, d);
		loss.setLambda(lambda);
		double learning_rate = 0.1; 
		LBFGS_my lbfgs_method = new LBFGS_my(loss,20); 
		lbfgs_method.setStepSize(learning_rate);
		SAGA saga = new SAGA(loss, 1.0/n); 
		GD gd = new GD(loss);
		gd.setStepSize(learning_rate);
		t.add(0.0);
		SAGA opt = new SAGA(loss.clone_loss(), 1.0/n);
		opt.Iterate(n*150);
		double loss_opt = loss.computeLoss(opt.getParam());
		System.out.println("opt optimized!!");
		convergence_lbfgs.add(Math.log(loss.computeLoss(lbfgs_method.getParam()))); 
		convergence_gd.add(Math.log(loss.computeLoss(gd.getParam())));
		convergence_saga.add(Math.log(loss.computeLoss(saga.getParam())));
		for(int i=0;i<T;i++){
			System.out.println("i:"+i);
			lbfgs_method.Iterate(1);
			gd.Iterate(1);
//			pegasos.Iterate(subT);
//			t.add(Math.log(i+1));
			saga.Iterate(n);
//			convergence_pegasos.add(Math.log(b.squaredNormOfDifferenceTo(pegasos.getParam()))); 
//			convergence_pegasos_m.add(Math.log(b.squaredNormOfDifferenceTo(pegasos_m.getParam())));
			double error_lbfgs = Math.log(loss.computeLoss(lbfgs_method.getParam())-loss_opt);
			System.out.println("lbfgs_error:"+error_lbfgs);
			double error_saga = Math.log(loss.computeLoss(saga.getParam())-loss_opt);
			System.out.println("saga_error:"+error_saga);
			convergence_lbfgs.add(error_lbfgs); 
			convergence_gd.add(Math.log(loss.computeLoss(gd.getParam()))-loss_opt);
			convergence_saga.add(error_saga);
		}
		List<List<Double>> series = new ArrayList<List<Double>>(); 
		series.add(convergence_lbfgs); 
		series.add(convergence_gd);
		series.add(convergence_saga); 
		List<String> names = new ArrayList<String>(); 
		names.add("lbfgs");
		names.add("gd"); 
		names.add("saga"); 
		XYLinesChart xyplot = new XYLinesChart(series, t, names, "lbfgs convergence", "iteration", "loss");
		xyplot.setVisible(true);
	}
}
