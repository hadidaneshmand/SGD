import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import opt.firstorder.Pegasos;
import opt.firstorder.SGD;
import opt.loss.Hinge_loss;
import plot.XYLinesChart;
import data.DataPoint;
import data.IOTools;
import data.SparsePoint;


public class SVM_adapt_r {
	public static List<DataPoint> getRandomsubset(int size,List<DataPoint> data){ 
		ArrayList<Integer> indices = new ArrayList<Integer>(); 
		Random r = new Random(); 
		for(int i=0;i<size;i++){ 
			indices.add(r.nextInt(data.size())); 
		}
		return Adapt_ss.getsubsample(data, indices); 
	}
	public static DataPoint getRandomDataPoint(int d){ 
		DataPoint out = new SparsePoint(); 
		Random r = new Random(); 
		for(int i=0;i<d;i++){ 
			out.set(i, r.nextDouble());
		}
		return out; 
	}
	public static void main(String[] args) {
		List<DataPoint> data = IOTools.readDataPointsFromFile("datas/ijcnn1", 0);
		int n = 49990; 
		int d = 22; 
		int T = 100; 
		int subT = 1000;
		double margin = 1.0/n; 
		double lambda = 1.0/n; 
		DataPoint b = new SparsePoint(); 
		Random r = new Random(); 
//		for(int i=0;i<d;i++){ 
//			b.set(i,r.nextDouble()); 
//		}
//		b = (DataPoint) b.normalize(); 
//		while(true){
//			if(data.size() == n){ 
//				break;
//			}
//			DataPoint r_data = getRandomDataPoint(d); 
//			r_data = (DataPoint) r_data.normalize(); 
//			if(r_data.scalarProduct(b)>= margin){ 
//				r_data.setLabel(+1);
//				data.add(r_data); 
//			}
//			else if(r_data.scalarProduct(b)<=margin){ 
//				r_data.setLabel(-1);
//				data.add(r_data);
//			}
//		}
		List<Double> convergence_pegasos = new ArrayList<Double>(); 
		List<Double> convergence_pegasos_m = new ArrayList<Double>(); 
		List<Double> t = new ArrayList<Double>(); 
		Hinge_loss h_loss = new Hinge_loss(data, d);
		Hinge_loss loss_adapt = new Hinge_loss(data, d); 
		loss_adapt.adaptiveSampling(2*d);
		h_loss.setLambda(lambda);
		SGD pegasos = new SGD(h_loss); 
		System.out.println("distance:"+b.squaredNormOfDifferenceTo(pegasos.getParam()));
		Pegasos pegasos_m = new Pegasos(loss_adapt);
		pegasos.setLearning_rate(1.0);
		t.add(0.0);
		convergence_pegasos.add(Math.log(h_loss.getLoss(pegasos.getParam()))); 
		convergence_pegasos_m.add(Math.log(h_loss.getLoss(pegasos_m.getParam())));
		for(int i=0;i<T;i++){
			System.out.println("i:"+i);
			pegasos_m.Iterate(subT);
			pegasos.Iterate(subT);
			t.add(subT*(i+1.0));
//			convergence_pegasos.add(Math.log(b.squaredNormOfDifferenceTo(pegasos.getParam()))); 
//			convergence_pegasos_m.add(Math.log(b.squaredNormOfDifferenceTo(pegasos_m.getParam())));
			convergence_pegasos.add(Math.log(h_loss.getLoss(pegasos.getParam()))); 
			convergence_pegasos_m.add(Math.log(h_loss.getLoss(pegasos_m.getParam())));
		}
		System.out.println("loss:"+h_loss.getLoss(b));
		List<List<Double>> series = new ArrayList<List<Double>>(); 
		series.add(convergence_pegasos); 
		series.add(convergence_pegasos_m);
		List<String> names = new ArrayList<String>(); 
		names.add("pegasos");
		names.add("pegasos m");
		System.out.println(convergence_pegasos_m);
		XYLinesChart xyplot = new XYLinesChart(series, t, names, "pegasos convergence", "iteration", "loss");
		xyplot.setVisible(true);
	}
}
