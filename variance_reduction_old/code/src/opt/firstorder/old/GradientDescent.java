package opt.firstorder.old;



import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import opt.Optimizer;
import opt.utils;
import opt.config.Config;
import data.DataPoint;
import data.DensePoint;
import data.SparsePoint;

public class GradientDescent implements Optimizer  {
	protected Config conf;
	protected List<DataPoint> data;
	protected DataPoint w;
	protected boolean verbose; 
	protected List<Double> obj_trace;
	protected double eta;
	protected DataPoint optimal; 
	protected double lambda;
	
	public void setParam(DataPoint in){ 
		this.w = in;
	}
	public void setOptimal(DataPoint in){ 
		optimal = in; 
	}
	public DataPoint getOptimal(){
		return optimal;
	}
	public void setVerbose(boolean verbose){ 
		this.verbose = verbose;
	}
	public void setEta(double eta){ 
		this.eta = eta; 
	}
	public DataPoint getParam(){ 
		return this.w;
	}
	public GradientDescent(List<DataPoint> data, Config conf,double lambda) {
		this.data = data; 
		this.conf = conf;
		w = new DensePoint(conf.featureDim);
		//TODO ask about this!
		SparsePoint p = (SparsePoint) data.get(0);
		for (int i =0;i<conf.featureDim;i++) {
			w.set(i, 0);
		}
		verbose = false;
		obj_trace = new ArrayList<Double>(); 
		eta = conf.eta0;
		this.lambda  = lambda; 
	}
	@Override
	public void setConfig(Config conf) {
		this.conf = conf;
	}
	/*
	 * Compute gradient over point p and given parameter w 
	 * TODO I have to change point p to index i
	 * TODO The different loss problem
	 */
	public DataPoint computeStochasticGradient(
			DataPoint p) {

		DataPoint g = null;

		switch (conf.lossType) {
		case BINARY_SVM: {
			// loss =  log[1 + exp(−ywTx)] + (lambda/2 * ||w||^2)
			double prod = p.scalarProduct(w);
			int y = (int) p.getLabel();
			prod = Math.exp(-1*prod *y);
			g = (DataPoint) p.multiply(-y);
			g = (DataPoint) g.multiply(prod/(1+prod));
			g = (DataPoint) g.add(w.multiply(lambda)); // add regularization term lambda*w (cost function is n*lambda*|w|^2)TODO ask about this !!!!
			break;
		}
		case REGRESSION: {
			// use squared loss: (w^T*x - y)^2 + (lambda/2 * ||w||^2)
			double y = p.getLabel();
			g = (DataPoint) p.multiply(2 * (w.scalarProduct(p) - y));
			g = (DataPoint) g.add(w.multiply(lambda)); // add regularizer			
			break;
		}
		case MULTICLASS_REGRESSION: {
			
			int f = conf.featureDim;
			g = new SparsePoint();		

			double y = p.getLabel();
			for (int c = 0; c < conf.nClasses; ++c) {
				
				// compute mu_c = exp(< w_j, p >)/Z
				// where Z = \sum_r exp(< w_r, p >)
				// this is also equal to mu_j = 1/(\sum_{r!=c} exp(< w_r, p >)) 
				DataPoint wc = (DataPoint) w.sub(c * f, (c + 1) * f);
				double dp = wc.scalarProduct(p);				
				double denom = 1.0;
				for (int r = 0; r < conf.nClasses; ++r) {
					if(c != r) {
						DataPoint wr = (DataPoint) w.sub(r * f, (r + 1) * f);
						double dpr = wr.scalarProduct(p);
						denom += Math.exp(dpr-dp);
					}
				}
				double mu = 0;
				if(!Double.isNaN(denom)) {
					mu = 1.0/denom;
				}
				
				//int yc = ((c == 0 && y == -1) || (c == 1 && y == 1)) ? 1 : 0;
				int yc = (c == (int)(y-conf.c0)) ? 1 : 0; 
				DataPoint gc = (DataPoint) p.multiply(mu - yc);
				gc = (DataPoint) gc.add(w.multiply(lambda)); // add regularization term lambda*w (cost function is n*lambda*|w|^2) TODO ask!!
				
				if(gc instanceof SparsePoint) {
					SparsePoint s = (SparsePoint) gc;
					for (int i : s.featureSet()) {
						g.set(c * f + i, s.get(i));
					}
				} else {					
					for (int i = 0; i < f; ++i) {
						g.set(c * f + i, gc.get(i));
					}
				}
			}
			break;
		}
		}

		return g;
	}
	
	
	/*
	 * Compute gradient over the whole dataset data
	 */
	public DataPoint computeAverageGradient() {
		DataPoint g = new SparsePoint();
		for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
			DataPoint p = (DataPoint) iter.next();
			DataPoint gi = computeStochasticGradient(p);
			g = (DataPoint) g.add(gi);
		}
		g = (DataPoint) g.multiply(1.0 / data.size());

		return g;
	}
	
	@Override
	public void optimize(int iteration_num) {
		Random generator = utils.getGenerator();
		if(conf.initType == Config.InitType.RANDOM) {
			SparsePoint p = (SparsePoint) data.get(0);
			for (int i : p.featureSet()) {
				w.set(i, generator.nextDouble());
			}
		}
		for (int k = 0; k < iteration_num; ++k) {
			DataPoint mu = computeAverageGradient();
			if (mu != null) {
				w = (DataPoint) w.subtract(mu.multiply(eta));
				double obj = computeObjective();
				obj_trace.add(obj);
				if(verbose){ 
					
					System.out.println("iteration:"+k+","+obj);
				}
			}
			
		}
	}
	
	public List<Double> getObjTrace(){
		return this.obj_trace;
	}
	@Override
	public void setData(List<DataPoint> data) {
		this.data = data; 
	}
	@Override
	public DataPoint getOptParam() {
		return this.w; 
	}
	@Override
	public double getObj() {
		double obj = 0; 

		switch(conf.objType) {
			case OBJ_LOSS: {
				obj = computeLoss();
				break;
			}
			case OBJ_CLASSIFICATION_ERROR: {
				obj = computeClassificationError();
				break;
			}
			case OBJ_DIST_TO_OPTIMUM: {
				obj = computeSquaredDistanceFromOpt();
				break;
			}
		}
		return obj;
	}
	/*
	 * TODO ask about binary svm (without regularizer) + lambda issue
	 */
public  double computeLoss() {
		
		double loss = 0;
		if(data == null) {
			return -1;
		}

		switch (conf.lossType) {
		case BINARY_SVM: {
			// loss =  \sum_x log[1 + exp(−ywTx)]/n + (lambda/2 * ||w||^2)
			for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				int y = (int) p.getLabel();
				loss += Math.log(1 + Math.exp(-1*y*p.scalarProduct(w))); 
			}
			loss /= data.size();
			loss += Math.pow(w.getNorm(),2)*lambda/2;
			break;
		}
		case REGRESSION: {
			// use squared loss: (1/n) * (w^T*x - y)^2
			for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				double y = p.getLabel();
				double t = (w.scalarProduct(p) - y);
				loss += t*t;
			}
			loss /= data.size();
			break;
		}
		case MULTICLASS_REGRESSION: {

			int f = conf.featureDim;
			SparsePoint w_n = (SparsePoint) w.normalize();

			double log_likelihood = 0;
			
			for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				int y = (int) p.getLabel() - conf.c0;
				
				double norm_coeff = 0;
				for (int c = 0; c < conf.nClasses; ++c) {
					DataPoint wc = (DataPoint) w_n.sub(c * f, (c + 1) * f);
					double dp = wc.scalarProduct(p);
					norm_coeff += Math.exp(dp);
				}
				
				if(norm_coeff != 0) {
					norm_coeff = Math.log(norm_coeff);
					
					DataPoint wy = (DataPoint) w_n.sub(y * f, (y + 1) * f);
					double dp = wy.scalarProduct(p);
					
					log_likelihood += dp - norm_coeff;
				}
				
			}
			loss = log_likelihood; // minimize loss
			
			break;
		}
		}
		return loss;
	}
  public double computeClassificationError() {

		double loss = 0;
		if(data == null) {
			return -1;
		}
		
		int nClasses = conf.nClasses;
	
		switch (conf.lossType) {
		case BINARY_SVM: {
			double[] correct = new double[nClasses];
			double[] incorrect = new double[nClasses];
			for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				int y = (int) p.getLabel();
				int offset = (y == -1) ? 0 : 1;
				double o = p.scalarProduct(w);
				if (y * o > 0) {
					++correct[offset];
				} else {
					++incorrect[offset];
				}
			}
	
			for (int c = 0; c < nClasses; ++c) {
				loss += incorrect[c];
			}
			loss /= data.size();
			break;
		}
		case REGRESSION: {
			// use squared loss: (w^T*x - y)^2
			for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				double y = p.getLabel();
				// double t = (w.scalarProduct(p) - y);
				// loss += t*t;
	
				int yp = (w.scalarProduct(p) > 0) ? 1 : -1;
				if (yp != (int) y) {
					++loss;
				}
			}
			loss /= data.size();
			break;
		}
		case MULTICLASS_REGRESSION: {
	
			int f = conf.featureDim;//TODO I should ask about feature vs data_param.f
	
			for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				double y = p.getLabel();
	
				/*
				 * double norm_coeff = 0; for (int c = 0; c < Config.nClasses;
				 * ++c) { DataPoint wc = (DataPoint) w.sub(c * f, (c + 1) * f);
				 * double dp = wc.scalarProduct(p); norm_coeff += Math.exp(dp);
				 * }
				 */
				double norm_coeff = 1.0; // don't need normalization constant for class assignment
	
				double max_mu = -Double.MAX_VALUE;
				int yp = 0;
				for (int c = 0; c < nClasses; ++c) {
					DataPoint wc = (DataPoint) w.sub(c * f, (c + 1) * f);
					double dp = wc.scalarProduct(p);
					double mu = Math.exp(dp) / norm_coeff;
					mu /= norm_coeff;
					if (max_mu < mu) {
						max_mu = mu;
						yp = c;
					}
				}
				yp += conf.c0;
				
				
				if (yp != (int) y) {
					++loss;
				}
			}
			loss /= data.size();
			break;
		}
		}
	return loss;
}
  public double computeSquaredDistanceFromOpt() {
		double dist = -1;
		DataPoint s = this.optimal;
		if(s!=null) {
			dist = w.squaredNormOfDifferenceTo(s);
		}
		return dist;
	}
  public double computeObjective() {
		double obj = 0;
		switch(conf.objType) {
			case OBJ_LOSS: {
				obj = computeLoss();
				break;
			}
			case OBJ_CLASSIFICATION_ERROR: {
				obj = computeClassificationError();
				break;
			}
			case OBJ_DIST_TO_OPTIMUM: {
				obj = computeSquaredDistanceFromOpt();
				break;
			}
		}
		return obj;
	}
  public double computeObjective(DataPoint win) {
		double obj = 0;
		switch(conf.objType) {
			case OBJ_LOSS: {
				obj = computeLoss(win);
				break;
			}
			case OBJ_CLASSIFICATION_ERROR: {
				obj = computeClassificationError(win);
				break;
			}
			case OBJ_DIST_TO_OPTIMUM: {
				obj = computeSquaredDistanceFromOpt(win);
				break;
			}
		}
		return obj;
	}
	 public double computeSquaredDistanceFromOpt(DataPoint w) {
		 double dist = -1;
			DataPoint s = this.optimal;
			if(s!=null) {
				SparsePoint w_n = (SparsePoint) w.normalize();
				SparsePoint s_n = (SparsePoint) s.normalize();
				dist = w_n.squaredNormOfDifferenceTo(s_n);;
			}
			return dist;
    }
	double computeClassificationError(DataPoint w) {

		double loss = 0;
		if(data == null) {
			return -1;
		}
		
		int nClasses = conf.nClasses;
	
		switch (conf.lossType) {
		case BINARY_SVM: {
			double[] correct = new double[nClasses];
			double[] incorrect = new double[nClasses];
			for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				int y = (int) p.getLabel();
				int offset = (y == -1) ? 0 : 1;
				double o = p.scalarProduct(w);
				if (y * o > 0) {
					++correct[offset];
				} else {
					++incorrect[offset];
				}
			}
	
			for (int c = 0; c < nClasses; ++c) {
				loss += incorrect[c];
			}
			loss /= data.size();
			break;
		}
		case REGRESSION: {
			// use squared loss: (w^T*x - y)^2
			for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				double y = p.getLabel();
				// double t = (w.scalarProduct(p) - y);
				// loss += t*t;
	
				int yp = (w.scalarProduct(p) > 0) ? 1 : -1;
				if (yp != (int) y) {
					++loss;
				}
			}
			loss /= data.size();
			break;
		}
		case MULTICLASS_REGRESSION: {
	
			int f = conf.featureDim;//TODO I should ask about feature vs data_param.f
	
			for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				double y = p.getLabel();
	
				/*
				 * double norm_coeff = 0; for (int c = 0; c < Config.nClasses;
				 * ++c) { DataPoint wc = (DataPoint) w.sub(c * f, (c + 1) * f);
				 * double dp = wc.scalarProduct(p); norm_coeff += Math.exp(dp);
				 * }
				 */
				double norm_coeff = 1.0; // don't need normalization constant for class assignment
	
				double max_mu = -Double.MAX_VALUE;
				int yp = 0;
				for (int c = 0; c < nClasses; ++c) {
					DataPoint wc = (DataPoint) w.sub(c * f, (c + 1) * f);
					double dp = wc.scalarProduct(p);
					double mu = Math.exp(dp) / norm_coeff;
					mu /= norm_coeff;
					if (max_mu < mu) {
						max_mu = mu;
						yp = c;
					}
				}
				yp += conf.c0;
				
				
				if (yp != (int) y) {
					++loss;
				}
			}
			loss /= data.size();
			break;
		}
		}
	return loss;
	}
	
	public double computeLoss(DataPoint w) {
		double loss = 0;
		if(data == null) {
			return -1;
		}

		switch (conf.lossType) {
		case BINARY_SVM: {
			// loss =  \sum_x log[1 + exp(−ywTx)]/n + (lambda/2 * ||w||^2)
			for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				int y = (int) p.getLabel();
				loss += Math.log(1 + Math.exp(-1*y*p.scalarProduct(w))); 
			}
			loss /= data.size();
			loss += Math.pow(w.getNorm(),2)*lambda/2;
			break;
		}
		case REGRESSION: {
			// use squared loss: (1/n) * (w^T*x - y)^2
			for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				double y = p.getLabel();
				double t = (w.scalarProduct(p) - y);
				loss += t*t;
			}
			loss /= data.size();
			break;
		}
		case MULTICLASS_REGRESSION: {

			int f = conf.featureDim;
			SparsePoint w_n = (SparsePoint) w.normalize();

			double log_likelihood = 0;
			
			for (Iterator<DataPoint> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				int y = (int) p.getLabel() - conf.c0;
				
				double norm_coeff = 0;
				for (int c = 0; c < conf.nClasses; ++c) {
					DataPoint wc = (DataPoint) w_n.sub(c * f, (c + 1) * f);
					double dp = wc.scalarProduct(p);
					norm_coeff += Math.exp(dp);
				}
				
				if(norm_coeff != 0) {
					norm_coeff = Math.log(norm_coeff);
					
					DataPoint wy = (DataPoint) w_n.sub(y * f, (y + 1) * f);
					double dp = wy.scalarProduct(p);
					
					log_likelihood += dp - norm_coeff;
				}
				
			}
			loss = log_likelihood; // minimize loss
			
			break;
		}
		}
		return loss;
    }
	@Override
	public void optimize(double tol) {
		Random generator = utils.getGenerator();
		if(conf.initType == Config.InitType.RANDOM) {
			SparsePoint p = (SparsePoint) data.get(0);
			for (int i : p.featureSet()) {
				w.set(i, generator.nextDouble());
			}
		}
		int k = 0;
		while (true) {
			DataPoint mu = computeAverageGradient();
			if (mu != null) {
				DataPoint new_w = (DataPoint) w.subtract(mu.multiply(eta));
				double tol0 = (new_w.subtract(w)).getNorm();
				if(tol0<tol){
					break;
				}
				w = new_w;
				double obj = computeObjective();
				obj_trace.add(obj);
				if(verbose){ 
					
					System.out.println("iteration:"+k+","+obj+",tol0:"+tol0);
				}
				k++;
			}
			
		}
	}
}
