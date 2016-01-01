/**
 * Implementation of various stochastic optimization methods
 * 
 * @author Aurelien Lucchi
 *
 */

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import data.DataPoint;
import data.IOTools;
import data.Point;
import data.SparsePoint;

public class optimize {

	public static class Data_param {

		Data_param() {
		}

		List<Point> data;
		int f; // dimension of the original features
		int m; // dimension of the features*number of class
		int nClasses;
		
		int size() { return data.size(); }
	}

	public static void main(String[] args) throws IOException {

		String configFilename = "config/config_heart.txt";
		if(args.length > 0) {
			configFilename = args[0];
		}
		Config.parseFile(configFilename);
		
		System.out.println("algType: " + Config.algType.toString());
		System.out.println("eta0: " + Config.eta0);
		System.out.println("datapointSelection " + Config.datapointSelection);
		System.out.println("lambda: " + Config.lambda);
		System.out.println("lossType: " + Config.lossType);
		System.out.println("nTrials: " + Config.nTrials);
		System.out.println("nPasses: " + Config.nPasses);
		System.out.println("nSamplesPerPass: " + Config.nSamplesPerPass);
		System.out.println("objType: " + Config.objType);
		System.out.println("randomizeValidationSet: " + Config.randomizeValidationSet);
		System.out.println("validation_set_ratio: " + Config.validation_set_ratio);	
		System.out.println("svrg_outer_pSamples: " + Config.svrg_outer_pSamples);
		System.out.println("svrg_pSamples: " + Config.svrg_pSamples);
		
		boolean run_training = Config.runTraining;
		
		if(run_training) {
			System.out.println("Training");
		} else {
			System.out.println("Testing");
		}
		
		boolean addOffset = false;

		new File(Config.logDir).mkdir();
		
		// Load training set		
		Data_param data_param = new Data_param();

		data_param.f = Config.featureDim;		
		if (addOffset) {
			++data_param.f;
		}
		data_param.m = data_param.f;
		data_param.nClasses = Config.nClasses;
		if (Config.lossType == Config.LossType.MULTICLASS_REGRESSION) {
			data_param.m = data_param.f * Config.nClasses;
		}
				
		List<Point> data = IOTools.readPointsFromFile(Config.dataPath, Config.startIndex);
		int n = data.size();
		System.out.println("Loaded " + n + " points from " + Config.dataPath);

		if (Config.lossType == Config.LossType.REGRESSION) {
			// sample 100 points and make sure their labels is either +1 or -1
			int inc = n/100;
			for(int k = 0; k < n; k += inc) {
				DataPoint d = (DataPoint) data.get(k);
				if(d.getLabel() != -1 && d.getLabel() != 1) {
					System.out.println("Error: datapoints should have labels +1/-1 for regression");
					System.exit(-1);
				}
			}
		}
		
		// create list of samples
		List<Integer> indices = new ArrayList<Integer>(n);
		for(int i = 0; i < n; ++i) {
			indices.add(i, i);
		}
		
		if(Config.randomizeValidationSet) {
			System.out.println("Shuffling data to create training and validation set");
			// shuffle list of samples to split the data into a training and validation set
			Random random = new Random();
			Collections.shuffle(indices, random);
		}
		
		List<Point> training_data = new ArrayList<Point>();
		List<Point> val_data = new ArrayList<Point>();
		int n_val = (int)(n*Config.validation_set_ratio);
		for(int i = 0; i < n_val; ++i) {
			val_data.add(i, data.get(indices.get(i)));
		}
		for(int i = n_val; i < n; ++i) {
			training_data.add(i-n_val, data.get(indices.get(i)));
		}
		data_param.data = training_data;
		System.out.println("Use " + n_val + " points for cross-validation and " + (n-n_val) + " for training");

		// set nSamplesPerPass if not set in the config file
		if(Config.nSamplesPerPass == 0) {
			Config.nSamplesPerPass = training_data.size();
		}
		
		// Cross-validation set
		Data_param data_param_val = new Data_param();
		data_param_val.f = data_param.f;
		data_param_val.m = data_param.m;
		data_param_val.nClasses = data_param.nClasses;
		data_param_val.data = val_data;		
		
		// Load test set
		Data_param data_param_test = new Data_param();
		data_param_test.f = data_param.f;
		data_param_test.m = data_param.m;
		data_param_test.nClasses = data_param.nClasses;
		data_param_test.data = IOTools.readPointsFromFile(Config.dataPath_test, Config.startIndex);
		System.out.println("Loaded " + data_param_test.data.size() + " points from " + Config.dataPath_test);

		if(Config.objType == Config.ObjType.OBJ_DIST_TO_OPTIMUM) {
			File optFile = new File(Config.optFilename);
			if(!optFile.exists() || optFile.isDirectory()) {
				utils.writeToFile("errors.txt", "File " + Config.optFilename + " does not exist or is a directory.");
				System.exit(1);
			}
		}
		
		long startTime = System.nanoTime();

		if(run_training) {
					
			new File(Config.modelDir).mkdir();			
			
			if(Config.algType == Config.AlgType.ALG_ADAGRAD || Config.algType == Config.AlgType.ALG_ALL) {
				
				System.out.println("Running Adagrad");
				
				List<DataPoint> list_weight_vectors = new ArrayList<DataPoint>();
				int best_model = 0;
				double min_loss = -1;
				
				for (int i = 0; i < Config.nTrials; ++i) {
					
						DataPoint w = adagrad(data_param, i, Config.logDir, "adagrad", data_param_val, data_param_test);
						list_weight_vectors.add(w);
						
						double loss = computeObjective(data_param, w);
						System.out.println("Loss = " + loss);
						double val_loss = computeObjective(data_param_val, w);
						System.out.println("Validation loss SGD = " + val_loss);						
						double test_loss = computeObjective(data_param_test, w);
						System.out.println("Test loss SGD = " + test_loss);

						if((i == 0) || (loss < min_loss)) {
							min_loss = loss;
							best_model = i;
						}
				}
				// TODO: Cross-validation
				DataPoint w = list_weight_vectors.get(best_model);
				w.writeToFile(Config.modelDir + "w_adagrad.txt");
			}
			
			if(Config.algType == Config.AlgType.ALG_GD || Config.algType == Config.AlgType.ALG_ALL) {
				
				System.out.println("Running GD");
				
				List<DataPoint> list_weight_vectors = new ArrayList<DataPoint>();
				int best_model = 0;
				double min_loss = -1;
				
				for (int i = 0; i < Config.nTrials; ++i) {
					
						DataPoint w = GD(data_param, i, Config.logDir, "gd", data_param_val, data_param_test);
						list_weight_vectors.add(w);
						
						double loss = computeObjective(data_param, w);
						System.out.println("Loss = " + loss);
						double val_loss = computeObjective(data_param_val, w);
						System.out.println("Validation loss SGD = " + val_loss);						
						double test_loss = computeObjective(data_param_test, w);
						System.out.println("Test loss SGD = " + test_loss);

						if((i == 0) || (loss < min_loss)) {
							min_loss = loss;
							best_model = i;
						}
				}
				
				// TODO: Cross-validation
				DataPoint w = list_weight_vectors.get(best_model);
				w.writeToFile(Config.modelDir + "w_gd.txt");
			}			
			
			if(Config.algType == Config.AlgType.ALG_SAGA || Config.algType == Config.AlgType.ALG_ALL) {
				
				System.out.println("Running SAGA");
				
				List<DataPoint> list_weight_vectors = new ArrayList<DataPoint>();
				int best_model = 0;
				double min_loss = -1;
				
				for (int i = 0; i < Config.nTrials; ++i) {
					
						DataPoint w = SAGA(data_param, i, Config.logDir, "saga", data_param_val, data_param_test);
						list_weight_vectors.add(w);
						
						double loss = computeObjective(data_param, w);
						System.out.println("Training loss SAGA = " + loss);
						double val_loss = computeObjective(data_param_val, w);
						System.out.println("Validation loss SAGA = " + val_loss);						
						double test_loss = computeObjective(data_param_test, w);
						System.out.println("Test loss SAGA = " + test_loss);

						if((i == 0) || (loss < min_loss)) {
							min_loss = loss;
							best_model = i;
						}
				}
				
				// TODO: Cross-validation
				DataPoint w = list_weight_vectors.get(best_model);
				w.writeToFile(Config.modelDir + "w_saga.txt");
			}
			
			if(Config.algType == Config.AlgType.ALG_SGD || Config.algType == Config.AlgType.ALG_ALL) {
				
				System.out.println("Running SGD");
				
				List<DataPoint> list_weight_vectors = new ArrayList<DataPoint>();
				int best_model = 0;
				double min_loss = -1;
				
				for (int i = 0; i < Config.nTrials; ++i) {
					
						DataPoint w = SGD(data_param, i, Config.logDir, "sgd", data_param_val, data_param_test);
						list_weight_vectors.add(w);
						
						double loss = computeObjective(data_param, w);
						System.out.println("Training loss SGD = " + loss);
						double val_loss = computeObjective(data_param_val, w);
						System.out.println("Validation loss SGD = " + val_loss);						
						double test_loss = computeObjective(data_param_test, w);
						System.out.println("Test loss SGD = " + test_loss);

						if((i == 0) || (loss < min_loss)) {
							min_loss = loss;
							best_model = i;
						}
				}
				
				// TODO: Cross-validation
				DataPoint w = list_weight_vectors.get(best_model);
				w.writeToFile(Config.modelDir + "w_sgd.txt");
			}
			
			
			if(Config.algType == Config.AlgType.ALG_SVRG || Config.algType == Config.AlgType.ALG_ALL) {
				
				System.out.println("Running SVRG");
				
				List<DataPoint> list_weight_vectors = new ArrayList<DataPoint>();
				int best_model = 0;
				double min_loss = -1;
				
				for (int i = 0; i < Config.nTrials; ++i) {
					
						double percentSampledGradients_innerLoop = Config.svrg_pSamples;
						int nSampledGradients_innerLoop = (int) Math.ceil((data_param.data.size()*percentSampledGradients_innerLoop));
						if(Config.svrg_pSamples == -1) {
							nSampledGradients_innerLoop = 1;
						}
					
						String suffix = "svrg";
						DataPoint w = SVRG(data_param, i, Config.logDir, suffix, null, nSampledGradients_innerLoop, data_param_val, data_param_test);
						list_weight_vectors.add(w);
						
						double loss = computeObjective(data_param_test, w);
						System.out.println("Test loss SVRG = " + loss);
						PrintWriter out_loss_sgd = new PrintWriter(new FileWriter(Config.logDir + suffix + "_test_loss" + i + ".txt", true));
						out_loss_sgd.println(loss);
						out_loss_sgd.close();
	
						if((i == 0) || (loss < min_loss)) {
							min_loss = loss;
							best_model = i;
						}
				}
				
				// TODO: Cross-validation
				DataPoint w = list_weight_vectors.get(best_model);
				w.writeToFile(Config.modelDir + "w_svrg.txt");
			}	
			
		} else {

			System.out.println("Loading " + Config.modelDir + "w_sgd.txt");
			List<Point> ws = IOTools.readPointsFromFile(Config.modelDir + "w_sgd.txt", Config.startIndex);
			DataPoint w = (DataPoint) ws.get(0);
			double loss = computeLoss(data_param, w);
			System.out.println("Loss = " + loss);

		}

		long estimatedTime = System.nanoTime() - startTime;
		System.out.println("Elapsed time = " + estimatedTime);
		
	}
	
	/*
	 * SGD
	 */
	public static DataPoint SGD(Data_param data_param, int ntry, String outputDir, String suffix, Data_param val_data_param, Data_param test_data_param) throws IOException {

		int nPasses = Config.nPasses;
		
		List<Point> data = data_param.data;

		DataPoint w = new SparsePoint();

		Random generator = utils.getGenerator();
		if(Config.initType == Config.InitType.RANDOM) {
			SparsePoint p = (SparsePoint) data.get(0);
			for (int i : p.featureSet()) {
				w.set(i, generator.nextDouble());
			}
		}

		w.writeToFile(outputDir + suffix + "_w" + ntry + ".txt");
		
		int r = 0; // count steps until next logging
		int nPointEvaluations = 0;
		
		computeObjectives(data_param, test_data_param, val_data_param,
		 		  w, ntry, outputDir, suffix, nPointEvaluations);
		
		for (int k = 0; k < nPasses; ++k) {

			for (int i = 0; i < Config.nSamplesPerPass; ++i) {
				
				int index = generator.nextInt(data.size());
				DataPoint p = (DataPoint) data.get(index);
				
				DataPoint g = computeStochasticGradient(data_param, p, w);
				++nPointEvaluations;
				++r;
				
				// Select step size
				double eta = Config.eta0;
				if(Config.T0 != -1) {
					// use a decreasing step size
					eta = Config.eta0*Config.T0/((k+1)+Config.T0);				
				}	
				
				w = (DataPoint) w.subtract(g.multiply(eta));
								
				if(r >= Config.loggingStep) {
					r = 0; // reset

					utils.writeToFile(outputDir + suffix + "_eta" + ntry + ".txt", eta);					
					utils.writeToFile(outputDir + suffix + "_norm_w" + ntry + ".txt", w.squaredNorm());
					utils.writeToFile(outputDir + suffix + "_norm_g" + ntry + ".txt", g.squaredNorm());
					
					computeObjectives(data_param, test_data_param, val_data_param,
					 		  w, ntry, outputDir, suffix, nPointEvaluations);
					
				}
				
			}
			
		}

		w.writeToFile(outputDir + suffix + "_w" + ntry + ".txt");

		return w;
	}
	

	/*
	 * Adagrad
	 */
	public static DataPoint adagrad(Data_param data_param, int ntry, String outputDir, String suffix, Data_param val_data_param, Data_param test_data_param) throws IOException {

		int nPasses = Config.nPasses;
		
		List<Point> data = data_param.data;

		DataPoint w = new SparsePoint();

		Random generator = utils.getGenerator();
		if(Config.initType == Config.InitType.RANDOM) {
			SparsePoint p = (SparsePoint) data.get(0);
			for (int i : p.featureSet()) {
				w.set(i, generator.nextDouble());
			}
		}

		w.writeToFile(outputDir + suffix + "_w" + ntry + ".txt");
		
		int r = 0; // count steps until next logging
		int nPointEvaluations = 0;
		
		computeObjectives(data_param, test_data_param, val_data_param,
		 		  w, ntry, outputDir, suffix, nPointEvaluations);
		
		DataPoint mu = new SparsePoint(); // average squared gradient
		
		for (int k = 0; k < nPasses; ++k) {

			for (int i = 0; i < Config.nSamplesPerPass; ++i) {
				
				int index = generator.nextInt(data.size());
				DataPoint p = (DataPoint) data.get(index);
				
				DataPoint g = computeStochasticGradient(data_param, p, w);
				++nPointEvaluations;
				++r;
				
				DataPoint g2 = (DataPoint) g.multiply(g);
				mu = (DataPoint) mu.add(g2);				
				g = (DataPoint) g.divide(mu.sqrt());
				
				//double eta = Config.eta0/(k+1);
				double eta = Config.eta0; // constant step size seems to be doing better
				w = (DataPoint) w.subtract(g.multiply(eta));
				
				if(r >= Config.loggingStep) {
					r = 0; // reset

					utils.writeToFile(outputDir + suffix + "_eta" + ntry + ".txt", eta);
										
					// variance
					//DataPoint v = computeVarianceGradient_adagrad(data_param, w, mu);
					//utils.writeToFile(outputDir + suffix + "_norm_v" + ntry + ".txt", v.squaredNorm());
					
					utils.writeToFile(outputDir + suffix + "_norm_w" + ntry + ".txt", w.squaredNorm());
					utils.writeToFile(outputDir + suffix + "_norm_g" + ntry + ".txt", g.squaredNorm());
					utils.writeToFile(outputDir + suffix + "_norm_mu" + ntry + ".txt", mu.sqrt().squaredNorm());
					
					computeObjectives(data_param, test_data_param, val_data_param,
							 		  w, ntry, outputDir, suffix, nPointEvaluations);
										
				}
				
			}
			
		}

		//w.writeToFile(outputDir + suffix + "_w" + ntry + ".txt");

		return w;
	}

	/*
	 * Compute matrix * gradient
	 * @param coeff_diagonal: coefficient used to initialize the diagonal of the Hessian matrix
	 */
	public static SparsePoint computeLBFGSDescentDirection(int tau, SparsePoint p0, double coeff_diagonal, List<DataPoint> list_s, List<DataPoint> list_y) {
		
		if(list_s.size() < 1) {
			return p0;
		}
		
		int t = list_s.size();
		
		//System.out.println("computeLBFGSDescentDirection, norm(g) = " + p0.squaredNorm());
		
		// initialize descent direction to the gradient
		SparsePoint p = p0;
		
		// Compute sequence of p vectors		
		// iterate through list of correction pairs starting at the most recent index
		List<Double> list_alphas = new ArrayList<Double>(t);
		for(int u = 0; u < t; ++u) {
			DataPoint yu = list_y.get(u);
			DataPoint su = list_s.get(u);
			double rho = 1.0/su.scalarProduct(yu);
			double sp = su.scalarProduct(p);
			double alpha = rho*sp;
			list_alphas.add(u, alpha);
			p = (SparsePoint) p.subtract(yu.multiply(alpha));
			//System.out.println("norm(s) = " + su.squaredNorm() + " norm(y) = " + yu.squaredNorm() + " rho = " + rho  + " norm(p) = " + p.squaredNorm() + " s.p = " + sp + " alpha = " + alpha + " norm(p) = " + p.squaredNorm());
		}
		
		//System.out.println("norm(p) = " + p.squaredNorm());
		
		// rescale p
		DataPoint y = list_y.get(0);
		DataPoint s = list_s.get(0);
		p = (SparsePoint) p.multiply(s.scalarProduct(y)/y.squaredNorm());
		
		// iterate through list of correction pairs starting at the least recent index
		// Compute sequence of q vectors
		// SparsePoint q = (SparsePoint) p.multiply(1.0/coeff_diagonal);
		for(int u = 0; u < t; ++u) {
			DataPoint yu = list_y.get(t-u-1);
			DataPoint su = list_s.get(t-u-1);
			double rho = 1.0/su.scalarProduct(yu);
			//double sp = su.scalarProduct(p);			
			double alpha = list_alphas.get(t-u-1);
			double beta = rho*yu.scalarProduct(p);
			p = (SparsePoint) p.add(su.multiply(alpha-beta));
			// TODO: Check magnitude here!
			//System.out.println("norm(s) = " + su.squaredNorm() + " norm(y) = " + yu.squaredNorm() + " rho = " + rho  + " norm(p) = " + p.squaredNorm() + " s.p = " + sp + " alpha = " + alpha + " norm(p) = " + p.squaredNorm());
			//System.out.println("norm(p) = " + p.squaredNorm());
		}
		
		//System.out.println("norm(q) = " + q.squaredNorm());
		//System.exit(-1);
		
		return p;
	}

	/*
	 * SAGA
	 */
	public static DataPoint SAGA(Data_param data_param, int ntry, String outputDir, String suffix, Data_param val_data_param, Data_param test_data_param) throws IOException {

		int nPasses = Config.nPasses;
		
		List<Point> data = data_param.data;
		int n = data.size();

		DataPoint w = new SparsePoint();

		Random generator = utils.getGenerator();
		if(Config.initType == Config.InitType.RANDOM) {
			SparsePoint p = (SparsePoint) data.get(0);
			for (int i : p.featureSet()) {
				w.set(i, generator.nextDouble());
			}
		}

		w.writeToFile(outputDir + suffix + "_w" + ntry + ".txt");
		
		// allocate memory to store one gradient per datapoint
		SparsePoint[] phi = new SparsePoint[n];
		int nGradients = 0; //number of gradients stored so far
		// average gradient
		SparsePoint avg_phi = new SparsePoint();

		int r = 0; // count steps until next logging
		int nPointEvaluations = 0;
		Config.nSamplesPerPass = Math.min(Config.nSamplesPerPass, n);
		
		computeObjectives(data_param, test_data_param, val_data_param,
		 		  w, ntry, outputDir, suffix, nPointEvaluations);
		
		for (int k = 0; k < nPasses; ++k) {
			
			// create list of samples
			List<Integer> indices = new ArrayList<Integer>(n);
			for(int i = 0; i < n; ++i) {
				indices.add(i, i);
			}
			Random random = new Random();
			Collections.shuffle(indices, random);
			
			for (int i = 0; i < Config.nSamplesPerPass; ++i) {
				
				//int index = generator.nextInt(n);
				int index = indices.get(i);
				DataPoint p = (DataPoint) data.get(index);
				
				// Compute stochastic gradient for p
				DataPoint gp = computeStochasticGradient(data_param, p, w);
				
				// Compute SAGA gradient
				DataPoint g = computeStochasticGradient_SAGA(data_param, p, w, phi, index, avg_phi, gp);
				
				++nPointEvaluations;
				++r;
				
				DataPoint delta_phi = gp;
				if(phi[index] != null) {
					delta_phi = (DataPoint) delta_phi.subtract(phi[index]);
					
					// update average phi gradient
					double a = 1.0/nGradients;
					//double b = (1.0-a);
					//avg_phi = (SparsePoint) delta_phi.multiply(a).add(avg_phi.multiply(b));
					avg_phi = (SparsePoint) delta_phi.multiply(a).add(avg_phi);
										
				} else {
					// new gradient
					
					++nGradients; // increment number of gradients
					
					// update average phi gradient
					double a = 1.0/nGradients;
					double b = (1.0-a);
					avg_phi = (SparsePoint) delta_phi.multiply(a).add(avg_phi.multiply(b));
				}
				
				// store gradient in table phi
				phi[index] = (SparsePoint)gp;
				
				/*
				 	// update average phi gradient
					double a = 1.0/n;
					//double b = (1.0-a);
					//avg_phi = (SparsePoint) delta_phi.multiply(a).add(avg_phi.multiply(b));
					avg_phi = (SparsePoint) delta_phi.multiply(a).add(avg_phi);
					
				 */
				
				//System.out.println("gp = " + gp.squaredNorm());
				//System.out.println("avg_phi = " + n + " " + avg_phi.squaredNorm());		
								
				/*
				// make sure mean is computed correctly
				SparsePoint aphi = new SparsePoint();
				//int n = phi.length;
				for(int j = 0; j < n; ++j) {
					if(phi[j] != null) {
						aphi = (SparsePoint) aphi.add(phi[j]);
					}
				}
				aphi = (SparsePoint) aphi.multiply(1.0/nGradients);
				System.out.println("SAGA diff = " + nGradients + " " + avg_phi.squaredNormOfDifferenceTo(aphi));
				*/

				// Select step size
				double eta = Config.eta0;
				if(Config.T0 != -1) {
					// use a decreasing step size
					eta = Config.eta0*Config.T0/((k+1)+Config.T0);				
				}

				// gradient step
				w = (DataPoint) w.subtract(g.multiply(eta));
								
				if(r >= Config.loggingStep) {
					r = 0; // reset

					utils.writeToFile(outputDir + suffix + "_eta" + ntry + ".txt", eta);
					
					utils.writeToFile(outputDir + suffix + "_norm_w" + ntry + ".txt", w.squaredNorm());
					utils.writeToFile(outputDir + suffix + "_norm_g" + ntry + ".txt", g.squaredNorm());
					utils.writeToFile(outputDir + suffix + "_norm_gp" + ntry + ".txt", gp.squaredNorm());
					utils.writeToFile(outputDir + suffix + "_norm_avg_phi" + ntry + ".txt", avg_phi.squaredNorm());
					
					utils.writeToFile(outputDir + suffix + "_nStoredGradients" + ntry + ".txt", nGradients);
					
					// variance
					//DataPoint v = computeVarianceGradient_SAGA(data_param, w, phi, index, avg_phi);
					//utils.writeToFile(outputDir + suffix + "_norm_v" + ntry + ".txt", Math.sqrt(v.squaredNorm()));
					
					computeObjectives(data_param, test_data_param, val_data_param,
					 		  w, ntry, outputDir, suffix, nPointEvaluations);
					
				}
				
			}
			
		}

		w.writeToFile(outputDir + suffix + "_w" + ntry + ".txt");

		return w;
	}
	
	
	
	
	/*
	 * Stochastic variance reduced gradient (SVRG)
	 * See Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. NIPS 2013.
	 */
	public static DataPoint SVRG(Data_param data_param, int ntry,
			String outputDir, String suffix, DataPoint w_init, int nSampledGradients, Data_param val_data_param, Data_param test_data_param)
			throws IOException {

		int nPasses = Config.nPasses;
		List<Point> data = data_param.data;
		int m = data_param.m;
		int nPointEvaluations = 0;

		// wes contains all the we's which are the estimated versions of the w's
		List<SparsePoint> wes = new ArrayList<SparsePoint>(nPasses);

		// Initialize we0
		SparsePoint we = new SparsePoint();

		Random generator = utils.getGenerator();
		
		if (w_init != null) {
			for (int i = 0; i < m; ++i) {
				we.set(i, w_init.get(i));
			}
		} else {
			if(Config.initType == Config.InitType.RANDOM) {
				// initialize non-zeros entries of the first data point
				SparsePoint p = (SparsePoint) data.get(0);				
				for (int i : p.featureSet()) {
					we.set(i, generator.nextDouble());
				}
			}
		}
		wes.add(0, we);

		SparsePoint w = new SparsePoint();

		// --------------------------------------------------------------------
		// Use SGD for first iteration
		// Only for logging purposes
		
		// set w0 = we
		for (int i : we.featureSet()) {
			w.set(i, we.get(i));
		}
		
		computeObjectives(data_param, test_data_param, val_data_param,
		 		  w, ntry, outputDir, suffix, nPointEvaluations);
		// --------------------------------------------------------------------
		
		int r = 0; // count steps until next logging

		utils.writeToFile("log.txt", "nSampledGradients = " + nSampledGradients);
		
		for (int s = 0; s < nPasses; ++s) {

			System.out.println("Pass " + s + ", r = " + r + "/" + Config.loggingStep);
			
			// we = we(s-1)
			we = wes.get(s);

			// Compute gradient over all data points at we
			DataPoint mu = computeAverageGradient(data_param, we, nSampledGradients);			
			nPointEvaluations += nSampledGradients;
			r += nSampledGradients;

			//System.out.println("Gradient computed " + s);
			
			// set w0 = we
			for (int i : we.featureSet()) {
				w.set(i, we.get(i));
			}
			
			for (int i = 0; i < Config.nSamplesPerPass; ++i) {
				int index = generator.nextInt(data.size());
				DataPoint p = (DataPoint) data.get(index);

				DataPoint g = computeStochasticGradient_SVRG(data_param, p, w, we, mu);
				double eta = Config.eta0;
				w = (SparsePoint) w.subtract(g.multiply(eta));
				
				nPointEvaluations += 2; // computation of g and ge
				r += 2;
				
				if(r >= Config.loggingStep) {
					r = 0; // reset
										
					utils.writeToFile(outputDir + suffix + "_norm_w" + ntry + ".txt", w.squaredNorm());
					utils.writeToFile(outputDir + suffix + "_norm_mu" + ntry + ".txt", mu.squaredNorm());
					
					// variance
					//DataPoint ga = computeAverageGradient(data_param, w, data.size());
					//DataPoint v = computeVarianceGradient_SVRG(data_param, w, we, ga, mu);
					//utils.writeToFile(outputDir + suffix + "_norm_v" + ntry + ".txt", Math.sqrt(v.squaredNorm()));
															
					computeObjectives(data_param, test_data_param, val_data_param,
					 		  w, ntry, outputDir, suffix, nPointEvaluations);
				}				
			}
			
			if(Config.samplingStrategy == Config.SamplingStrategy.SVRG_LINEAR_GROWTH_SAMPLING) {
				nSampledGradients = (int) Math.min(1.1*nSampledGradients, data.size());
				utils.writeToFile(outputDir + suffix + "_percentSampledGradients" + ntry + ".txt", nSampledGradients);
			}
			
			// Option I
			wes.add(s + 1, w);			
		}

		// write last predictor
		w.writeToFile(outputDir + suffix + "_w" + ntry + ".txt");
		
		return w;
	}
	
	/*
	 * GD
	 */
	public static DataPoint GD(Data_param data_param, int ntry, String outputDir, String suffix, Data_param val_data_param, Data_param test_data_param) throws IOException {

		int nPasses = Config.nPasses;
		
		List<Point> data = data_param.data;

		DataPoint w = new SparsePoint();

		Random generator = utils.getGenerator();
		if(Config.initType == Config.InitType.RANDOM) {
			SparsePoint p = (SparsePoint) data.get(0);
			for (int i : p.featureSet()) {
				w.set(i, generator.nextDouble());
			}
		}

		int r = 0; // count steps until next logging
		int nPointEvaluations = 0;
				
		for (int k = 0; k < nPasses; ++k) {

			DataPoint mu = computeAverageGradient(data_param, w, data.size());
			nPointEvaluations += data.size();
			
			if (mu != null) {
				double eta = Config.eta0/(k+1);
				w = (DataPoint) w.subtract(mu.multiply(eta));
			}
			
			++r;
			if(r >= Config.loggingStep) {
				r = 0; // reset
				
				utils.writeToFile(outputDir + suffix + "_norm_w" + ntry + ".txt", w.squaredNorm());
				utils.writeToFile(outputDir + suffix + "_norm_mu" + ntry + ".txt", mu.squaredNorm());
			
				computeObjectives(data_param, test_data_param, val_data_param,
				 		  w, ntry, outputDir, suffix, nPointEvaluations);
			}
				
			
		}

		w.writeToFile(outputDir + suffix + "_w" + ntry + ".txt");

		return w;
	}
	

	/*
	 * Compute gradient over the whole dataset or a subset if percentSampledGradients < 1.0
	 * @param percentSampledGradients is a number between 0 and 1.0
	 */
	public static DataPoint computeAverageGradient(Data_param data_param,
			DataPoint w, int nSampledGradients) {

		if(nSampledGradients < data_param.data.size()) {
			System.out.println("Computing average gradient over " + nSampledGradients + " data points");
			return computeSampledAverageGradient(data_param, w, nSampledGradients);
		}
		
		List<Point> data = data_param.data;

		DataPoint g = new SparsePoint();
		for (Iterator<Point> iter = data.iterator(); iter.hasNext();) {
			DataPoint p = (DataPoint) iter.next();
			DataPoint gi = computeStochasticGradient(data_param, p, w);
			g = (DataPoint) g.add(gi);
		}
		g = (DataPoint) g.multiply(1.0 / data.size());

		return g;
	}
		
	public static DataPoint computeSampledAverageGradientFromSamples(Data_param data_param,
			DataPoint w, List<Integer> list_samples) {

		List<Point> data = data_param.data;
		DataPoint g = new SparsePoint();
		
		for(int i = 0; i < list_samples.size(); ++i) {
			int idx = list_samples.get(i);
			DataPoint p = (DataPoint) data.get(idx);
			DataPoint gi = computeStochasticGradient(data_param, p, w);
			g = (DataPoint) g.add(gi);
		}
		g = (DataPoint) g.multiply(1.0 / list_samples.size());

		return g;
	}
	
	public static DataPoint computeSampledAverageGradient(Data_param data_param,
			DataPoint w, int k) {

		List<Point> data = data_param.data;
		int n = data.size();
		DataPoint g = new SparsePoint();
		
		Random randomGenerator = new Random();
		for(int i = 0; i < k; ++i) {
			int idx = randomGenerator.nextInt(n);
			DataPoint p = (DataPoint) data.get(idx);
			DataPoint gi = computeStochasticGradient(data_param, p, w);
			g = (DataPoint) g.add(gi);
		}
		g = (DataPoint) g.multiply(1.0 / k);

		return g;
	}

	public static DataPoint computeStochasticGradient(Data_param data_param,
			DataPoint p, DataPoint w) {

		DataPoint g = null;

		switch (Config.lossType) {
		case BINARY_SVM: {
			// loss = max(0, 1-y*w^T*x) + (lambda * ||w||^2)
			double o = p.scalarProduct(w);
			int y = (int) p.getLabel();
			if (y * o < 0) {
				g = (DataPoint) p.multiply(-y);
				g = (DataPoint) g.add(w.multiply(Config.lambda)); // add regularization term lambda*w (cost function is n*lambda*|w|^2)
			} else {
				if(Config.lambda != 0) {
					g = (DataPoint) w.multiply(Config.lambda);
				} else {
					g = new SparsePoint();
				}
			}
			break;
		}
		case REGRESSION: {
			// use squared loss:  \sum (w^T*x - y)^2 + lambda*||w||^2
			double y = p.getLabel();
			g = (DataPoint) p.multiply(2 * (w.scalarProduct(p) - y));
			g = (DataPoint) g.add(w.multiply(Config.lambda)); // add regularizer			
			break;
		}
		case MULTICLASS_REGRESSION: {
			
			int f = data_param.f;
			
			g = new SparsePoint();		

			double y = p.getLabel();
			for (int c = 0; c < Config.nClasses; ++c) {
				
				// compute mu_c = exp(< w_j, p >)/Z
				// where Z = \sum_r exp(< w_r, p >)
				// this is also equal to mu_j = 1/(\sum_{r!=c} exp(< w_r, p >)) 
				DataPoint wc = (DataPoint) w.sub(c * f, (c + 1) * f);
				double dp = wc.scalarProduct(p);				
				double denom = 1.0;
				for (int r = 0; r < Config.nClasses; ++r) {
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
				int yc = (c == (int)(y-Config.c0)) ? 1 : 0; 
				DataPoint gc = (DataPoint) p.multiply(mu - yc);
				gc.add(w.multiply(Config.lambda)); // add regularization term lambda*w (cost function is n*lambda*|w|^2)
				
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

	public static DataPoint computeStochasticGradient_SAGA(Data_param data_param,
			DataPoint p, DataPoint w, SparsePoint[] phi, int index, SparsePoint avg_phi, DataPoint gp) {
		
		DataPoint g = gp;

		if(phi[index] != null) {
			g = (DataPoint) g.subtract(phi[index]);
			
			// subtract average over phi_j
			g = (DataPoint) g.add(avg_phi);
		}
		
		return g;
	}
	
	public static DataPoint computeStochasticGradient_SVRG(Data_param data_param,
			DataPoint p, DataPoint w, DataPoint we, DataPoint mu) {
		DataPoint g = computeStochasticGradient(data_param, p, w);
		DataPoint ge = computeStochasticGradient(data_param, p, we);
		g = (DataPoint) g.add(mu.subtract(ge));
		return g;
	}
	
	public static DataPoint computeStochasticGradient_SVRG(Data_param data_param,
			List<Integer> list_samples, DataPoint w, DataPoint we, DataPoint mu) {
		DataPoint g = computeSampledAverageGradientFromSamples(data_param, w, list_samples);
		DataPoint ge = computeSampledAverageGradientFromSamples(data_param, w, list_samples);
		g = (DataPoint) g.add(mu.subtract(ge));
		return g;
	}

	/*
	 * Compute variance of the stochastic gradients
	 * Only compute diagonal of the covariance matrix
	 * @param ga is the mean
	 */
	public static DataPoint computeVarianceGradient(Data_param data_param,
													DataPoint w, DataPoint ga) {
		
		List<Point> data = data_param.data;
		DataPoint v = new SparsePoint();
		
		// Compute diagonal of the covariance matrix C = E[(g-E[g])^2]
		for (Iterator<Point> iter = data.iterator(); iter.hasNext();) {
			DataPoint p = (DataPoint) iter.next();
			
			// Compute (g_i-E[g_i])^2
			DataPoint gi = computeStochasticGradient(data_param, p, w);
			gi = (DataPoint) gi.subtract(ga);
			DataPoint gi_squared = (DataPoint) gi.multiply(gi);			
			v = (DataPoint) v.add(gi_squared);
		}
		v = (DataPoint) v.multiply(1.0 / data.size());

		return v;
	}
	
	/*
	 * Compute variance of the stochastic gradients
	 * @param mu is the mean
	 */
	public static DataPoint computeVarianceGradient_SVRG(Data_param data_param,
													DataPoint w, DataPoint we, DataPoint ga, DataPoint mu) {
				
		List<Point> data = data_param.data;
		int n = data.size();
		
		int k = (int) Math.ceil((n*Config.variance_computation_pSamples));	
		List<Integer> samples = utils.getRandomSamples(k, n);
		
		DataPoint v = new SparsePoint();		
		for(int i = 0; i < k; i++) {
			DataPoint p = (DataPoint) data.get(samples.get(i));		
			DataPoint gi = computeStochasticGradient_SVRG(data_param, p, w, we, mu);
			gi = (DataPoint) gi.subtract(ga);
			DataPoint gi_squared = (DataPoint) gi.multiply(gi);			
			v = (DataPoint) v.add(gi_squared);
		}
		v = (DataPoint) v.multiply(1.0 / data.size());

		return v;
	}

	public static void computeObjectives(	Data_param data_param, Data_param test_data_param, Data_param val_data_param,
											DataPoint w, int ntry, String outputDir, String suffix, int nPointEvaluations) {
		
		String loss_filename = outputDir + suffix + "_loss" + ntry + ".txt";
		String val_loss_filename = outputDir + suffix + "_val_loss" + ntry + ".txt";
		String test_loss_filename = outputDir + suffix + "_test_loss" + ntry + ".txt";
		
		if((Config.lossComputation.getNumVal() & Config.LossComputation.COMPUTE_TRAINING_LOSS.getNumVal()) != 0) {
			double loss = computeObjective(data_param, w);
			utils.writeToFile(loss_filename, nPointEvaluations, loss);
			
			// Output classification error
			String error_filename = outputDir + suffix + "_error" + ntry + ".txt";
			double error = computeClassificationError(data_param, w);
			utils.writeToFile(error_filename, nPointEvaluations, error);						
		}
		if((Config.lossComputation.getNumVal() & Config.LossComputation.COMPUTE_TEST_LOSS.getNumVal()) != 0) {
			double test_loss = computeObjective(test_data_param, w);
			utils.writeToFile(test_loss_filename, nPointEvaluations, test_loss);
		}
		if((Config.lossComputation.getNumVal() & Config.LossComputation.COMPUTE_VALIDATION_LOSS.getNumVal()) != 0) {
			double val_loss = computeObjective(val_data_param, w);
			utils.writeToFile(val_loss_filename, nPointEvaluations, val_loss);
		}
	}
	
	public static double computeObjective(Data_param data_param, DataPoint w) {
		double obj = 0;
		switch(Config.objType) {
			case OBJ_LOSS: {
				obj = computeLoss(data_param, w);
				break;
			}
			case OBJ_CLASSIFICATION_ERROR: {
				obj = computeClassificationError(data_param, w);
				break;
			}
			case OBJ_DIST_TO_OPTIMUM: {
				obj = computeSquaredDistanceFromOpt(w);
				break;
			}
		}
		return obj;
	}
	
	public static double computeSquaredDistanceFromOpt(DataPoint w) {
		double dist = -1;
		File optFile = new File(Config.optFilename);
		if(optFile.exists() && !optFile.isDirectory()) {
			SparsePoint s = (SparsePoint) utils.loadOptFromFile(Config.optFilename);
			
			SparsePoint w_n = (SparsePoint) w.normalize();
			SparsePoint s_n = (SparsePoint) s.normalize();
			
			dist = w_n.squaredNormOfDifferenceTo(s_n);;
		}
		return dist;
	}
	
	public static double computeLoss(Data_param data_param, DataPoint w) {
		
		double loss = 0;
		if(data_param == null) {
			return -1;
		}
		
		List<Point> data = data_param.data;

		switch (Config.lossType) {
		case BINARY_SVM: {
			for (Iterator<Point> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				int y = (int) p.getLabel();
				loss += Math.max(0, 1 - y*p.scalarProduct(w)); 
			}

			loss /= data.size();
			break;
		}
		case REGRESSION: {
			// use squared loss: (1/n) * (w^T*x - y)^2
			for (Iterator<Point> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				double y = p.getLabel();
				double t = (w.scalarProduct(p) - y);
				loss += t*t;
			}
			loss /= data.size();
			break;
		}
		case MULTICLASS_REGRESSION: {

			int f = data_param.f;
			SparsePoint w_n = (SparsePoint) w.normalize();

			double log_likelihood = 0;
			
			for (Iterator<Point> iter = data.iterator(); iter.hasNext();) {
				DataPoint p = (DataPoint) iter.next();
				int y = (int) p.getLabel() - Config.c0;
				
				double norm_coeff = 0;
				for (int c = 0; c < Config.nClasses; ++c) {
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

	public static double computeClassificationError(Data_param data_param, DataPoint w) {

		double loss = 0;
		if(data_param == null) {
			return -1;
		}
		
		int nClasses = Config.nClasses;
		List<Point> data = data_param.data;

		switch (Config.lossType) {
		case BINARY_SVM: {
			double[] correct = new double[nClasses];
			double[] incorrect = new double[nClasses];
			for (Iterator<Point> iter = data.iterator(); iter.hasNext();) {
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
			for (Iterator<Point> iter = data.iterator(); iter.hasNext();) {
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

			int f = data_param.f;

			for (Iterator<Point> iter = data.iterator(); iter.hasNext();) {
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
				yp += Config.c0;
				
				
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
	
}
