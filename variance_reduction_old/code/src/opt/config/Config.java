package opt.config;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;


public class Config {
	public boolean agressive_step ;
	public boolean doubling; 
	public Config() {
		// Exists only to defeat instantiation.
		
		lossType = LossType.BINARY_SVM;
		algType = Config.AlgType.ALG_ALL;
	}

	public void parseFile(String filename) {

		try {
			agressive_step = false; 
			doubling = false;
			BufferedReader fp = new BufferedReader(new FileReader(filename));
			String line;
			while ((line = fp.readLine()) != null) {
				try {
					StringTokenizer st = new StringTokenizer(line, "=");
					String name = st.nextToken().toLowerCase(); // lower case!!
					String value = st.nextToken();
					switch(name) {
		
						// Warning: only enter words in non-capitalize letters!!
						case "saga_step": 
							switch (value) {
							case "agressive":
								agressive_step = true; 
								break;
	
							default:
								break;
						    }
						case "doubling": 
							switch(value){
							case "true":
								doubling = true;
							}
							
						case "alg_type":
							switch(value) {
					    		case "adagrad":
					    			algType = AlgType.ALG_ADAGRAD;
					    			break;				    			
							    case "gd":
								    algType = AlgType.ALG_GD;
								    break;		    	
								case "saga":
									algType = AlgType.ALG_SAGA;
									break;								    
								case "sgd":
									algType = AlgType.ALG_SGD;
									break;
								case "svrg":
								case "svrg_sample":
									algType = AlgType.ALG_SVRG;
									break;									    	
								default:
									algType = AlgType.ALG_ALL;
									break;										
							}
							break;
					
						case "c0":
							c0 = Integer.parseInt(value);
							break;				
						
						case "companionrateofchange":
							companionRateOfChange = Double.parseDouble(value);
							break;
							
						case "datapath":
							dataPath = value;
							break;

						case "datapath_test":
							dataPath_test = value;
							break;							
							
						case "datapoint_selection":
							switch(value) {
							case "random":
								datapointSelection = DataPointSelection.RANDOM;
								break;
							default:
							case "zero":
								datapointSelection = DataPointSelection.ZERO;
								break;								
							}
							break;
							
						case "eta0":
							eta0 = Double.parseDouble(value);
							break;				
							
						case "featuredim":
							featureDim = Integer.parseInt(value);
							break;					
							
						case "init_type":
							switch(value) {
								case "random":
									initType = InitType.RANDOM;
									break;
								default:
								case "zero":
									initType = InitType.ZERO;
									break;								
							}
							break;									
							
						case "lambda":
							lambda = Double.parseDouble(value);
							break;
							
						case "loss_computation":
							lossComputation = LossComputation.values()[Integer.parseInt(value)];
							break;							
							
						case "loss_type":
							switch(value) {
								case "regression":
									lossType = LossType.REGRESSION;
									break;
								case "multiclass_regression":
									lossType = LossType.MULTICLASS_REGRESSION;
									break;		
								case "binary_svm":
									lossType = LossType.BINARY_SVM;
									break;									
							}
							break;							
							
						case "modeldir":
							modelDir = value;
							break;							
							
						case "logdir":
							logDir = value;
							break;							
							
						case "loggingstep":
							loggingStep = Integer.parseInt(value);
							break;								
							
						case "nclasses":
							nClasses = Integer.parseInt(value);
							break;				

						case "npasses":
							nPasses = Integer.parseInt(value);
							break;							

						case "nsamplesperpass":
							nSamplesPerPass = Integer.parseInt(value);
							break;	
							
						case "ntrials":
							nTrials = Integer.parseInt(value);
							break;	

						case "objtype":
							switch(value) {
								case "loss":
									objType = ObjType.OBJ_LOSS;
									break;
								case "classification_error":
									objType = ObjType.OBJ_CLASSIFICATION_ERROR;
									break;									
								case "dist_opt":
									objType = ObjType.OBJ_DIST_TO_OPTIMUM;
									break;			
							}
							break;							
							
						case "optfilename":
							optFilename = value;
							break;
							
						case "randomizevalidationset":
							randomizeValidationSet = Integer.parseInt(value) == 1;
							break;
							
						case "runtraining":
							runTraining = Integer.parseInt(value) == 1;
							break;						

						case "sampling_strategy":
							switch(value) {
								case "linear_growth":
									samplingStrategy = SamplingStrategy.SVRG_LINEAR_GROWTH_SAMPLING;
									break;
								default:
								case "constant":
									samplingStrategy = SamplingStrategy.SVRG_CST_SAMPLING;
									break;										
							}
							break;								
							
						case "startindex":
							startIndex = Integer.parseInt(value);
							break;
						
						case "svrg_psamples":
							svrg_pSamples = Double.parseDouble(value);
							break;
							
						case "svrg_outer_psamples":
							svrg_outer_pSamples = Double.parseDouble(value);
							break;
							
						case "t0":
							T0 = Double.parseDouble(value);
							break;
							
						case "validation_set_ratio":
							validation_set_ratio = Double.parseDouble(value);
							break;
							
						case "variance_computation_psamples":
							variance_computation_pSamples = Double.parseDouble(value);
							break;
							
					}
				} catch(NoSuchElementException e) {
					//
				}
			}
			fp.close();
		} catch (IOException e) {
			System.out.println("Could not read from file " + filename + " due to " + e.getMessage());
		}
		
	}
		
	public enum LossComputation {
		// use binary encoding
		COMPUTE_NO_LOSS (0),
		COMPUTE_TRAINING_LOSS (1),
		COMPUTE_VALIDATION_LOSS (2),
		COMPUTE_TEST_LOSS (4),
		COMPUTE_ALL_LOSS (7); // 1+2+4 = all values
		
	    private int numVal = 7;

	    LossComputation(int numVal) {
	        this.numVal = numVal;
	    }

	    public int getNumVal() {
	        return numVal;
	    }
	}
	
	public enum LossType {
		REGRESSION, // square loss
		MULTICLASS_REGRESSION,
		BINARY_SVM
	}
	
	public  LossType lossType = LossType.MULTICLASS_REGRESSION;
	
	public enum AlgType {
		ALG_ADAGRAD,			// Duchi10
		ALG_GD,				// gradient descent
		ALG_SAGA, 				// SAGA
		ALG_SGD, 			// stochastic gradient descent
		ALG_SVRG,		// SVRG
		ALG_ALL 			// run all
	}
	
	public enum SamplingStrategy {
		SVRG_CST_SAMPLING,
		SVRG_LINEAR_GROWTH_SAMPLING // increase number of sampled points at each iteration
	}
	
	public  AlgType algType = AlgType.ALG_SGD; 

	public enum InitType {
		ZERO, RANDOM
	}

	public enum DataPointSelection {
		ZERO, RANDOM
	}
	
	public enum ObjType {
		OBJ_LOSS,
		OBJ_CLASSIFICATION_ERROR,
		OBJ_DIST_TO_OPTIMUM
	}
	
	public  int c0 = 0;

	public  double companionRateOfChange = 0.5;
	
	public  DataPointSelection datapointSelection = DataPointSelection.RANDOM;
	
	public  double eta0 = 1e-3;	
	
	public  String dataPath = "";
	
	public  String dataPath_test = "";
	
	public  int featureDim = 13;
	
	public  InitType initType = InitType.ZERO;
	
	public  double lambda = 0; // regularizer coefficient
	
	public  String logDir = "";
	
	// defines the number of steps between each recording of the loss
	public  int loggingStep = 10000;
	
	public  LossComputation lossComputation = LossComputation.COMPUTE_ALL_LOSS;
	
	public  String modelDir = "model/";
	
	public  int nClasses = 2;
	
	public  int nPasses = 50;
	
	public  int nSamplesPerPass = 0;
	
	public  int nTrials = 1;
	
	public  ObjType objType = ObjType.OBJ_LOSS;
	
	public  String optFilename = "";
	
	public  boolean randomizeValidationSet = true;
	
	public  boolean runTraining = true;
	
	public  SamplingStrategy samplingStrategy = SamplingStrategy.SVRG_CST_SAMPLING;
	
	// index of the first feature in the libsvm file format
	public  int startIndex = 1;
	
	// opt loss value
	public  double T0 = -1;
	
	public  double variance_computation_pSamples = 1.0;
	
	public  double svrg_pSamples = 1.0;
	
	// percent of samples for the inner loop
	public  double svrg_outer_pSamples = 1.0;
	
	public  double validation_set_ratio = 0.1;
}
