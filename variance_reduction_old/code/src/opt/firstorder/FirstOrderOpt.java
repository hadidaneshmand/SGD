package opt.firstorder;

import opt.loss.Loss;
import data.DataPoint;
import data.DensePoint;

public abstract class FirstOrderOpt {
	protected Loss loss; 
	protected DataPoint w; 
	protected double step_size = 0.8; 
	protected double num_computed_gradients = 0;
	protected double time;
	protected String name; 
	public abstract void setName();
	protected double c_1 = 0.0001;
	protected double c_2 = 0.92;
	protected double beta = 0.8; // for searching the 
	protected int max_itr = 20;
	protected double stepSize_LineSearch = 0.001; 
	protected int numb_iter_linesearch = 10; 
	protected double alpha_line_search = 0.1; 
	public void setName(String name){
		this.name = name; 
	}
	public abstract void iterate_once();
	
	public int getDataSize(){
		return loss.getDataSize(); 
	}
	public void one_pass(){
	   Iterate(1);
	}
	public FirstOrderOpt(Loss loss) {
		this.setLoss(loss);
		w = new DensePoint(loss.getDimension());
		for (int i =0;i<loss.getDimension();i++) {
			w.set(i, 0.0);
		}
//		setName(); 
	}
	public double exact_line_search(DataPoint p_k){ 
		double alpha = 0.1; 
		DataPoint g = (DataPoint) w.add(p_k.multiply(alpha));
		while(Double.isInfinite(loss.computeLoss(g))){ 
			alpha = alpha*beta; 
			g = (DataPoint) w.add(p_k.multiply(alpha));
		}
		for(int i = 0; i< numb_iter_linesearch; i++){ 
			alpha = alpha - stepSize_LineSearch*p_k.scalarProduct(loss.getAverageGradient(g));
			System.out.println("loss(g):"+loss.computeLoss(g));
			System.out.println("alpha:"+alpha);
		}
		return alpha; 
	}
	public double wolfe_condition_step(DataPoint p_k, DataPoint gradient){ 
		double alpha = 1; 
		DataPoint w_new = (DataPoint) w.add(p_k.multiply(alpha)); 
	    double f_new = getLoss().computeLoss(w_new);
	    if(Double.isInfinite(f_new)){ 
	    	for(int i=0;i<10;i++){ 
	    		alpha = alpha*0.1; 
	    		w_new = (DataPoint) w.add(p_k.multiply(alpha)); 
	    		f_new = getLoss().computeLoss(w_new);
	    		System.out.println("alpha:"+alpha+",f_new:"+f_new+",f:"+loss.computeLoss(w));
	    		if(!Double.isInfinite(f_new)){
	    			break; 
	    		}
	    	} 
	    }
	    DataPoint g_new = getLoss().getAverageGradient(w_new);
	    double f = getLoss().computeLoss(w); 
	    double condition_II_one_side =getC_2()*p_k.scalarProduct(gradient);
	    int c = 0; 
//	    System.out.println("==========check wolf conditions=============");
//	    System.out.println("condition_II_one_side:"+condition_II_one_side+",otherside:"+p_k.scalarProduct(g_new) );
	    while(true){
	    	if(f_new <= f+getC_1()*alpha*gradient.scalarProduct(p_k) && p_k.scalarProduct(g_new) >= condition_II_one_side){
	    		break; 
	    	}
	    	if(c>getMax_itr()){ 
	    		return loss.getLambda(); 
	    	}
	    	c++;
	    	System.out.println("======iteration:"+c+",alpha:"+alpha);
	    	alpha = alpha*getBeta(); 
	    	w_new = (DataPoint) w.add(p_k.multiply(alpha)); 
	    	f_new = getLoss().computeLoss(w_new);
	    	g_new = getLoss().getAverageGradient(w_new);
	    	System.out.println("f:"+f);
	    	System.out.println("fnew:"+f_new+",otherside:"+(f+getC_1()*alpha*gradient.scalarProduct(p_k)));
	    	System.out.println("condition_II_one_side:"+condition_II_one_side+",otherside:"+p_k.scalarProduct(g_new) );
	    }
	    return alpha;
	}
	public double backtracking_line_search(DataPoint direction){ 
		System.out.println("Doing Line Search!");
		double out = 1.0; 
		double f = loss.computeLoss(w); 
		DataPoint w_new = (DataPoint) w.add(direction.multiply(out)); 
		double f_new = loss.computeLoss(w_new); 
		DataPoint gradient = loss.getAverageGradient(w); 
		while(f_new > (f + alpha_line_search*out*gradient.scalarProduct(direction))){ 
			out = out*beta; 
			w_new = (DataPoint) w.add(direction.multiply(out)); 
			f_new = loss.computeLoss(w_new);
		}
		return out; 
	}
	
	
	public void Iterate(int stepNum){
		for(int i=0;i<stepNum;i++){ 
			long startT = System.currentTimeMillis(); 
			iterate_once();
			long endT = System.currentTimeMillis(); 
			long detlaM = (endT-startT); 
			double deltaT = detlaM/1000.0; 
			update_iterations(deltaT);
		}
	}
	public String getName(){
		if(name == null){
			setName();
		}
		return name; 
	}
	public void setParam(DataPoint w){ 
		this.w = w; 
	}
	public DataPoint getParam(){ 
		return this.w; 
	}
	public double getStepSize() {
		return step_size;
	}
	public void setStepSize(double step_size) {
		this.step_size = step_size;
	}
	public abstract FirstOrderOpt clone_method();
	
	public DataPoint clone_w(){ 
		DataPoint w_past = new DensePoint(getLoss().getDimension());
		for(int i=0;i<getLoss().getDimension();i++){ 
			w_past.set(i, w.get(i));
		}
		return w_past;
	}

	public double getNum_computed_gradients() {
		return num_computed_gradients;
	}

	public void setNum_computed_gradients(double num_computed_gradients) {
		this.num_computed_gradients = num_computed_gradients;
	}

	public Loss getLoss() {
		return loss;
	}

	public void setLoss(Loss loss) {
		this.loss = loss;
	}
	public double getC_1() {
		return c_1;
	}
	public void setC_1(double c_1) {
		this.c_1 = c_1;
	}
	public double getC_2() {
		return c_2;
	}
	public void setC_2(double c_2) {
		this.c_2 = c_2;
	}
	public double getBeta() {
		return beta;
	}
	public void setBeta(double beta) {
		this.beta = beta;
	}
	public int getMax_itr() {
		return max_itr;
	}
	public void setMax_itr(int max_itr) {
		this.max_itr = max_itr;
	}
	
	public void update_iterations(double time){
		this.time+= time;
		num_computed_gradients+= getLoss().getDataSize(); 
	}
	public double getTime() {
		return time;
	}
	public void setTime(double time) {
		this.time = time;
	}
	
}
