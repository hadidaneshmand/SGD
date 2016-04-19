package opt;

public class SingleSampleSize extends IndexStrategy {
	int initialItrs; 
	int T; 
	public SingleSampleSize(int total, int initalSampleSize,int initialItrs) {
		super(total);
		this.initialItrs = initialItrs; 
		this.ss = initalSampleSize; 
		T = 0; 
	}
	@Override
	public int Tack() {
		T++;
		if(T>initialItrs){ 
			ss = n; 
		}
		return ss;
	}
	@Override
	public SampleSizeStrategy clone_strategy() {
		SingleSampleSize out = new SingleSampleSize(n, ss, initialItrs); 
		out.inds = this.inds; 
		out.T = T; 
		out.initialItrs = this.initialItrs; 
		return out;
	}
	@Override
	public String getName() {
		return "single-ss";
	}

}
