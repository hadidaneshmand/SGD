package stat;

public class RandomVar {
	    private double relativeProb;
	    private int index;
	    public RandomVar(int index, double relativeProb) {
	    	this.relativeProb = relativeProb; 
	    	this.index = index; 
	    }	
		public double getRelativeProb() {
			return relativeProb;
		}
		public void setRelativeProb(double relativeProb) {
			this.relativeProb = relativeProb;
		}
		public int getIndex() {
			return index;
		}
		public void setIndex(int index) {
			this.index = index;
		} 
}
