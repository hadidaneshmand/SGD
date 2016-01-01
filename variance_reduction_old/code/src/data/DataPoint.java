
package data;

public abstract class DataPoint extends Point {
	
	private double label = 0; // a possible label of the point, any double value
	
	public double getLabel() {
		return label;
	}
	public void setLabel(double label) {
		//if (binaryLabel != 1 && binaryLabel != -1)
		//	System.out.println("binaryLabel should only be set to 1 or -1, other values are not permitted.");
		this.label = label;
	}
}
