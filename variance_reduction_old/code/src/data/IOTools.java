package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.StringTokenizer;
import java.util.Vector;

/**
 * Input and Output functions for classification data from machine learning and
 * SVM related stuff. Contains a reader for the SVMlight data format.
 * 
 * @author Martin
 * @author Dave
 * 
 */
public class IOTools {

	/**
	 * Generates a list of random, unlabeled points.
	 * 
	 * @param numPoints
	 * @param dimension
	 * @return
	 */
	public static List<Point> generateRandomPoints(int numPoints, int dimension) {
		System.out.println("Generate "+numPoints+" random points in dimension "+dimension+".");
		return DensePoint.randomPoints(numPoints, dimension);
	}

	/**
	 * Generates a list of random, labeled points, one class being from the positive orthant, the other being from the negative one.
	 * 
	 * @param numPoints
	 * @param dimension
	 * @return
	 */
	public static List<Point> generateRandomTwoPolytopes(int numPoints, int dimension) {
		System.out.println("Generate "+numPoints+" random two polyopes (one in the positive orthant, one in the negative one) in dimension "+dimension+".");
		return DensePoint.randomPointsTwoPolytope(numPoints, dimension);
	}
	
	/**
	 * Read a set of points from standard input. the first line must represent
	 * the dimension (number of features) of each point to follow. The resulting
	 * points will have NO LABELS!
	 * 
	 * @param args
	 * @return
	 */
	public static List<Point> readPointsFromStandardInput(String[] args) {
		List<Point> points = new ArrayList<Point>();
		if (args.length > 0) { // read dimension and points from standard input
			int dimension = Integer.valueOf(args[0]);
			int i = 1;
			while (i < args.length) {
				Double[] p = new Double[dimension];
				for (int d = 0; d < dimension; d++)
					p[d] = Double.valueOf(args[i++]);
				points.add(new DensePoint(p));
			}
			System.out.println("Read " + points.size()
					+ " points of dimension "+dimension+" from standard input. \n");
		} else { // generate random points
			System.out.println("Could not read from standard input. No points given? First row should give the dimension of the points!");
		}
		return points;
	}
	
	/**
	 * Read a set of labeled points from a text file, in SVMlight format.
	 * Each row of the text file represents one data-point. The first entry is the
	 * label, and following entries are always feature:value pairs, each separated by a
	 * colon.
	 * Tailing comments '#' are not supported at the moment.
	 * 
	 * @param filename
	 * @return
	 */
	public static List<Point> readPointsFromFile(String filename, int startIndex) {
		List<Point> points = new ArrayList<Point>();

		try {
			BufferedReader fp = new BufferedReader(new FileReader(filename));
			String line;
			while ((line = fp.readLine()) != null) {
				try {
					DataPoint point = new SparsePoint();
					StringTokenizer st = new StringTokenizer(line, " +\t\n\r\f:");
					double label = Double.valueOf(st.nextToken());					// label has to be at the first position of the text row
					point.setLabel(label);

					while (st.hasMoreTokens()) {
						int feature = Integer.valueOf(st.nextToken()) - startIndex;
						double value = Double.valueOf(st.nextToken());
						point.set(feature, value);
					}
					
					points.add(point);
					
				} catch (NumberFormatException e) {
					System.out.println("Could not read datapoint number "+points.size() + " since Line "+line+" seems to be not properly formatted: "+e.getMessage());
				}
			}
			fp.close();
		} catch (IOException e) {
			System.out.println("Could not read from file " + filename + " due to " + e.getMessage());
		}
		return points;
	}
	
	public static List<DataPoint> readDataPointsFromFile(String filename, int startIndex) {
		List<DataPoint> points = new ArrayList<DataPoint>();

		try {
			BufferedReader fp = new BufferedReader(new FileReader(filename));
			String line;
			while ((line = fp.readLine()) != null) {
				try {
					DataPoint point = new SparsePoint();
					StringTokenizer st = new StringTokenizer(line, " +\t\n\r\f:");
					double label = Double.valueOf(st.nextToken());					// label has to be at the first position of the text row
					point.setLabel(label);

					while (st.hasMoreTokens()) {
						int feature = Integer.valueOf(st.nextToken()) - startIndex;
						double value = Double.valueOf(st.nextToken());
						point.set(feature, value);
					}
					
					points.add(point);
					
				} catch (NumberFormatException e) {
					System.out.println("Could not read datapoint number "+points.size() + " since Line "+line+" seems to be not properly formatted: "+e.getMessage());
				}
			}
			fp.close();
		} catch (IOException e) {
			System.out.println("Could not read from file " + filename + " due to " + e.getMessage());
		}
		return points;
	}

	public static List<Point> readDensePointsFromFile(String filename, int startIndex, boolean addOffset, int m) {
		List<Point> points = new ArrayList<Point>();

		try {
			BufferedReader fp = new BufferedReader(new FileReader(filename));
			String line;
			while ((line = fp.readLine()) != null) {
				try {
					DataPoint point = new DensePoint(m);
					StringTokenizer st = new StringTokenizer(line, " +\t\n\r\f:");
					double label = Double.valueOf(st.nextToken());					// label has to be at the first position of the text row
					point.setLabel(label);

					while (st.hasMoreTokens()) {
						int feature = Integer.valueOf(st.nextToken()) - startIndex;
						double value = Double.valueOf(st.nextToken());
						point.set(feature, value);
					}
					if(addOffset) {
						point.set(m-startIndex, 1); // add constant value
					}
					
					points.add(point);
					
				} catch (NumberFormatException e) {
					System.out.println("Could not read datapoint number "+points.size() + " since Line "+line+" seems to be not properly formatted: "+e.getMessage());
				}
			}
			fp.close();
		} catch (IOException e) {
			System.out.println("Could not read from file " + filename + " due to " + e.getMessage());
		}
		return points;
	}
	
	
	/**
	 * Writes a list of points to disk, in SVMlight format
	 * 
	 * @param filename
	 * @param points
	 */
	public static void writePointsToFile(String filename, List<Point> points) {
		System.out.println("Trying to write the points to disk as a textfile in SVMlight format. filename: "+filename);
		try {
			FileWriter fr = new FileWriter(filename);
			PrintWriter pr = new PrintWriter(fr);
			
			for (Point point : points) {
				//it's stupid, but features have to be in ascending order for LIBSVM to work.
				SparsePoint sp = (SparsePoint)point;
				List<Integer> features = new ArrayList<Integer>(sp.featureSet());
				Collections.sort(features);
				
				pr.print(sp.getLabel()+" ");
				for (Integer f : features)
					pr.print(f+":"+sp.get(f)+" ");
				pr.println();
			}

			pr.flush();
			pr.close();
			System.out.println("Successfully wrote "+points.size()+" points to file "+filename);
		} catch (IOException e) {
			System.out.println("Could not write to file "+filename+" due to "+e.getMessage());
		}
	}

	/**
	 * Read a set of labeled points from a text file, but ignores every 10th
	 * point. Those points are returned separately as the testing points.
	 * 
	 * @param filename
	 * @return a partition of all points into two lists of points, the training
	 *         points and the test points
	 */
	public static Vector<List<Point>> readPointsFromFileAndExtractTestPoints(String filename) {
		return readPointsFromFileAndExtractTestPoints(filename, 10);
	}
	/**
	 * Read a set of labeled points from a text file, but ignores every
	 * 'testPointsFraction'th point. Those points are returned separately as the
	 * testing points, the others are returned as training points.
	 * 
	 * @param filename
	 * @param testPointsFraction
	 *            the fraction (1/this) of all points is extracted for testing
	 * @return a partition of all points into two lists of points, the training
	 *         points and the test points
	 */
	public static Vector<List<Point>> readPointsFromFileAndExtractTestPoints(String filename, int testPointsFraction) {
		return readPointsFromFileAndExtractTestPoints(filename, testPointsFraction, 0);
	}

	/**
	 * Read a set of labeled points from a text file, but ignores every
	 * 'testPointsFraction'th point. Those points are returned separately as the
	 * testing points, the others are returned as training points.
	 * 
	 * @param filename
	 * @param testPointsFraction
	 *            the fraction (1/this) of all points is extracted for testing
	 * @param testPointsModulus
	 *            that piece of the cake is taken for testing
	 * @return a partition of all points into two lists of points, the training
	 *         points and the test points
	 */
	public static Vector<List<Point>> readPointsFromFileAndExtractTestPoints(String filename, int testPointsFraction, int testPointsModulus) {
		List<Point> allPoints = readPointsFromFile(filename, 0);
		return splitIntoTrainingAndTestPoints(allPoints, testPointsFraction, testPointsModulus);
	}
	
	/**
	 * Splits a given list of points in to training points and test points.
	 * Every 'testPointsFraction'th point is taken as a test point, the others
	 * are returned as training points.
	 * 
	 * @param allPoints
	 *            a list of points
	 * @param testPointsFraction
	 *            the fraction (1/this) of all points is extracted for testing
	 * @param testPointsModulus
	 *            that piece of the cake is taken for testing
	 * @return a partition of all points into two lists of points, the training
	 *         points and the test points
	 */
	public static Vector<List<Point>> splitIntoTrainingAndTestPoints(List<Point> allPoints, int testPointsFraction, int testPointsModulus) {
		List<Point> trainingPoints = new ArrayList<Point>();
		List<Point> testPoints = new ArrayList<Point>();
		for (int i = 0; i < allPoints.size(); i++) {
			Point point = allPoints.get(i);
			if (i % testPointsFraction == testPointsModulus)
				testPoints.add( point );
			else
				trainingPoints.add( point );
		}
		Vector<List<Point>> result = new Vector<List<Point>>();
		result.add(trainingPoints); result.add(testPoints);
		System.out.println("split all "+allPoints.size()+" points into "+trainingPoints.size()+" trainingPoints and "+testPoints.size()+" testPoints.");
		return result;
	}
	
	/**
	 * Read a set of labeled points from a text file and ordering the the points according to their label.
	 * That is first come all points with label 1 followed by the points with label -1.
	 * 
	 * @param filename
	 * 
	 * @return
	 * 				ordered list of points.
	 */
	@Deprecated // use original method, sorting now takes place in TwoPolytopeDistanceProblem.
	public static List<Point> readPointsFromFileTwoPolytopes(String filename) {
		List<Point> points = new ArrayList<Point>();
		List<Point> bluePoints = new ArrayList<Point>();
		List<Point> redPoints = new ArrayList<Point>();

		try {
			BufferedReader fp = new BufferedReader(new FileReader(filename));
			String line;
			while ((line = fp.readLine()) != null) {
				try {
					DataPoint point = new SparsePoint();
					StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
					double label = Double.valueOf(st.nextToken());					// label has to be at the first position of the text row
					point.setLabel(label);

					while (st.hasMoreTokens()) {
						int feature = Integer.valueOf(st.nextToken());
						double value = Double.valueOf(st.nextToken());
						point.set(feature, value);
					}
					if (point.getLabel() == 1)
						bluePoints.add(point);
					else
						redPoints.add(point);
				} catch (NumberFormatException e) {
					System.out.println("Could not read datapoint number "+ points.size() + " since Line "+line+" seems to be not properly formatted: "+e.getMessage());
				}
			}
			fp.close();
		} catch (IOException e) {
			System.out.println("Could not read from file " + filename + " due to " + e.getMessage());
		}
		points.addAll(bluePoints);
		points.addAll(redPoints);
		return points;
	}
	
	/**
	 * Checks if the list of points contains only points of the correct label.
	 * 
	 * @param pointsList
	 * @return
	 * 					Returns true if the first list contains only point of label 1 and
	 * 					the second list only points of label -1.
	 */
	
	public static boolean checkLabels(List<List<SparsePoint>> pointsList){
		List<SparsePoint> firstPoints = pointsList.get(0);
		List<SparsePoint> secondPoints = pointsList.get(1);
		boolean bool = true;
		for (int i = 1; 1 < firstPoints.size(); i++){
			if (firstPoints.get(i).getLabel() == -1)
				bool = false;
		}
		for (int i = 1; 1 < secondPoints.size(); i++){
			if (secondPoints.get(i).getLabel() == 1)
				bool = false;
		}
		return bool;
	}
		
}
