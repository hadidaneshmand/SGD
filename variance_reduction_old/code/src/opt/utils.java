package opt;
/**
 * Implementation of various stochastic optimization methods
 * 
 * @author Aurelien Lucchi
 *
 */

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import data.DataPoint;
import data.IOTools;
import data.Point;


public class utils {

	static Random generator = null;
	
	/*
	 * Generate k samples from 0 to n
	 */
	public static List<Integer> getRandomSamples(int k, int n) {
		List<Integer> list_samples = new ArrayList<Integer>();
		
		if(k == n) {
			for(int i = 0; i < n; ++i) {
				list_samples.add(i);
			}
		} else {
			Random randomGenerator = getGenerator();
			for(int i = 0; i < k; ++i) {
				int idx = randomGenerator.nextInt(n);
				list_samples.add(idx);
			}
		}
		return list_samples;
	}
	
	public static Random getGenerator() {
		return new Random();
	}
	
	public static DataPoint loadOptFromFile(String filename) {
		int startIndex = 0;
		List<Point> list_points = IOTools.readPointsFromFile(filename, startIndex);
		return (DataPoint) list_points.get(1);
	}	
	
	public static void writeToFile(String filename, int value1, Double value2) {
		try {
			PrintWriter out = new PrintWriter(new FileWriter(filename, true));
			out.println(value1 + "\t" + value2);
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}	
	
	public static void writeToFile(String filename, int value) {
		try {
			PrintWriter out = new PrintWriter(new FileWriter(filename, true));
			out.println(value);
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void writeToFile(String filename, String value) {
		try {
			PrintWriter out = new PrintWriter(new FileWriter(filename, true));
			out.println(value);
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void writeToFile(String filename, Double value) {
		try {
			PrintWriter out = new PrintWriter(new FileWriter(filename, true));
			out.println(value);
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
}
