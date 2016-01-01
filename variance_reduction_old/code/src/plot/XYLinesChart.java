package plot;
import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

 
public class XYLinesChart extends JFrame {
	 String chartTitle;
     String xAxisLabel;
     String yAxisLabel;
     List<List<Double>> series;
     List<Double> t;
     List<String> seriesnames;
     JFreeChart chart;
     private static final int X = 4;
     private static final int Y = 6;
     private static final int W = 100;
     private static final int H = 20;
    public XYLinesChart(List<List<Double>> series,List<Double> t, List<String> seriesnames, String title, String xAxisLabel, String yAxisLabel) {
        super(title);
        chartTitle = title; 
        this.xAxisLabel = xAxisLabel; 
        this.yAxisLabel = yAxisLabel;
        this.series = series; 
        this.seriesnames = seriesnames;
        this.t = t; 
        JPanel chartPanel = createChartPanel();
        add(chartPanel, BorderLayout.CENTER);
        
        setSize(640, 480);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
    }
 
    private JPanel createChartPanel() {
       
     
        XYDataset dataset = createDataset();
     
        chart = ChartFactory.createXYLineChart(chartTitle,
                xAxisLabel, yAxisLabel, dataset);
        
        XYPlot xy = chart.getXYPlot(); 
        XYItemRenderer xyir = xy.getRenderer();
        for(int i=0;i<series.size();i++){ 
        	
			xyir.setSeriesStroke(i, new BasicStroke((float) 2.5));
        }
        
        return new ChartPanel(chart);
    }
 
    private XYDataset createDataset() {
        XYSeriesCollection dataset = new XYSeriesCollection();
        for(int i =0;i<series.size();i++){ 
        	XYSeries series_i = new XYSeries(seriesnames.get(i));
            for(int j = 0;j< series.get(i).size();j++){ 
            	series_i.add(t.get(j),(series.get(i)).get(j)); 
            }
            dataset.addSeries(series_i);
        }
        
        
     
        return dataset;
    }
 
    public static void main(String[] args) {
        
    }
    public void save(File chartfile,int width,int height){ 
    	 
        try {
			ChartUtilities.saveChartAsJPEG( chartfile ,1, chart , width , height );
		} catch (IOException e) {
			e.printStackTrace();
		}
        
       
    }
    public void save(String chartfile,int width,int height){ 
   	 
        try {
			ChartUtilities.saveChartAsJPEG( new File(chartfile+".JPEG") ,1, chart , width , height );
		} catch (IOException e) {
			e.printStackTrace();
		}
        File textfile = new File(chartfile+".txt");
        try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(textfile));
			 for(int i=0;i<series.size();i++){ 
		        	bw.write(seriesnames.get(i));
		        	List<Double> serie = series.get(i);
		        	for(int j=0;j<serie.size();j++){ 
		        		bw.write(" "+serie.get(i));
		        	}
		        	bw.write("\n");
		     }
			 bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		} 
       
    }
}