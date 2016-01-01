package plot;
import java.awt.BorderLayout;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

 
public class XYLineChart extends JFrame {
	 String chartTitle;
     String xAxisLabel;
     String yAxisLabel;
     List<Double> series;
     String seriesname;
    public XYLineChart(List<Double> series, String seriesname, String title, String xAxisLabel, String yAxisLabel) {
        super(title);
        chartTitle = title; 
        this.xAxisLabel = xAxisLabel; 
        this.yAxisLabel = yAxisLabel;
        this.series = series; 
        this.seriesname = seriesname;
        JPanel chartPanel = createChartPanel();
        add(chartPanel, BorderLayout.CENTER);
 
        setSize(640, 480);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
    }
 
    private JPanel createChartPanel() {
       
     
        XYDataset dataset = createDataset();
     
        JFreeChart chart = ChartFactory.createXYLineChart(chartTitle,
                xAxisLabel, yAxisLabel, dataset);
        return new ChartPanel(chart);
    }
 
    private XYDataset createDataset() {
        XYSeriesCollection dataset = new XYSeriesCollection();
        XYSeries series1 = new XYSeries(seriesname);
        for(int i = 0; i< series.size();i++){ 
        	series1.add(i,series.get(i)); 
        }
        
     
        dataset.addSeries(series1);
     
        return dataset;
    }
 
    public static void main(String[] args) {
        
    }
}