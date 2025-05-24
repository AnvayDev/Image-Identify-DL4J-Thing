package com.anvay.bioinformatics;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
public class FashionMNISTClassifier extends JFrame {
    private ComputationGraph model;
    private final JLabel imageLabel;
    private final JLabel resultLabel;
    private List<String> labels;
    private static final double[] IMAGENET_MEAN = {0.485, 0.456, 0.406};
    private static final double[] IMAGENET_STD = {0.229, 0.224, 0.225};
    public FashionMNISTClassifier() throws Exception {
        setTitle("Image Classifier");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);   setLayout(new BorderLayout());
        initializeModel(); initializeLabels();
        imageLabel = new JLabel("No Image Selected", SwingConstants.CENTER);
        imageLabel.setPreferredSize(new Dimension(200, 200));  add(imageLabel, BorderLayout.CENTER);
        resultLabel = new JLabel("Result will be displayed here", SwingConstants.CENTER);
        resultLabel.setVerticalAlignment(SwingConstants.TOP);
        JScrollPane scrollPane = new JScrollPane(resultLabel);
        scrollPane.setPreferredSize(new Dimension(400, 100));
        add(scrollPane, BorderLayout.SOUTH);
        JButton selectImageButton = new JButton("Select Image");
        selectImageButton.addActionListener(this::selectAndClassifyImage);
        add(selectImageButton, BorderLayout.NORTH);
        setSize(400, 500);
        setLocationRelativeTo(null);
        setVisible(true);
    }
  private void initializeModel() throws Exception {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Select Model File");
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File modelFile = fileChooser.getSelectedFile();
            if (modelFile.exists()) {
                model = ModelSerializer.restoreComputationGraph(modelFile);
            } else {
             throw new Exception("Not a Model File: " + modelFile.getAbsolutePath());
          }
        } 
    private void initializeLabels() {
        labels = new ArrayList<>();
      //fuck
        String[] commonLabels = {
                "idk i couldnt find a text file with the labels till now"
        };try {
            File labelFile = new File("C:\\Users\\super\\Downloads\\imagenet1000_clsidx_to_labels.txt");
            if (labelFile.exists()) {
                try (BufferedReader br = new BufferedReader(new FileReader(labelFile))) { String line;
                    while ((line = br.readLine()) != null) {
                        labels.add(line.trim());
                    }
                }
            } else {
                Collections.addAll(labels, commonLabels);
                for (int i = labels.size(); i < 1000; i++) {
                    labels.add("Class_" + i);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            labels.clear();
            }
        }
    } private void selectAndClassifyImage(ActionEvent e) {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Select Image");
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File imageFile = fileChooser.getSelectedFile();
            try {
                BufferedImage originalImg = ImageIO.read(imageFile);
                BufferedImage preprocessedImg = preprocessImage(originalImg);
                Image scaledImg = preprocessedImg.getScaledInstance(200, 200, Image.SCALE_SMOOTH);
                imageLabel.setIcon(new ImageIcon(scaledImg));

                INDArray input = imageToINDArray(preprocessedImg);
                INDArray[] output = model.output(input);
                INDArray probabilities = output[0];

                // Get top 5 predictions
                int[] topIndices = getTopK(probabilities);
                StringBuilder resultText = new StringBuilder("<html>");
                for (int idx : topIndices) {
                    double probability = probabilities.getDouble(idx);
                    String label = idx < labels.size() ? labels.get(idx) : "Unknown";
                    resultText.append(String.format("%s: %.2f%%<br>", label, probability * 100));
                }
                resultText.append("</html>");
                resultLabel.setText(resultText.toString());
            } catch (Exception ex) {
                ex.printStackTrace();
                JOptionPane.showMessageDialog(this, "Error processing image:\n" + ex.getMessage());
            }
        }
    }
    private BufferedImage preprocessImage(BufferedImage original) {

        BufferedImage resized = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(original, 0, 0, 256, 256, null);
        g.dispose();
        int x = (256 - 224) / 2;
        int y = (256 - 224) / 2;
        return resized.getSubimage(x, y, 224, 224);
    }
    private INDArray imageToINDArray(BufferedImage image) {
        int width = 224;
        int height = 224;
        INDArray input = Nd4j.create(1, 3, height, width);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color color = new Color(image.getRGB(x, y));
                double normalizedRed = (color.getRed() / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                double normalizedGreen = (color.getGreen() / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                double normalizedBlue = (color.getBlue() / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
                input.putScalar(new int[]{0, 0, y, x}, normalizedRed);
                input.putScalar(new int[]{0, 1, y, x}, normalizedGreen);
                input.putScalar(new int[]{0, 2, y, x}, normalizedBlue);
            }
        }
        return input;
    }
    private int[] getTopK(INDArray probabilities) {
        int[] topG = new int[5];
        INDArray flattened = probabilities.dup();

        for (int i = 0; i < 5; i++) {
            int maxIndex = 0;
            double maxValue = flattened.getDouble(0);

            for (int j = 1; j < flattened.length(); j++) {
                double value = flattened.getDouble(j);
                if (value > maxValue) {
                    maxValue = value;
                    maxIndex = j;
                }
            }
            topG[i] = maxIndex;
            flattened.putScalar(maxIndex, Double.NEGATIVE_INFINITY);
        }
        return topG;
    }
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                new FashionMNISTClassifier();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }
}
