# Gait Analysis for Gender Identification

This project applies machine learning and deep learning models to analyze gait data for identifying the gender of individuals based on their walking patterns. It utilizes the CASIA-B gait dataset, involving Convolutional Neural Networks (CNN) and machine learning algorithms like Random Forest, Decision Tree, and K-Nearest Neighbors (KNN) to classify and evaluate performance.

![Screenshot 2024-10-28 163817](https://github.com/user-attachments/assets/860c6b08-357e-47dd-84b5-7b474986c505)

![Screenshot 2024-10-28 163844](https://github.com/user-attachments/assets/73536c35-6512-4896-9d82-e8324f9e9f1d)


## Project Structure

- **Data Collection**: Use of the CASIA-B dataset containing de-noised, labeled images of individuals.
- **Data Preprocessing**: Techniques including image resizing, normalization, one-hot encoding, and data augmentation.
- **Deep Learning Model (CNN)**: Trained on images to identify unique gait features.
- **Machine Learning Models**: Feature extraction using HOG, LBP, and PCA, followed by classification using Random Forest, Decision Tree, and KNN.
- **Evaluation**: Use of accuracy, precision, recall, and F1-score metrics for model performance comparison.

## Contents

1. **Introduction**: Overview of gait analysis applications in biometric recognition, healthcare, and surveillance.
2. **Methodology**: Detailed description of preprocessing steps, feature extraction, and model training.
3. **Results**: Comparison of model accuracies and fine-tuning effects.
4. **Discussion**: Analysis of model results and performance against related literature.
5. **Conclusion and Future Work**: Summary of findings and recommendations for real-time applications.

## Data Collection

The CASIA-B dataset includes 124 subjects captured from various angles and under different conditions (normal, wearing a coat, and carrying a bag). Images are preprocessed to remove noise and standardize sizes.

## Model Descriptions

- **Convolutional Neural Network (CNN)**: Architecture designed with 12 layers, including convolutional, max-pooling, and dropout layers. Achieved the highest accuracy (98.83%) after hyperparameter tuning.
- **Random Forest**: Combined with HOG, LBP, and PCA for feature extraction, achieving an accuracy of 97.94% after tuning.
- **Decision Tree**: Features extracted using DenseNet-121, achieving 89.3% accuracy.
- **K-Nearest Neighbors (KNN)**: Feature extraction with ResNet-50, reaching an accuracy of 97.23%.

## Evaluation Metrics

- **Accuracy**: Percentage of correct predictions across all classifications.
- **Precision, Recall, and F1-Score**: Metrics used for evaluating model performance on male and female classifications.

## Results and Discussion

- **CNN** achieved the best performance with an accuracy of 98.83%.
- **Random Forest and KNN** also performed well, showing the effectiveness of feature extraction methods.
- **Decision Tree** had comparatively lower accuracy, indicating that it may be less suitable for high-dimensional gait data.

## Future Work

- **Real-time Applications**: Deployment of gait analysis in live environments for enhanced security and healthcare applications.
- **Advanced Models**: Exploration of improved deep learning architectures for even greater accuracy and efficiency.

## Requirements

- `tensorflow`
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `opencv-python`
- `scipy`

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Contact
**Author**: Rishwanth Mithra 

**Supervisor**: Dr. Saideh Ferdowsi

**Institution**: University of Essex, School of Mathematics, Statistics, and Actuarial Science

**Date**: September 2024

This project demonstrates the feasibility and accuracy of using gait as a biometric for gender classification, providing insights for future research and applications.
