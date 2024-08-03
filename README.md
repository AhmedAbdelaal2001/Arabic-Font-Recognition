# Arabic-Font-Recognition

A classifier that takes an image containing a paragraph written in Arabic and classifies the paragraph into one of four fonts: Scheherazade New, Marhey, Lemonada, or IBM Plex Sans Arabic.

## Instructions to Run the Code

1. **Provide the Dataset**:
   - Ensure you have a dataset in a zip file named `data.zip`.
   - Place the `data.zip` file in the root directory of the project.

2. **Open the Jupyter Notebook**:
   - Open the `Predict.ipynb` notebook.

3. **Run All Cells**:
   - In the Jupyter Notebook, select `Cell` from the top menu.
   - Click on `Run All` to execute all the cells in the notebook.

Within each directory, you will find a README explaining its contents; please view them to understand the flow.

## Pipeline

### 1. Preprocessing
The preprocessing stage involves several steps to enhance the quality of the input images:
- **Noise Removal**: Salt and pepper noise is removed using median filters.
- **Binarization**: Images are binarized using Otsu's thresholding technique.
- **Foreground/Background Correction**: Any reversals between the foreground and background are fixed by examining the image borders.
- **Rotation Correction**: Rotations are corrected by drawing a bounding box around the text, calculating the angle of the box, and adjusting it accordingly.
- **Cropping**: Images are cropped to retain only the text portion.

![image](https://github.com/user-attachments/assets/9e09e708-4601-441f-b026-8bd34af1cd88)

### 2. Feature Selection and Extraction
For feature selection and extraction, we experimented with various methods before finalizing on the best approach:
- **Initial Experiments**: Gabor Features and Laws Energy Measures were tested but did not yield satisfactory results.
- **Final Approach**: We use a combination of SIFT (Scale-Invariant Feature Transform) and BoVW (Bag of Visual Words) features. SIFT features are extracted from each image and fed into a trained KMeans model, which assigns each feature vector to one of 200 clusters. A histogram of these assignments forms a 200-dimensional vector, which serves as the final feature vector.

![image](https://github.com/user-attachments/assets/1fad5753-7336-49b5-8a7e-84fc79c41fec)

### 3. Model Selection and Training
Various models were tested to find the best performance:
- **Models Tested**: Logistic regression, decision trees, random forests, XGBoost, shallow neural networks, and convolutional neural networks on raw images.
- **Best Model**: The best performance was achieved with an SVM (Support Vector Machine) model with a polynomial kernel, using the SIFT + BoVW features.

![image](https://github.com/user-attachments/assets/8103a41f-24ed-4baa-8422-5532b3240f0f)

### 4. Performance Analysis
The performance of the final model is highly impressive:
- **Accuracy**: The model achieves an accuracy of 99.875% on the test set, correctly classifying 799 out of 800 examples.
- **F1-Score**: The F1-score is very close to 1.00, indicating excellent precision and recall.

![image](https://github.com/user-attachments/assets/34027fa7-0a10-463f-8438-16e6bbfd5935)

