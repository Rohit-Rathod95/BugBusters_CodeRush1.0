data sets link:
https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi%3A10.18150%2Frepod.0107441
https://www.kaggle.com/datasets/broach/button-tone-sz
https://physionet.org/


Step 1 – Data Collection  
EEG data of healthy and affected individuals is collected from open medical repositories.

 Step 2 – Data Preprocessing  
- Noise removal using filters (band-pass filtering).  
- Normalization of EEG signals.  
- Artifact removal to ensure clean data.  

Step 3 – Feature Extraction  
- Time-domain features.  
- Frequency-domain features using FFT.  
- Power spectral density and wavelet transforms.  

Step 4 – Feature Selection  
Important features are selected to reduce dimensionality and improve accuracy.  

 Step 5 – Dataset Splitting  
Data is split into training and testing sets for evaluation.  

=Step 6 – Model Training  
Logistic Regression is applied to classify EEG signals into *Healthy* or *Affected*.  

 Step 7 – Hyperparameter Tuning  
Cross-validation is used for optimizing model parameters.  

 Step 8 – Model Evaluation  
Model is evaluated using:  
- *Accuracy*  
- Cross-validation for robustness  

 Impact
1. Helps in *early detection* of schizophrenia.  
2. Provides a *non-invasive diagnostic method*.  
3. Assists doctors with *decision support*.  
4. Improves *mental healthcare outcomes*.  


