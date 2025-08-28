# 🧠 EEG + Machine Learning Portfolio Project  

This project combines **neuroscience** and **machine learning** by decoding EEG signals to classify simple mental states.  
The pipeline is designed to be **educational, reproducible, and visually appealing** – perfect for a portfolio showcase.  

---

## 📌 Project Goals
- Explore real EEG data (open datasets, e.g., PhysioNet).
- Preprocess signals (filtering, ICA, epoching).
- Extract meaningful features (bandpower, CSP).
- Train and evaluate ML models (LDA, SVM, etc.).
- Generate **clear visualizations** for each stage.

---

## 📂 Repository Structure  
eeg-ml-project/  
├─ notebooks/ # Jupyter notebooks for exploration  
├─ src/ # Python modules for preprocessing & visualization  
├─ reports/figures/ # Saved plots and figures  
├─ models/ # Trained ML models  
├─ data/ # EEG datasets (ignored in git)  
├─ requirements.txt  
├─ README.md  
└─ LICENSE  


---

## 🔬 Methods
1. **Exploratory Data Analysis (EDA)**  
   - Raw EEG signals  
   - Power Spectral Density (PSD)  
   - Spectrograms and topographic maps  

2. **Preprocessing**  
   - Bandpass filtering (1–40 Hz)  
   - Artifact removal (ICA)  
   - Epoch segmentation  

3. **Feature Engineering**  
   - Bandpower in delta, theta, alpha, beta, gamma ranges  
   - Common Spatial Patterns (CSP)  

4. **Machine Learning**  
   - Baselines: LDA, SVM  
   - Metrics: Accuracy, ROC/AUC, Confusion Matrix  

---

## 📊 Visual Outputs
The project produces a variety of figures stored in `reports/figures/`, including:
- Raw vs. filtered EEG signals  
- PSD plots across frequency bands  
- Spectrograms and topomaps  
- Confusion matrix  
- ROC curves and learning curves  

Example (placeholder):  

![Confusion Matrix](reports/figures/confusion_matrix.png)  

---

## 🚀 Getting Started
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/eeg-ml-project.git
cd eeg-ml-project
pip install -r requirements.txt
