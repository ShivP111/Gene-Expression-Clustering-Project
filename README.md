# Gene-Expression-Clustering-Project
This project applies unsupervised machine learning to cluster breast cancer samples based on their gene expression profiles. Using the publicly available GSE45827 dataset, we explore how different clustering methods group patients into biologically meaningful subtypes (Basal, LumA, LumB, Her2).

# 🧬 Gene Expression Clustering

> **Unsupervised Machine Learning applied to Breast Cancer Gene Expression Data**

This project explores how machine learning can identify **hidden patterns in gene expression data** and cluster breast cancer samples into biologically meaningful groups. Using **Kaggle’s Breast Cancer Gene Expression dataset (GSE45827)**, we apply **PCA**, **K-Means**, and **Hierarchical Clustering** to group samples and evaluate how well clusters align with known cancer subtypes (*Basal, LumA, LumB, Her2*).

---

## 📂 Dataset

* **Source:** [Kaggle – Breast Cancer Gene Expression Profiles (GSE45827)](https://www.kaggle.com/)
* **Samples:** 151 patients
* **Features:** 54,675 gene expression values per patient
* **Labels:** Known breast cancer subtypes (Basal, LumA, LumB, Her2)

---

## 🔑 Workflow

### 1. **Data Preparation**

* Loaded CSV dataset from Kaggle into Google Colab.
* Removed duplicate and missing values.
* Verified 151 unique patient samples.

### 2. **Feature Engineering**

* Dropped non-gene columns (`samples`, `type`).
* Standardized features with **StandardScaler**.
* Reduced dimensionality with **PCA**:

  * Option 1: Fixed 50 components.
  * Option 2: Capture 90% variance (\~106 PCs).

### 3. **Clustering**

* **K-Means Clustering**

  * Used **Elbow Method** and **Silhouette Score** to find optimal k.
  * Best fit: k = 4 (matches biological subtypes).
* **Hierarchical Clustering**

  * Ward’s linkage + Euclidean distance.
  * Visualized dendrogram to check structure.

### 4. **Evaluation**

* Compared clusters with true subtype labels using **Adjusted Rand Index (ARI)**.
* Results:

  * K-Means ARI ≈ **0.41**
  * Hierarchical ARI ≈ **0.56** (performed better).

### 5. **Biological Interpretation**

* Computed **average gene expression per cluster**.
* Identified **top 5 highly expressed genes** in each cluster.
* Visualized gene expression differences across clusters using **heatmaps**.

---

## 📊 Results

* Clustering revealed **clear separation of breast cancer subtypes**.
* **Hierarchical clustering outperformed K-Means** in matching true biological labels.
* Heatmaps highlighted **gene expression signatures** that may help differentiate subtypes.

---

## 🚀 Tech Stack

* **Programming:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Scipy, Matplotlib, Seaborn
* **Environment:** Google Colab
* **Dataset Source:** Kaggle

---

## 📌 Key Learnings

* High-dimensional biomedical data requires **dimensionality reduction (PCA)** before clustering.
* **Clustering evaluation metrics (ARI, Silhouette Score)** are critical when ground truth labels exist.
* **Hierarchical clustering** may capture biological structures better than K-Means.
* Heatmaps help interpret **biological meaning** of machine learning clusters.

---

## 📷 Visuals (Example Outputs)

* PCA variance explained plot
* K-Means Elbow & Silhouette method
* Dendrogram (Hierarchical Clustering)
* Heatmap of gene expression profiles

---

## 📖 How to Run

1. Clone the repository:

   git clone https://github.com/your-username/gene-expression-clustering.git
   cd gene-expression-clustering

2. Open in **Google Colab** or Jupyter Notebook.

3. Upload dataset from Kaggle into `/data/` folder.

4. Install dependencies (if running locally):

   pip install -r requirements.txt

5. Run the notebook step by step.

---

## 🔮 Future Work

* Try other clustering methods (DBSCAN, Gaussian Mixture Models).
* Perform **feature selection** to reduce noise genes.
* Use **biological pathway enrichment analysis** on cluster-specific genes.
* Build an interactive **dashboard** for visualizing clustering results.

---

## 👨‍🎓 Author

📌 *This project was created for educational purposes to demonstrate machine learning in genomics.*
