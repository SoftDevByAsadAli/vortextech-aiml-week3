# VortexTech AI/ML Internship – Week 3: Regression and Clustering on Real Data

## What this is
My Week 3 submission for the VortexTech AI & ML Internship Track. This week I combined
two techniques on one dataset: a **regression** model predicting a continuous value, and
**K-Means clustering** to find hidden geographic groupings in the data.

**Dataset used:** California Housing Prices (20,640 rows, 10 columns)
**Source:** Public CSV mirror of the classic California census/housing dataset
(`ageron/handson-ml2` on GitHub)

## What was done

### Regression — predicting `median_house_value`
- Cleaned the dataset: median-filled 207 missing values in `total_bedrooms`, one-hot
  encoded the categorical `ocean_proximity` column
- 80/20 train/test split
- Trained and compared two models:
  - **Linear Regression** → RMSE $70,061 · R² 0.625
  - **Random Forest Regressor** → RMSE $49,404 · R² 0.814
- Random Forest was selected as the stronger model — lower error, higher explained
  variance, better suited to the non-linear relationships in housing data
- Plotted feature importances — `median_income` and geographic position
  (`latitude`/`longitude`) matter most

### Clustering — geographic K-Means
- Used only `longitude`/`latitude`, scaled with `StandardScaler`
- Ran the elbow method across k = 1 to 10 to justify the number of clusters
- Chose **k = 5** based on where the inertia curve flattens
- Visualized the 5 clusters on a 2D scatter plot (longitude vs. latitude)
- Cross-referenced clusters against income/price to confirm they carry real signal —
  coastal clusters (Bay Area, LA, San Diego) show meaningfully higher income and home
  values than inland clusters

## Files
- `week3_regression_clustering.ipynb` — the full notebook with markdown explanations,
  regression models, elbow method, and cluster visualization
- `housing.csv` — the dataset used
- `README.md` — this file

## How to run it
1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/vortextech-aiml-week3.git
   cd vortextech-aiml-week3
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook week3_regression_clustering.ipynb
   ```
4. Run all cells (`Cell → Run All`) to reproduce the regression models, elbow plot,
   and cluster visualization.

## Author
Asad Ali — VortexTech AI & ML Internship Track, Week 3
