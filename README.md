# Customer Segmentation with K-means and Factor Analysis

## Project Overview
This project analyzes the `customer-personality.csv` dataset to segment customers using two unsupervised learning approaches:

1. K-means clustering on purchase-behavior features.
2. Factor Analysis (FA) followed by K-means clustering.

The goal is to identify meaningful customer groups and translate them into practical business recommendations.

The full written report is available in [Machine_Learning_unsupervised_FA.pdf](./Machine_Learning_unsupervised_FA.pdf).

## Repository Contents
- `ML_CustomerSegment_FA.R`: end-to-end analysis script (preprocessing, clustering, validation, and plotting).
- `customer-personality.csv`: input dataset.
- `Machine_Learning_unsupervised_FA.pdf`: full report.
- `README.md`: project summary and key outputs.

## Data Source
The analysis uses the dataset in `customer-personality.csv`.

## Methodology

### Method 1: K-means on Purchase Behavior (Preprocessed Data)
The analysis assumes that web visits and purchase counts across channels are key behavioral indicators. Clusters are extracted from these features and interpreted as customer segments.

To validate whether predicted labels are meaningful, the cluster output is treated as the dependent variable and other customer attributes are used as independent variables in a supervised check (decision tree). The conclusion is that customer profile variables (e.g., household composition) and consumption habits (e.g., product-category spending) help explain cluster membership, supporting the validity of the clustering output.

### Method 2: Factor Analysis + K-means
Besides behavior-related fields, customer profile and consumption variables are also used for segmentation. Because of high dimensionality, dimension reduction is applied before clustering.

PCA, ICA, and FA were considered. FA was selected because:
- It captures unique variance and error terms.
- It retained a smaller latent dimension (8 factors) than PCA (more than 10) while keeping satisfactory explained variance.
- It provided clearer separation in early dimensions, improving cluster interpretability in theory.

### Practical Workflow
1. Run K-means clustering with behavior-related columns.
2. Run FA and then K-means clustering on factor scores.
3. Match clustered output back to customer information.
4. Compare overlap/separation and distribution plots between methods.

During FA, non-factorial categorical variables in customer information (e.g., marital status categories in raw form) are dropped before modeling.

## Results

### 1) K-means Clustering with Original Data
From the elbow plot (k = 1 to 10), the largest variance drop occurs at **k = 4**, so 4 clusters are selected.

**Plot 1 - Distribution of Clusters**  
![Picture 1](https://user-images.githubusercontent.com/61338647/170355269-01912370-170b-410d-a4ac-83fbfc38f50f.png)

Observations:
- Some features (e.g., enrollment date and birth year) show overlap.
- Clearer separation appears on major spending variables (wine, fruits, meat, fish, sweets, gold).
- Purchase-channel behavior also shows identifiable relationships.

### 2) K-means Clustering with Factor Analysis
Best FA model results:
- Chi-square statistic: **7.182** on **7** degrees of freedom.
- p-value: **0.349** (> 0.05), indicating good fit with **8 factors**.
- Cumulative explained variance (eigenvalue-based): **87%**.
- Most influential factors by eigenvalues: **5.98, 1.78, 1.01, 0.98**.

Rotated FA provides clearer loading structure than unrotated FA, improving interpretation.

**Plot 2 - Factor Analysis Loadings**  
![Picture 2](https://user-images.githubusercontent.com/61338647/170355305-2d4f8322-0c9e-4190-84fc-bb6d098e7015.png)

Interpretation highlights:
- `Income` is mainly influenced by Factors 1, 2, 3, and 6.
- Factor 2 has the strongest impact on `Dt_Customer`.
- `Recency` has weak loading overall (0.108 on Factor 8).
- Category spending variables (`MntFruits`, `MntFishProducts`, `MntSweetProducts`) are strongly linked with Factor 1.
- Factor 3 strongly impacts `MntWines` (loading = 1.034).
- Factor 4 strongly impacts `MntGoldProds` (loading = 0.973).
- `NumDealsPurchases` is mainly influenced by Factors 2 and 6.
- Factor 8 strongly impacts `NumWebPurchases`.
- `NumCatalogPurchases` loads across multiple factors.
- `NumStorePurchases` is most affected by Factor 5.
- `NumWebVisitsMonth` is most affected by Factor 2 (loading = 0.862).

**Plot 3 - Original Variables Distribution**  
![Picture 3](https://user-images.githubusercontent.com/61338647/170355313-48a303e8-71da-416d-aafe-e09759fd9f24.png)

Elbow analysis indicates **3 clusters** are optimal after FA (explaining more than 80% variance). Cluster distribution across factors shows stronger separation than raw-feature clustering in several dimensions.

## Findings

**Plot 4 - Original Variables Distribution**  
![Picture 4](https://user-images.githubusercontent.com/61338647/170355326-922d07be-3741-4608-927a-7d2eff5ce404.png)

**Plot 5 - Factor Analysis of Factors Distribution**  
![Picture 5](https://user-images.githubusercontent.com/61338647/170355340-5256ca93-e161-4b58-bd74-196b3536c003.png)

**Plot 6 - Factor Analysis and Original Data Clusters Distinction Distribution**  
![Picture 6](https://user-images.githubusercontent.com/61338647/170355357-0b26508b-c6bd-4bce-a937-46e6fc746b15.png)

Overall findings:
- Both methods identify similar key drivers (notably income and major product spending behavior).
- FA-based clustering shows clearer cluster distinction and less overlap.
- Original-data clustering still captures important behavioral patterns but with more overlap/noise.

### K-means with Original Data
**Plot 7 - Original Data Cluster Mean & Snake Plot**  
![Picture 7](https://user-images.githubusercontent.com/61338647/170355370-8b8d0c87-3eab-40c5-b865-da5428a14683.png)

Purchasing behavior differs across four clusters, and variation can be linked to demographic characteristics and overall spending levels.

### K-means with Factor Analysis
**Plot 8 - Factor Analysis Cluster Mean & Snake Plot**  
![Picture 8](https://user-images.githubusercontent.com/61338647/170355388-ddf68df6-d469-4c81-bbbd-e65b77f88397.png)

Based on FA + K-means output, cluster-specific insights are summarized below.

**Plot 9 - Cluster Findings Summary**  
![Picture 9](https://user-images.githubusercontent.com/61338647/170355396-b42a449b-7907-412e-849a-4dddf83f8e2d.png)

## Business Recommendations

### Customer Segment 1
Cluster 1 has lower monetary potential than Cluster 3, but it is still worth targeted investment. Frequent promotions (especially wine discounts) are recommended. Since this segment is comfortable with online shopping, digital discount campaigns across channels should be prioritized.

### Customer Segment 2
Cluster 2 has the lowest monetary potential and contribution. The company should minimize investment here, as major commercial initiatives are likely to generate lower ROI.

### Customer Segment 3
Cluster 3 has high monetary value and strong buying activity. The company should prioritize this segment through premium product strategy (especially fish and wine), exclusive in-store services, and high-touch marketing. Digital initiatives are also viable because this segment is willing to buy online.

## Reproducibility

### Requirements
Install R packages used in `ML_CustomerSegment_FA.R` (e.g., `tidyverse`, `ClusterR`, `cluster`, `psych`, `GPArotation`, `factoextra`, `caret`, `rattle`, `ggpubr`, `corrplot`, `RColorBrewer`, `lubridate`, `naniar`, `reshape`).

### Run
Execute the script from the repository root:

```bash
Rscript ML_CustomerSegment_FA.R
```

This will perform preprocessing, clustering, validation, and generate the plots used in the analysis.
