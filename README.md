## 1. Introduction
This report aims to provide a detailed analysis of the customer personality data set. Section 2 describes the methodology applied to explore the data set. The empirical results are presented in section 3. The main findings on clusters are discussed in section 4. The final section provides practical recommendations for the company to better serve consumers in each cluster.  

## 2. Methodology and Steps
### Method 1 - Using purchase behaviour of pre-processed data and applying k-mean clustering
We assume that web visits and purchase numbers in different channels are companies’ main concern as consumer behaviours, so we attempt to extract clusters from these features and categorize consumers via extracted labels.
To examine the predicted labels’ validity, we will use cluster outputs as dependent variable and others as independent variables to figure out if the clusters are well enough to epitomize different types of consumers. A simple supervised machine learning can deal with the validation. The conclusion is that the individual information (kids at home etc.) and consumption habits (meat consumption amounts etc.) can signify their consumer behaviours, bolstering the validity of clustering outputs and it’s also feasible to check decisive features to classify different types of consumers.
### Method 2 - Factor Analysis and applying k-mean clustering
Aside from behaviour-related data, we also assume that remaining data, mainly consisting of consumer’s individual information and consumption habits, are reasonable source of customer segmentation. Given its multi-dimensionality, we need to deploy a dimension reduction strategy to overcome its complexity and discern pivotal features. PCA, ICA and FA are taken into considerations.  
  
Compared to PCA and ICA, factor analysis can capture unique variance of specific and errors. It also retains a smaller dimension (8) than PCA (more than 10) while keeping a satisfactory explained variance. Likewise, Factor analysis is designed to project a high-dimension data to a lower vector space on account of feature’s specific variance and error. Compared to simple clustering, it gives an ordered attention to features that best discriminate the dataset. Thus, in contrast to unweighted clustering, FA can split the data more separately in the first several dimensions, leading to a more accurate categorization in theory.  

### In Practice
This section will begin by presenting the K-means clustering with behaviour-related columns. After that, the K-means clustering with Factor analysis will be presented.  
We dropped the non-factorial categorical variables in consumer information (marital status etc.) and processed the factor analysis and clustering with remaining data. We matched the clustered output to consumer information and examined whether the labels accommodate to the categorical variables.
We drew the overlapping and distribution plot in first two dimensions when using different methods, compared their effect and evaluated two clustering methods.

## 3. Results
This section will begin by presenting the K-means clustering with original data. After that, the K-means clustering with Factor analysis will be presented.
3.1 K-means Clustering with Original Data
Based on the populated elbow plot in R code, the point with significant drop in variance is when k = 4 from the range of 1 to 10. Hence, 4 is the chosen optimal cluster number.

![Picture 1](https://user-images.githubusercontent.com/61338647/170355269-01912370-170b-410d-a4ac-83fbfc38f50f.png)

As can be seen from the distribution **Plot 1**, there are overlaps of observations from some features such as customer’s enrolment date and birth year, hence, no clear pattern is seen. However, a visible clear pattern is seen for other significant features such as income, amount spent on wine, fruits, meat, fish, sweet and gold in the last 2 years. Also, relationships can be identified from the dispersion patterns of purchases made through different channels.  

![Picture 2](https://user-images.githubusercontent.com/61338647/170355305-2d4f8322-0c9e-4190-84fc-bb6d098e7015.png)

![Picture 3](https://user-images.githubusercontent.com/61338647/170355313-48a303e8-71da-416d-aafe-e09759fd9f24.png)

![Picture 4](https://user-images.githubusercontent.com/61338647/170355326-922d07be-3741-4608-927a-7d2eff5ce404.png)

![Picture 5](https://user-images.githubusercontent.com/61338647/170355340-5256ca93-e161-4b58-bd74-196b3536c003.png)

![Picture 6](https://user-images.githubusercontent.com/61338647/170355357-0b26508b-c6bd-4bce-a937-46e6fc746b15.png)

![Picture 7](https://user-images.githubusercontent.com/61338647/170355370-8b8d0c87-3eab-40c5-b865-da5428a14683.png)

![Picture 8](https://user-images.githubusercontent.com/61338647/170355388-ddf68df6-d469-4c81-bbbd-e65b77f88397.png)

<img width="1045" alt="Picture 9" src="https://user-images.githubusercontent.com/61338647/170355396-b42a449b-7907-412e-849a-4dddf83f8e2d.png">






