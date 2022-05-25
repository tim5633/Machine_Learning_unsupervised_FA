library(lubridate) #for converting datetime
library(ClusterR) # for K means clustering 
library(cluster) # for K means clustering 
library(tidyverse) # meta package of all tidyverse packages (using for plotting the distribution plot)
library(naniar) # package to check missing data
library(corrplot) # co-efficiency 
library(caret) # for doing the classfication tree
library(rattle) #for fancyRpartPlot
library(psych) # factor analysis packages
library(GPArotation) # factor analysis packages
library(ggplot2) # fopr plotting purpose
library(factoextra) # for choosing the best cluster
library(dplyr) # for heat map
library(RColorBrewer) # for heat map and corrlation plot
library(reshape) # Transform the matrix in long format
library(ggpubr) # for plotting of distribution with single dimantion

################################################################################
###### Step 1 : Data Pre-processing ############################################
################################################################################
# Step 1: Read the Survey Data
cp_raw = read.csv("customer-personality.csv",header=T)

# Step 2: Drop the useless column (ID is not going to be helpful when analyzing)
cp_raw <- cp_raw[,!(names(cp_raw) %in% c("ID"))]

# Step 3: Collapsing the Education into two Categories: graduate and basic
cp_raw$Education <- replace(cp_raw$Education, cp_raw$Education=="PhD", "graduate")
cp_raw$Education <- replace(cp_raw$Education, cp_raw$Education=="Master", "graduate")
cp_raw$Education <- replace(cp_raw$Education, cp_raw$Education=="Graduation", "graduate")
cp_raw$Education <- replace(cp_raw$Education, cp_raw$Education=="2n Cycle", "Basic")
cp_raw$Education <- replace(cp_raw$Education, cp_raw$Education=="non-graduate", "Basic")
# Converting Factor to 1/0 Dummy
cp_raw <- data.frame(cp_raw[ , ! colnames(cp_raw) %in% "Education"], # Create dummy data
                     model.matrix( ~ Education - 1, cp_raw))
# Collapsing the Marital_Status into two Categories: 1 with single and 2 with together
cp_raw$Marital_Status <- replace(cp_raw$Marital_Status, cp_raw$Marital_Status=="Divorced", 1)
cp_raw$Marital_Status <- replace(cp_raw$Marital_Status, cp_raw$Marital_Status=="Widow", 1)
cp_raw$Marital_Status <- replace(cp_raw$Marital_Status, cp_raw$Marital_Status=="Single", 1)
cp_raw$Marital_Status <- replace(cp_raw$Marital_Status, cp_raw$Marital_Status=="Alone", 1)
cp_raw$Marital_Status <- replace(cp_raw$Marital_Status, cp_raw$Marital_Status=="Absurd", 1)
cp_raw$Marital_Status <- replace(cp_raw$Marital_Status, cp_raw$Marital_Status=="YOLO", 1)
cp_raw$Marital_Status <- replace(cp_raw$Marital_Status, cp_raw$Marital_Status=="Married", 2)
cp_raw$Marital_Status <- replace(cp_raw$Marital_Status, cp_raw$Marital_Status=="Together", 2)
# Converting Factor to 1/0 Dummy
cp_raw <- data.frame(cp_raw[ , ! colnames(cp_raw) %in% "Marital_Status"], # Create dummy data
                     model.matrix( ~ Marital_Status - 1, cp_raw))
# Drop n-1 of the original categorical vairables of EducationBasic and Marital_Status1
cp_raw <- cp_raw[,!(names(cp_raw) %in% c("EducationBasic"))]
cp_raw <- cp_raw[,!(names(cp_raw) %in% c("Marital_Status1"))]
# drop the categorical columns
cp_raw <- cp_raw[,!(names(cp_raw) %in% c("Education","Marital_Status"))]

# Step 4: converting Year_Birth and Dt_Customer
# Dt_Customer
cp_raw$Dt_Customer = dmy(cp_raw$Dt_Customer) # change the format of date to "y-m-d"
cp_raw$Dt_Customer = as.Date(cp_raw$Dt_Customer) # convert character to date
cp_raw$Dt_Customer = as.numeric(cp_raw$Dt_Customer, "months")
# the bigger of the month (exact value), 
# the later they regrister (more close to new customer, so the date is reversing)
# why using "month" is that we could cauture more precise value
# Year_Birth
cp_raw$Year_Birth = as.numeric(cp_raw$Year_Birth, "years")
# the same logic as Dt_Customer, the larger of the Year, the younger of the customer
# so the years is reversing as well.
# why we're not using 2022 - datapoint is we're not sure if the exact date of the dataset
# if using the base year to calculate, would be not so accurate (we here only look at the corresponds trends)

# Step 5: missing data processing
miss_var_summary(cp_raw) # only the income has missing data of only 24 rows
# drop the NA value directly (not going to influence the output), or we could use the missForest to fill
cp_raw <- cp_raw[complete.cases(cp_raw), ]

# Step 7 : Removing the outlier (only be significant in Income)
boxplot(cp_raw) #there's a significant outlier at Income, cp_raw[c(2)]
out_lier <- boxplot(cp_raw[c(2)], plot=FALSE)$out
cp_raw <- cp_raw[-which(cp_raw$Income %in% out_lier),]
boxplot(cp_raw[c(2)]) # make sure there's no outlier
df <- cp_raw

# Step 6 : MinMaxScaler for range 0 to 1 
# the is for plotting and cluster purpose (to compare with the diversity of cluster)
# but for FA we wouldn't need to
MinMaxScaler_function<- function(x){
  for(i in seq(length(x))){
    x[,i] <- (x[,i]-min(x[,i]))/(max(x[,i])-min(x[,i]))
  }
  return(x)
}

################################################################################
###### Step 2 : Check co-efficiency and distribution of the data set ###########
################################################################################
# co-efficiency and distribution to how we're going to deal with the data (including the spilt)
corr <- cor(df[,unlist(lapply(df,is.numeric))][,-1])
corrplot(corr,type="upper", order="hclust",
         col=brewer.pal(n=8, name="RdYlBu"),method = 'number',  
         number.cex = 0.5, number.font =2, tl.col = 'black', 
         tl.cex = 0.5)
# Histograms and density lines
# from cloumn1 to column10
par(mfrow=c(2, 5))
colnames <- dimnames(df)[[2]]
for (i in 1:10) {
  hist(df[,i], main=colnames[i], probability=TRUE, col="gray", border="white")
  d <- density(df[,i])
  lines(d, col="red")
}
#from column11 to column19
par(mfrow=c(2, 5))
colnames <- dimnames(df)[[2]]
for (i in 11:19) {
  hist(df[,i], main=colnames[i], probability=TRUE, col="gray", border="white")
  d <- density(df[,i])
  lines(d, col="red")
}
# we could observe that the distribution is not gaussian distribution

################################################################################
###### Step 3 : Clustering with K-means without purchase number and web visits #
################################################################################
# Since we believe that purchase number and web visits is the most closing to the company's caring data
# We're going to use customer's daily life data not accosiating with the company (df_life)
# to making cluster and see if there's any pattern to the company purchase behavior
df_life <- df[-c(17,16,15,14,13)]
df_purchase_behavior <- df[c(17,16,15,14,13)]

# scale data 0-1 for the purposing of clustering
df_life <- MinMaxScaler_function(df_life)
df_purchase_behavior <- MinMaxScaler_function(df_purchase_behavior)

# choosing the best number of cluster label with K-mean and different methods
fviz_nbclust(df_purchase_behavior, kmeans, method = 'wss') # best: 4

# set the number of clusters of 4 and write back to df_life and df_purchase
km_original <- kmeans(df_purchase_behavior, centers=4)
df_life$cluster <- km_original[[1]]
df_purchase_behavior$cluster <- km_original[[1]]

################################################################################
###### Step 4 : plotting to see trends with cluster of original data############
################################################################################
# plot the cluster distribution of different clusters
ori_df_scale <- MinMaxScaler_function(df)
ori_df_scale$cluster <- km_original[[1]]
ori_df_scale$cluster=factor(ori_df_scale$cluster)
ori_df_scale %>% 
  pivot_longer(cols = 1:19, names_to = "variables", values_to = "values") %>% 
  ggplot(data = ., aes(x = variables, y = values, color = cluster, 
                       group = cluster,shape= cluster)) +
  geom_jitter()+
  ggtitle("Cluster Distribution of Original Dataset")+
  theme(axis.text.x = element_text(angle = 60, hjust = 1))

# for the variables and clusters's plotting 
# we're scale here for interperation from 0 to 1 but will return the value when explain
# group_by the dataset by cluster
df_ori_plot <- MinMaxScaler_function(df)
df_ori_plot$cluster <- km_original[[1]]
ori_group <- data.frame(df_ori_plot %>% group_by(cluster)  %>%
                          summarise(Year_Birth = mean(Year_Birth),
                                    Income = mean(Income),
                                    Kidhome = mean(Kidhome),
                                    Teenhome = mean(Teenhome),
                                    Dt_Customer = mean(Dt_Customer),
                                    Recency = mean(Recency),
                                    MntWines = mean(MntWines),
                                    MntFruits = mean(MntFruits),
                                    MntMeatProducts = mean(MntMeatProducts),
                                    MntFishProducts = mean(MntFishProducts),
                                    MntSweetProducts = mean(MntSweetProducts),
                                    MntGoldProds = mean(MntGoldProds),
                                    NumDealsPurchases = mean(NumDealsPurchases),
                                    NumWebPurchases = mean(NumWebPurchases),
                                    NumCatalogPurchases = mean(NumCatalogPurchases),
                                    NumStorePurchases = mean(NumStorePurchases),
                                    NumWebVisitsMonth = mean(NumWebVisitsMonth),
                                    Educationgraduate = mean(Educationgraduate),
                                    Marital_Status2 = mean(Marital_Status2),
                                    .groups = 'drop'))
# melt 
ori_melt <- melt(ori_group, id.vars = 'cluster', 
                 measure.vars = c("Year_Birth","Income","Kidhome",
                                  "Teenhome","Dt_Customer","Recency",
                                  "MntWines","MntFruits","MntMeatProducts",
                                  "MntFishProducts","MntSweetProducts","MntGoldProds",
                                  "NumDealsPurchases","NumWebPurchases","NumCatalogPurchases",
                                  "NumStorePurchases","NumWebVisitsMonth","Educationgraduate",
                                  "Marital_Status2"),
                 variable_name = 'Original',
                 value.name = 'value')

ori_melt$cluster=factor(ori_melt$cluster)

#get the mean of each cluster within variable and plot heatmap and snake plot
# heatmap distribution plot
c = ggplot(ori_melt, aes(x=Original, y=cluster, fill = value)) + 
  geom_tile(color = "white",lwd = 0.2, linetype = 1) +
  coord_fixed()+
  geom_text(aes(label = round(value,2)), color = "white", size = 4)+
  ggtitle("Original Dataset Average clusters score heatmap")+
  theme(
    axis.title.x=element_blank(),
    axis.text.x = element_text(angle = 90))


# snake plot
d = ggplot(ori_melt, aes(x= Original, y=value, color = cluster)) + 
  geom_line(aes(color = cluster, group = cluster)) + 
  theme_minimal()+ 
  ggtitle("Original Dataset Clusters snake plot")+
  theme(axis.text.x = element_text(angle = 90))
# merging together to the plot (might need to use the Zoom to see more clear)
ggpubr::ggarrange(c, d,
                  ncol = 1, nrow = 2)

# using the fviz_cluster to plot overall distribution of the cluster
factoextra::fviz_cluster(km_original, data = df_purchase_behavior[, -5],
                         palette = c("#0073C2FF", "#FC4E07", "#07fc4e","#07b5fc"),
                         geom = "point",
                         main = "Original Cluster plot",
                         ellipse.type = "convex",
                         ggtheme = theme_bw())

# for each variables we plot out to see the distribution
df_ori_plot$cluster <- factor(df_ori_plot$cluster)
o1 =  ggdensity(df_ori_plot, x = "Dt_Customer",
                add = "mean", rug = TRUE,
                color = "cluster", fill = "cluster",
                palette = c("#0073C2FF", "#FC4E07", "#07fc4e","#07b5fc"))
o2 = ggdensity(df_ori_plot, x = "Recency",
               add = "mean", rug = TRUE,
               color = "cluster", fill = "cluster",
               palette = c("#0073C2FF", "#FC4E07", "#07fc4e","#07b5fc"))
o3 = ggdensity(df_ori_plot, x = "Educationgraduate",
               add = "mean", rug = TRUE,
               color = "cluster", fill = "cluster",
               palette = c("#0073C2FF", "#FC4E07", "#07fc4e","#07b5fc"))
o4 = ggdensity(df_ori_plot, x = "Year_Birth",
               add = "mean", rug = TRUE,
               color = "cluster", fill = "cluster",
               palette = c("#0073C2FF", "#FC4E07", "#07fc4e","#07b5fc"))
ggpubr::ggarrange(o1, o2, o3, o4, ncol = 4, nrow = 1)

################################################################################
###### Step 5 : life and purchase behavior of connection and important variables
################################################################################
#for the conscientiousness of the cluster 
# we check whether df_life could directly apply the cluster from df_purchase_behavior
# with using the decision tree
# changing the cluster to factor
df_life$cluster=factor(df_life$cluster)

# set the train and test split
set.seed(33) 
train.index=createDataPartition(df_life[,ncol(df_life)],p=0.7,list=FALSE) 
train=df_life[train.index,]
test=df_life[-train.index,]

# using carer to find the optimal solution
set.seed(48) 
trControl=trainControl( method = "repeatedcv", 
                        number = 5, # 5-fold cross-validation
                        repeats = 1,)
# create model with fix random seed
dtFit=train(train[c(-15)],
            train$cluster,
            method = "rpart",
            tuneLength=10, 
            metric="Accuracy",
            trControl = trControl) # cp = 0.004133180  accu = 0.7144650
dtFit

# check detail of the tree 
print(dtFit$finalModel) # with 5 leaves, 4variables

# plot a tree
par(mfrow=c(1, 1))
decision_tree = fancyRpartPlot(dtFit$finalModel,
               main="Purchase behavior",
               uniform=TRUE,
               tweak=1.1,
               lwd = 2,
               sub="") 
# make the prediction 
pred_dt = predict(dtFit$finalModel,
                  newdata=test[,-which(colnames(test)=="cluster")],
                  type="class")
print(mean(pred_dt==test[,which(colnames(test)=="cluster")])) # decisiontree 0.6954545

################################################################################
###### Step 6 : Exploratory Factor Analysis : using customer behavior ##########
################################################################################
# we now using the Exploratory Factor Analysis to see 
# whether the cluster would perform better or have different finding from the original dataset

# remove all the customer information data 
# categorical and non-continues data are not fitted to the FA
df_behaviors <- df[-c(1,3,4,18,19)]
df_information <- df[c(1,3,4,18,19)]
## rotation model(with choosing the best chi square p-value )
for (i in seq(10)){ 
  # We are using the rotation = "promax" but not "none"
  # after observing we found that the components and P-value are both the same
  # but promax rotation model could help us to reduce the variables showing in other FA components
  # would be more straightforward to explain
  rotation = factanal(x = df_behaviors, factors = i, rotation = "promax")
  # We also printed out the p-value of chi-squared to see how many components would be significant
  # also when the variables is too many, the factanal would have error and stop.
  print(paste("factor is",i,"wtih p-value",rotation[[11]][[1]]))
}
# factor is 8 wtih p-value 0.348776576382112
fa_res = factanal(x = df_behaviors, factors = 8, rotation = "promax", scores = "Bartlett")
fa_res # print out to see the details: The p-value is 0.349 
# Factor Scores and create a df to record
FA_df <- as.data.frame(fa_res$scores)

# calculate correlation
fa_corr = cor(df_behaviors)
# get eigen factor 
fa_eigen = eigen(fa_corr) # ie the loading within each components in FA
fa_eigen$values # ie the loading of each components in FA
# cumulative sum
cumsum(fa_eigen$values)/sum(fa_eigen$values) #  5:0.7517612 6:0.7982041 7:0.8379502

################################################################################
###### Step 7 : FA Clustering of FA_df with K-means ############################
################################################################################
# scaling before clustering and plotting (will turn back to the actual value when explainging)
df_information <- MinMaxScaler_function(df_information)
FA_df_scale <- MinMaxScaler_function(FA_df) # scaling before doing the cluster
# using the wss method to find the optimal K
fviz_nbclust(FA_df_scale, kmeans, method = 'wss') # we decided to choose three cluster: 3 cluster
# Perform K-Means Clustering with Optimal K
km_FA <- kmeans(FA_df, 3, nstart = 25,) #when at 3, the slope is started to be gentle
FA_df$cluster <- km_FA[[1]] # write the cluster label back
df_information$cluster <- km_FA[[1]]  # write the cluster label back

################################################################################
###### Step 8: Plotting and Customer information ##############################
################################################################################
# plot the cluster distribution of different clusters
# we're scale here for interperation from 0 to 1 but will return the value when expalin 
FA_df_scale <- MinMaxScaler_function(FA_df[c(-9)])
FA_df_scale$cluster <- km_FA[[1]]
FA_df_scale %>% 
  pivot_longer(cols = 1:8, names_to = "factors", values_to = "values") %>% 
  ggplot(data = ., 
         aes(x = factors, 
             y = values, 
             color = as.factor(cluster), 
             group = cluster,
             shape= as.factor(cluster))) +
  geom_jitter()+
  ggtitle("Cluster Distribution of Factor Analysis")+
  theme(axis.text.x = element_text(angle = 60, hjust = 1))

# weight_matrix with the FA to see the loading 
fa_weight_matrix <- broom::tidy(fa_res) %>% 
  pivot_longer(starts_with("fl"), names_to = "factor", values_to = "loading")
fa_weight_matrix <- dplyr::rename(fa_weight_matrix, comp=factor)
#plotting out the heatmap of FA
fa_loading_plot <- ggplot(fa_weight_matrix, aes(x = comp, y = variable, fill = loading)) +
  geom_tile() + 
  geom_text(aes(label = round(loading,2)), color = "white", size = 4)+
  labs(title = "FA behaviors loadings",
       x = NULL,
       y = NULL) + 
  scico::scale_fill_scico(palette = "cork", limits = c(-1,1)) + 
  coord_fixed(ratio = 1/2)

print(fa_loading_plot)

# for the variables and clusters's plotting 
# we're scale here for interperation from 0 to 1 but will return the value when explain
df_FA_plot <- MinMaxScaler_function(df)
df_FA_plot$cluster <- km_FA[[1]]
# groupby with the cluster and get the mean
FA_group <- data.frame(df_FA_plot %>% group_by(cluster)  %>%
                         summarise(Year_Birth = mean(Year_Birth),
                                   Income = mean(Income),
                                   Kidhome = mean(Kidhome),
                                   Teenhome = mean(Teenhome),
                                   Dt_Customer = mean(Dt_Customer),
                                   Recency = mean(Recency),
                                   MntWines = mean(MntWines),
                                   MntFruits = mean(MntFruits),
                                   MntMeatProducts = mean(MntMeatProducts),
                                   MntFishProducts = mean(MntFishProducts),
                                   MntSweetProducts = mean(MntSweetProducts),
                                   MntGoldProds = mean(MntGoldProds),
                                   NumDealsPurchases = mean(NumDealsPurchases),
                                   NumWebPurchases = mean(NumWebPurchases),
                                   NumCatalogPurchases = mean(NumCatalogPurchases),
                                   NumStorePurchases = mean(NumStorePurchases),
                                   NumWebVisitsMonth = mean(NumWebVisitsMonth),
                                   Educationgraduate = mean(Educationgraduate),
                                   Marital_Status2 = mean(Marital_Status2),
                                   .groups = 'drop'))
# melt 
FA_melt <- melt(FA_group, id.vars = 'cluster', 
                measure.vars = c("Year_Birth","Income","Kidhome",
                                 "Teenhome","Dt_Customer","Recency",
                                 "MntWines","MntFruits","MntMeatProducts",
                                 "MntFishProducts","MntSweetProducts","MntGoldProds",
                                 "NumDealsPurchases","NumWebPurchases","NumCatalogPurchases",
                                 "NumStorePurchases","NumWebVisitsMonth","Educationgraduate",
                                 "Marital_Status2"),
                variable_name = 'Factor_Analysis',
                value.name = 'value')

FA_melt$cluster=factor(FA_melt$cluster)

# get the mean of each cluster within variable and plot heatmap and snake plot
# heatmap distribution plot
a = ggplot(FA_melt, aes(x=Factor_Analysis, y=cluster, fill = value)) + 
  geom_tile(color = "white",lwd = 0.2, linetype = 1) +
  coord_fixed()+
  geom_text(aes(label = round(value,2)), color = "white", size = 4)+
  ggtitle("Factor Analysis Average clusters score heatmap")+
  theme(
    axis.title.x=element_blank(),
    axis.text.x = element_text(angle = 90))

# snake plot
b = ggplot(FA_melt, aes(x= Factor_Analysis,
                        y=value,
                        color = cluster,
)) + 
  geom_line(aes(color = cluster, group = cluster)) + 
  theme_minimal()+ 
  ggtitle("Factor Analysis Clusters snake plot")+
  theme(axis.text.x = element_text(angle = 90))
# final output (need to use zoom to see more clearly)
ggpubr::ggarrange(a, b,
                  ncol = 1, nrow = 2)

# using the fviz_cluster to plot overall distribution of the cluster
factoextra::fviz_cluster(km_FA, data = FA_df[, -7],
                         palette = c("#0073C2FF", "#FC4E07", "#07fc4e"),
                         geom = "point",
                         main = "FA Cluster plot",
                         ellipse.type = "convex",
                         ggtheme = theme_bw())
# distribution with single dimantion
FA_df_scale$cluster <- factor(FA_df_scale$cluster)
fa1 = ggdensity(FA_df_scale, x = "Factor1",
                add = "mean", rug = TRUE,
                color = "cluster", fill = "cluster",
                palette = c("#0073C2FF", "#FC4E07", "#07fc4e"))
fa2 = ggdensity(FA_df_scale, x = "Factor2",
                add = "mean", rug = TRUE,
                color = "cluster", fill = "cluster",
                palette = c("#0073C2FF", "#FC4E07", "#07fc4e"))
fa3 = ggdensity(FA_df_scale, x = "Factor3",
                add = "mean", rug = TRUE,
                color = "cluster", fill = "cluster",
                palette = c("#0073C2FF", "#FC4E07", "#07fc4e"))
fa4 = ggdensity(FA_df_scale, x = "Factor4",
                add = "mean", rug = TRUE,
                color = "cluster", fill = "cluster",
                palette = c("#0073C2FF", "#FC4E07", "#07fc4e"))
ggpubr::ggarrange(fa1, fa2, fa3, fa4, ncol = 4, nrow = 1)


