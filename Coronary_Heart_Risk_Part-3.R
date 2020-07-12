# SET WORKING DIRECTORY ---------------------------------------------------

setwd("/Users/olivertang/Desktop/R/DSBA/WK24 - CAPSTONE PROJECT")
chd = read.csv('Coronary_heart_risk_study.csv', sep=',',header=T)




# LIBRARIES ---------------------------------------------------------------

library(markdown)
library(tidyverse)
library(tidyr)
library(DataExplorer)
library(DescTools) # Cramers V
library(ggplot2)
library(corrplot)
library(pander)
library(forcats)
library(Hmisc)
library(pastecs)
library(psych)
library(expss)
library(mice)
library(modeest) # Mode
library(GGally) # EDA
library(caTools) # PCA
# install.packages('caret', dependencies = TRUE)
library(caret) # PCA / K Fold Cross Validation
library(e1071) # PCA
library(DMwR) # SMOTE
library(pROC) # AUC/ROC
library(class) # KNN
library(randomForest) # 
# install.packages("finalfit") # also installs GG Allay
library(e1071) # Naive Bayes
library(rpart) # DECISION TREES
library(randomForest) # Random Forest
# install.packages("xgboost")
library(xgboost) # xgboost
library(finalfit)
library(gridExtra)
library(scales)
library(ggcorrplot)
library(forcats) 
#library(dplyr)
install.packages('BiocManager')



# RENAMING AND CHANGING DATA TYPES ----------------------------------------

# Re-naming variables for ease of use
chd = chd%>% rename(Gender = male, Age = age, Education = education, Smoker = currentSmoker, CigsPerDay = cigsPerDay, BPMeds = BPMeds, Stroke = prevalentStroke, Hypertensive = prevalentHyp, Diabetes = diabetes, Cholesterol.Levels = totChol, SystolicBP = sysBP, DiastolicBP = diaBP, HeartRate = heartRate, Glucose = glucose, TenYr.chd = TenYearCHD)

# Convert columns Gender, Smoker, BPMeds, Stroke, Hypertensive and Diabetes need to be converted into factors for EDA
chd$Gender <-as.factor(chd$Gender)
chd$Smoker <-as.factor(chd$Smoker)
chd$BPMeds <-as.factor(chd$BPMeds)
chd$Stroke <-as.factor(chd$Stroke)
chd$Hypertensive <-as.factor(chd$Hypertensive)
chd$Diabetes <-as.factor(chd$Diabetes)

## Converting Education into ordered factors:
# High school (1), Undergrad(2), Grad(3) and Advanced Profn(4)
chd$Education = factor(chd$Education, levels = c("1", "2", "3", "4"), order = TRUE)

## Converting TenYr.chd into factor variable. 
# (“1”, means “Yes”, “0” means “No”)
chd$TenYr.chd <- as.factor(chd$TenYr.chd)


# DROP EDUCATION VARIABLE -------------------------------------------------

## Education variable serves no real purpose so we will dro it from our dataset going forward. Choronary Heart Disease can strike anyone from all walks of life and is dependent upon lifestyle and medical condition, not level of education.
chd <- chd[ -c(3) ]
colnames(chd)







# # IMBALANCED DATA ---------------------------------------------------------
# 
# # Simple Bar Plot
# counts <- table(chd$TenYearCHD)
# barplot(counts, main="Ten Year CHD Prediction",
#         xlab="Ten Year CHD Prediction", names.arg=c("No", "Yes"))
# 
# # Challenges with the dataset:
# 
# ##Imbalanced classification problem occurs in disease detection when the rate of the disease in the public is very low. The positive class — disease (Ten year CHD positive prediction "1") — is greatly outnumbered by the negative class ("0"). These types of problems are common cases in data science when accuracy is not a good measure for assessing model performance.
# 
# ## The low percentage of people sampled to be predicted in have CHD in ten years time (TenYr.chd) compared to the overall sample is 15.18%. And will imbalance data analysis. And this will cause a problem in machine learning later on. Also variables such as history of stroke, where only 0.58% (or 25 out of 4240); blood pressure history where only 2.92% (or 124 out of 4240); diabetic history where only 2.57% (or 109 out of 4240); are severely imbalanced classes. You can get high overall accuracy without much effort (precision - percent of positive classifications that are truly positive), but without generating any good insights. The overall accuracy might be high, but for the minority class (TenYr.chd), you will have very low recall (percent of truly positive instances that were classified as such).
# 
# 
# ## In situations where we want to detect instances of a minority class, as in our case study to understand why people get choronary heart disease, we are usually concerned more so with recall than precision. 
# 
# ## Becaues intuitively, we know that proclaiming all data points as negative in the detection of CHD is not helpful and, instead, we should focus on identifying the positive cases. The metric our intuition tells us we should maximize is known in statistics as, recall, or the ability of a model to find all the relevant cases within a dataset. The precise definition of recall is the number of true positives divided by the number of true positives plus the number of false negatives. True positives are data point classified as positive by the model that actually are positive (meaning they are correct), and false negatives are data points the model identifies as negative that actually are positive (incorrect). In our CHD prediction case study, true positives are correctly identified people predicted to get CHD within 10 years, and false negatives would be individuals that the model predicts not to get CHD, but in fact actually did get CHD. Recall can be thought as of a model’s ability to find all the data points of interest in a dataset.
# 
# ## The techniques I will use to overcome imbalanced data in this case study:
# 
# # -	SMOTE (Synthetic Minority Over-sampling Technique) 
# 
# # SMOTE methodology can handle class imbalance problems, by 
# # • synthesis of new minority class instances
# # • over-sampling of minority class
# # • under-sampling of majority class, and 
# # • tweaking the cost function to make misclassification of minority instances more important than misclassification of majority instances
# 
# # Classification using class-imbalanced data is biased in favor of the majority class. The bias is even larger for high-dimensional data, where the number of variables greatly exceeds the number of samples. The problem can be attenuated by undersampling or oversampling, which produce class-balanced data. Generally undersampling is helpful, while random oversampling is not. Synthetic Minority Oversampling Technique (SMOTE) is a very popular oversampling method that is widely used to improve random oversampling
# 
# SMOTE TEMPLATE -------------------------------------------------------------------
# smote_dataset <- chd
# # Splitting the ensemble_dataset into the Training set and Test set
# library(caTools)
# set.seed(123)
# split = sample.split(smote_dataset$TenYr.chd, SplitRatio = 0.75)
# training_set_smote = subset(smote_dataset, split == TRUE)
# test_set_smote = subset(smote_dataset, split == FALSE)
# 
# table(smote_dataset$TenYr.chd)
# smote.train <-subset(smote_dataset, split == TRUE)
# smote.test<-subset(smote_dataset, split == FALSE)
# 
# training_set_smote$TenYr.chd <-as.factor(training_set_smote$TenYr.chd)
# balanced.gd <- SMOTE(TenYr.chd ~., training_set_smote, perc.over= 50000, 
#                      k = 5, perc.under = 100)
# 
# table(training_set_smote$TenYr.chd)
# table(balanced.gd$TenYr.chd)
# 
# count_smote <- table(balanced.gd$TenYr.chd)
# barplot(count_smote, main="Ten Year CHD Prediction After SMOTE",
#         xlab="Ten Year CHD Prediction", names.arg=c("No: 257500", "Yes: 258015"))
# 
# # Add this to original dataset and use as when needed
# str(balanced.gd$TenYr.chd)

# MISSING VALUE TREATMENT -------------------------------------------------

## Total of 15.21% of missing values in 
plot_missing(chd)  +
  theme(axis.text = element_text(color = "black", size = '15'))

sum(is.na(chd))

## 1. Group missing values in Education; conver NA to "Other" DROPPED!!
# chd$Education <- fct_explicit_na(chd$Education, na_level = "Other")

######## Missing Pattern visual


explanatory = c("Gender", "Age", "Smoker", "Stroke", "Hypertensive", "Diabetes", "SystolicBP", "DiastolicBP", "HeartRate", "CigsPerDay","BPMeds","Cholesterol.Levels","BMI","Glucose")
dependent = "TenYr.chd"
chd %>% missing_pattern(dependent, explanatory)


####### histogram visual of variables with missing data

hist_1<- ggplot(data=chd, aes(x=CigsPerDay)) + geom_histogram() 
hist_2<- ggplot(data=chd, aes(x=HeartRate)) + geom_histogram() 
hist_3<- ggplot(data=chd, aes(x=BMI)) + geom_histogram() 
hist_4<- ggplot(data=chd, aes(x=Cholesterol.Levels)) + geom_histogram() 
hist_5<- ggplot(data=chd, aes(x=Glucose)) + geom_histogram() 
grid.arrange(hist_1, hist_2, hist_3, hist_4, hist_5, ncol = 3)


## 2. Dealing with missing values for BP Meds
sum(is.na(chd$BPMeds)) 
summary(chd$BPMeds)
plot(chd$BPMeds, xlab="(0=no)         (1=yes)", ylab="Blood Pressure Meds")

# 53 missing values is 1.26% of the total dataset. This is a very small number of observations to worry about. I could have either chose to eliminate them, but In my opinion, it is always better to keep data than to discard it. And when the missing percentage is very low (<5%) for nominal or categorical values, we can replace the missing values with MODE of that variable.

# Calculate Mode for BP Meds
# install.packages("modeest")
modeBPMeds <- (chd$BPMeds)
mfv(modeBPMeds)
chd$BPMeds[is.na(chd$BPMeds)] <- 0 # Convert all NA's to '0'



## 3. Impute strategy for all missing continuos variables
# https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
# https://www.theanalysisfactor.com/missing-data-mechanism/


# Cigarettes
# 29 missing. For Cigarettes smoked per day there are only 29 missing values which is minimal. As the histogram shows more than 50% of cigarettes smoked per day is less than 5 I will simply use the mode to fill in the missing data.
sum(is.na(chd$CigsPerDay)) 
summary(chd$CigsPerDay)
chd$CigsPerDay = ifelse(is.na(chd$CigsPerDay),
                        ave(chd$CigsPerDay, FUN = function(x) mean(x, na.rm = TRUE)),
                        chd$CigsPerDay)


# BMI
# 19 missing. For BMI there are only 19 missing values which is minimal, and is a normal normal distribution. So I will impute the median for the missing values.
sum(is.na(chd$BMI)) 
summary(chd$BMI)
chd$BMI = ifelse(is.na(chd$BMI),
                 ave(chd$BMI, FUN = function(x) mean(x, na.rm = TRUE)),
                 chd$BMI)


# Cholesterol.Levels
# 50 missing. For Cholesterol.Levels there are only 50 missing values which is minimal, and is a normal normal distribution. However, there are a couple of extreme obese data points which will skew the data, so I'm thinking to use the median for the missing values.
sum(is.na(chd$Cholesterol.Levels)) 
summary(chd$Cholesterol.Levels)
chd$Cholesterol.Levels = ifelse(is.na(chd$Cholesterol.Levels),
                                ave(chd$Cholesterol.Levels, FUN = function(x) mean(x, na.rm = TRUE)),
                                chd$Cholesterol.Levels)


# HeartRate
# 1 missing. Could either remove or convert to mean. Makes no difference as it is only 1. I'll choose mean just to keep all the data.
sum(is.na(chd$HeartRate)) 
chd$HeartRate = ifelse(is.na(chd$HeartRate),
                                ave(chd$HeartRate, FUN = function(x) mean(x, na.rm = TRUE)),
                                chd$HeartRate)



# Glucose
## 388 missing. This requires a little more forethought because of 9.15% is a significant portion of missing data. Based on research from UK Diabetes (https://www.diabetes.co.uk/diabetes_care/blood-sugar-level-ranges.html), blood glucose is highly related to diabetes, which is one of the variables in our dataset. There are also different ranges of glucose levels for before and after meals as well as different types of diabetes, type 1 and type 2. 

## Diabetes is also related to other health factors such as BMI,  high blood pressure, etc., and this in turn has knock-on effects, so taking this into consideration makes the situation a little more complicated than simply imputing the average.
# 
# As there are only 388 missing Glucose values, I could manually filter the raw csv file and quickly eyball them for a sanity check and formulate a plan.
# 
# The strategy I have decided to adopt to impute missing glucose values are as follows:

# 1. Filter by all NA is Glucose column to see if there are any patterns. Result was no pattern found.
  
# 2. Exploring the BMI column. As BMI increases, insulin resistance also increases which results in increased blood glucose level in body. Since body weight is associated with BMI, it may be expected that BMI should correlate with blood glucose levels. (https://www.ijcmr.com/uploads/7/7/4/6/77464738/ijcmr_1592.pdf)
  
# 3. When taking the average in the BMI column for all 388 NA's in the Glucose column, the average BMI is 24.94.
  
# 4. Now filter the BMI column for only 24.94 value, we find the Glucose range is between 60 and 100. Actual values are [60, 78, 79, 87, 88, 100].
  
# 5. So, we will take the median of the 5 values above, not the mean, because 100 looks like it will skew the data, so the median will be affected by that.
  
# 6. Result, median = 83. I will imputer the missing values with 83.

sum(is.na(chd$Glucose)) 
summary(chd$Glucose)
chd$Glucose = ifelse(is.na(chd$Glucose),
                        ave(chd$Glucose, FUN = function(x) median(x, na.rm = TRUE)),
                        chd$Glucose)
sum(is.na(chd$Glucose)) 


colSums(is.na(chd)) 
# OUTLIER TREATMENT  --------------------------------------------------------

## check where outliers are
## Visualize frequency distributions for all continous variables using boxplot
p1<- ggplot(data=chd, aes(x=CigsPerDay)) + geom_boxplot(outlier.color = "Red", outlier.size = 1.0) 
p2<- ggplot(data=chd, aes(x=Age)) + geom_boxplot(outlier.color = "Red", outlier.size = 1.0) 
p3<- ggplot(data=chd, aes(x=Cholesterol.Levels)) + geom_boxplot(outlier.color = "Red", outlier.size = 1.0) 
p4<- ggplot(data=chd, aes(x=SystolicBP)) + geom_boxplot(outlier.color = "Red", outlier.size = 1.0) 
p5<- ggplot(data=chd, aes(x=DiastolicBP)) + geom_boxplot(outlier.color = "Red", outlier.size = 1.0)
p6<- ggplot(data=chd, aes(x=BMI)) + geom_boxplot(outlier.color = "Red", outlier.size = 1.0) 
p7<- ggplot(data=chd, aes(x=HeartRate)) + geom_boxplot(outlier.color = "Red", outlier.size = 1.0) 
p8<- ggplot(data=chd, aes(x=Glucose)) + geom_boxplot(outlier.color = "Red", outlier.size = 1.0) 
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, ncol = 3)

## Outliers are unusual values in your dataset, and they can distort statistical analyses and violate their assumptions. They can increase the variability in your data, which decreases statistical power. Consequently, excluding outliers can cause your results to become statistically significant.

## In broad strokes, there are three causes for outliers
## 1) • data entry or measurement errors:
## Errors can occur during measurement and data entry. During data entry, typos can produce weird values. Imagine that wer'e measuring the height of adult men and gather the following dataset.



## 2) • sampling problems and unusual conditions:
## study might accidentally obtain an item or person that is not from the target population. There are several ways this can occur. For example, unusual events or characteristics can occur that deviate from the defined population. Perhaps the experimenter measures the item or subject under abnormal conditions. In other cases, you can accidentally collect an item that falls outside your target population, and, thus, it might have unusual characteristics.


## 3) • natural variation:
## Natural variation can also produce outliers—and it’s not necessarily a problem.
## Distribution of Z-scores for finding outliers.All data distributions have a spread of values. Extreme values can occur, but they have lower probabilities. If your sample size is large enough, you’re bound to obtain unusual values. In a normal distribution, approximately 1 in 340 observations will be at least three standard deviations away from the mean. However, random chance might include extreme values in smaller datasets! In other words, the process or population you’re studying might produce weird values naturally. There’s nothing wrong with these data points. They’re unusual, but they are a normal part of the data distribution.


## The ones I would give considertation removing are: CigsPerDay

## 2. Systolic BP with one oulier at 295, which is nearly 50 .... more that the next least .... 
## 3. Cholesterol Levels 


## 1) CigsPerDay
## 70 Cigarettes smoked per day not only does it stand out, but it’s an extremely rare value. That's equivalent to 3.5 boxes of 20 pack cigarettes. Though not impossible not impossible to achieve, I would remove this one outlier. 
boxplot(chd$CigsPerDay)
summary(chd$CigsPerDay)
cigs_outlier_removed <- chd$CigsPerDay
length(cigs_outlier_removed)

bench <- 20 + 1.5*IQR(chd$CigsPerDay)
bench # 50 is the IQR

cigs_outlier_removed[cigs_outlier_removed > 50]
cigs_outlier_removed[cigs_outlier_removed < 50]

cigs_outlier_removed <- cigs_outlier_removed[cigs_outlier_removed < bench]
summary(cigs_outlier_removed)
boxplot(cigs_outlier_removed)
length(cigs_outlier_removed)


## 2) Cholesterol.Levels
## Levels of 100 to 129 mg/dL are acceptable for people with no health issues but may be of more concern for those with heart disease or heart disease risk factors. A reading of 130 to 159 mg/dL is borderline high and 160 to 189 mg/dL is high. A reading of 190 mg/dL or higher is considered very high. (https://www.medicalnewstoday.com/articles/315900)

## Another study showed 200.8 mg/dl as being the best predictor of mortality at 12 months (https://www.sciencedirect.com/science/article/pii/S0735109703012002)

## Here we have a max of 696 mg/dL. Surely this  an error. If 190 mg/dL is considered high and dangerous, then 696 mg/dL is fatal. Hence I would remove anything 600 mg/dL and over, but keep the others because even though they are outliers, there’s nothing wrong with these data points. They’re unusually high, but they are a normal part of the data distribution.
boxplot(chd$Cholesterol.Levels)
summary(chd$Cholesterol.Levels)
length(chd$Cholesterol.Levels)

Cholesterol_outlier_removed <- chd$Cholesterol.Levels
length(Cholesterol_outlier_removed)

bench <- 599
bench 

Cholesterol_outlier_removed[Cholesterol_outlier_removed > 599]
Cholesterol_outlier_removed[Cholesterol_outlier_removed < 599]

Cholesterol_outlier_removed <- Cholesterol_outlier_removed[Cholesterol_outlier_removed < bench]
summary(Cholesterol_outlier_removed)
boxplot(Cholesterol_outlier_removed)
length(Cholesterol_outlier_removed)


## 3) Systolic / Diastolic Blood Pressure / BMI / Heart Rate / Glucose:
## These are all a natural variation of the dataset and we need to keep these for predictions.
## Natural variation can also produce outliers—and it’s not necessarily a problem.






# MULTI-COLLINEARITY PCA TEMPLATE ------------------------------------------------------

### pre-processing data for PCA:
PCA_dataset <- chd # assign dataset new name
PCA_dataset <- PCA_dataset[ -c(3) ]
# FEATURE SCALING FOR PCA TEMPLATE --------------------------------------------------------

## Features for scaling needs to be numeric for PCA
PCA_dataset$TenYr.chd<-as.numeric(PCA_dataset$TenYr.chd)
PCA_dataset$Age<-as.numeric(PCA_dataset$Age)
PCA_dataset$Gender<-as.numeric(PCA_dataset$Gender)
PCA_dataset$Smoker<-as.numeric(PCA_dataset$Smoker)
PCA_dataset$BPMeds<-as.numeric(PCA_dataset$BPMeds)
PCA_dataset$Stroke<-as.numeric(PCA_dataset$Stroke)
PCA_dataset$Hypertensive<-as.numeric(PCA_dataset$Hypertensive)
PCA_dataset$Diabetes<-as.numeric(PCA_dataset$Diabetes)

### TRAINING / TEST SET SPLIT FOR PCA 

set.seed(123) # keeps things consistent by using the same starting point
# Splitting the dataset into the Training set and Test set for PCA
split = sample.split(PCA_dataset$TenYr.chd, SplitRatio = 0.8)
training_set_pca = subset(PCA_dataset, split == TRUE)
test_set_pca = subset(PCA_dataset, split == FALSE)


str(PCA_dataset)
## Feature scaling for PCA
training_set_pca[-15] = scale(training_set_pca[-15])
test_set_pca[-15] = scale(test_set_pca[-15])

# APPLYING PCA ------------------------------------------------------------

pca = preProcess(x = training_set_pca[-15], method = 'pca', pcaComp = 3) 
# tinker with the pcaComp between 2-4 and see which is the most accurate
# NB: everytime you run the pca do it from the split test/train line 502
training_set_pca = predict(pca, training_set_pca)
training_set_pca = training_set_pca[c(2, 3, 4, 1)] # swap the colums
test_set_pca = predict(pca, test_set_pca)
test_set_pca = test_set_pca[c(2, 3, 4, 1)] # swap the colums




















##############  MODELLING SECTION -------------------------------------------------------










############## 1. LOGISTIC REGRESSION ----------------------------------------------------

# Assign new dataset
logistic_dataset = chd
str(logistic_dataset)

# dont' forget to treat outliers and missing values, but keep stystolic and diastolic

## ENCODING CATEGORICAL DATA ####### 
# Encoding the target feature as factor
logistic_dataset$TenYr.chd = factor(logistic_dataset$TenYr.chd, levels = c(0, 1), labels = c(0, 1))

# Encoding categorical data
logistic_dataset$Gender = factor(logistic_dataset$Gender, levels = c(0, 1), labels = c(0, 1))
logistic_dataset$Smoker = factor(logistic_dataset$Smoker, levels = c(0, 1), labels = c(0, 1))
logistic_dataset$BPMeds = factor(logistic_dataset$BPMeds, levels = c(0, 1), labels = c(0, 1))
logistic_dataset$Stroke = factor(logistic_dataset$Stroke, levels = c(0, 1), labels = c(0, 1))
logistic_dataset$Hypertensive = factor(logistic_dataset$Hypertensive, levels = c(0, 1), labels = c(0, 1))
logistic_dataset$Diabetes = factor(logistic_dataset$Diabetes, levels = c(0, 1), labels = c(0, 1))

sum(is.na(logistic_dataset)) # Double check no Missing Values


## TRAINNING / TEST SET SPILT ####### 

## Feature scaling needs to be numeric
logistic_dataset$Gender<-as.numeric(logistic_dataset$Gender)
logistic_dataset$Smoker<-as.numeric(logistic_dataset$Smoker)
logistic_dataset$BPMeds<-as.numeric(logistic_dataset$BPMeds)
logistic_dataset$Stroke<-as.numeric(logistic_dataset$Stroke)
logistic_dataset$Hypertensive<-as.numeric(logistic_dataset$Hypertensive)
logistic_dataset$Diabetes<-as.numeric(logistic_dataset$Diabetes)
# logistic_dataset$TenYr.chd<-as.numeric(logistic_dataset$TenYr.chd)

sum(is.na(logistic_dataset)) # Double check no Missing Values

str(logistic_dataset)

# install.packages('caTools')
# library(caTools)
set.seed(123)
split = sample.split(logistic_dataset$TenYr.chd, SplitRatio = 0.75)
training_set_logistic = subset(logistic_dataset, split == TRUE)
test_set_logistic = subset(logistic_dataset, split == FALSE)


## FEATURE SCALING ####### 
training_set_logistic[-15] = scale(training_set_logistic[-15])
test_set_logistic[-15] = scale(test_set_logistic[-15])
# str(training_set_logistic)

sum(is.na(logistic_dataset)) # Double check no Missing Values



## FIT THE LOGISTIC CLASSIFIER MODEL ####### 
# Fitting Logistic Regression to the Training set
classifier = glm(formula = TenYr.chd ~ .,
                 family = binomial,
                 data = training_set_logistic)

summary(classifier) # interpret for report

## LOGISTIC REGRESSION PREDICTIONS ####### 
# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set_logistic[-15])
y_pred = ifelse(prob_pred> 0.5, 1, 0)


## CONFUSION MATRIX BEFORE K-FOLD CROSS VALIDATION ####### 
threshold=0.5 # try 0.95 higher saftey net (how to choose threshold)
cm = table(test_set_logistic[, 15], y_pred) 
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')

# https://statinfer.com/203-4-2-calculating-sensitivity-and-specificity-in-r/
sensitivity(cm) # 0.8586538
specificity(cm) # 0.7


## APPLYING K-FOLD CROSS VALIDATION FOR ACCURACY ----------------------------------------
folds = createFolds(training_set_logistic$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set_logistic[-x, ]
  test_fold = training_set_logistic[x, ]
  classifier = glm(formula = TenYr.chd ~ .,
                   family = binomial,
                   data = training_fold)
  prob_pred = predict(classifier, type = 'response', newdata = test_fold[-15])
  y_pred = ifelse(prob_pred> 0.5, 1, 0)
  # cm = table(test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
cv
accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
accuracy # 0.9080182%

# threshold=0.5 # try 0.95 higher saftey net (how to choose threshold)
# # cm = table(test_set_logistic[, 15], y_pred)
# names(dimnames(cm)) <- c("predicted", "observed")
# confusionMatrix(cm, positive='1')

## APPLYING K-FOLD CROSS VALIDATION FOR CONFUSION MATRIX ----------------------------------------
folds = createFolds(training_set_logistic$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set_logistic[-x, ]
  test_fold = training_set_logistic[x, ]
  classifier = glm(formula = TenYr.chd ~ .,
                   family = binomial,
                   data = training_fold)
  prob_pred = predict(classifier, type = 'response', newdata = test_fold[-15])
  y_pred = ifelse(prob_pred> 0.5, 1, 0)
  cm = table(test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(cm)
})
# cv
# accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
# accuracy # 0.922507%

# threshold=0.5 # try 0.95 higher saftey net (how to choose threshold)
# cm = table(test_set_logistic[, 15], y_pred)
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')


## AUC / ROC ####### 
set.seed(123)
test_set_logistic[, 15] <- sort(test_set_logistic[, 15])
plot(x=test_set_logistic[, 15] , y=y_pred)

## fit a logistic regression to the data...
glm.fit=glm(y_pred ~ test_set_logistic[, 15], family=binomial)
lines(test_set_logistic[, 15], glm.fit$fitted.values)

roc(y_pred, glm.fit$fitted.values, plot=TRUE)
par(pty = "s") 
roc(y_pred, glm.fit$fitted.values, plot=TRUE)

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE)

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage")

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4)

## We can calculate the area under the curve...
roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

## ...and the partial area under the curve.
# roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90), auc.polygon = TRUE, auc.polygon.col = "#377eb822")

# AUC: 52.60%














############## 2. LOGISTIC REGRESSION WITH SMOTE  --------------------------
# (https://rpubs.com/abhaypadda/smote-for-imbalanced-data)

# use SMOTE to handle class imbalance and then I’ve use Logistic and Random Forest to predict if the probability of Ten Year CHD.

## Remove rows that do not have target variable values (already done in missing value treatment)

logistic_dataset <- chd
colnames(logistic_dataset)

## ENCODING CATEGORICAL DATA ####### 

# Encoding the target feature as factor
logistic_dataset$TenYr.chd = factor(logistic_dataset$TenYr.chd, levels = c(0, 1), labels = c(0, 1))

# Encoding categorical data
logistic_dataset$Gender = factor(logistic_dataset$Gender, levels = c(0, 1), labels = c(0, 1))
logistic_dataset$Smoker = factor(logistic_dataset$Smoker, levels = c(0, 1), labels = c(0, 1))
logistic_dataset$BPMeds = factor(logistic_dataset$BPMeds, levels = c(0, 1), labels = c(0, 1))
logistic_dataset$Stroke = factor(logistic_dataset$Stroke, levels = c(0, 1), labels = c(0, 1))
logistic_dataset$Hypertensive = factor(logistic_dataset$Hypertensive, levels = c(0, 1), labels = c(0, 1))
logistic_dataset$Diabetes = factor(logistic_dataset$Diabetes, levels = c(0, 1), labels = c(0, 1))

sum(is.na(logistic_dataset)) # Double check no Missing Values


## Feature scaling needs to be numeric
logistic_dataset$Gender<-as.numeric(logistic_dataset$Gender)
logistic_dataset$Smoker<-as.numeric(logistic_dataset$Smoker)
logistic_dataset$BPMeds<-as.numeric(logistic_dataset$BPMeds)
logistic_dataset$Stroke<-as.numeric(logistic_dataset$Stroke)
logistic_dataset$Hypertensive<-as.numeric(logistic_dataset$Hypertensive)
logistic_dataset$Diabetes<-as.numeric(logistic_dataset$Diabetes)
# logistic_dataset$TenYr.chd<-as.numeric(logistic_dataset$TenYr.chd)

sum(is.na(logistic_dataset)) # Double check no Missing Values



## SMOTE ####### 
## Loading DMwr to balance the unbalanced class
library(DMwR)

## Smote : Synthetic Minority Oversampling Technique To Handle Class Imbalancy In Binary Classification
balanced.data <- SMOTE(TenYr.chd ~., logistic_dataset, perc.over = 4800, k = 5, perc.under = 1000)

## SPLIT TRAIN/TEST SET #######  

set.seed(123)
split = sample.split(balanced.data$TenYr.chd, SplitRatio = 0.75)
training = subset(balanced.data, split == TRUE)
test = subset(balanced.data, split == FALSE)


## Let's check the count of unique value in the target variable
as.data.frame(table(balanced.data$TenYr.chd))

## FIT CLASSIFIER TO LOGISTIC REGRESSION ####### 

# Fitting Logistic Regression to the Training set
classifier_smote = glm(formula = TenYr.chd ~ .,
                   family = binomial,
                   data = training)

summary(classifier_smote) # interpret for report

## LOGISTIC REGRESSION SMOTE PREDICTIONS ####### 

# Predicting the Test set results
prob_pred_smote = predict(classifier_smote, type = 'response', newdata = test[-15])
y_pred_smote = ifelse(prob_pred_smote> 0.75, 1, 0)


## CONFUSION MATRIX BEFORE K-FOLD CROSS VALIDATION ####### 
threshold=0.75 # try 0.95 higher saftey net (how to choose threshold)
cm = table(test[,15], y_pred_smote) 
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')



# https://statinfer.com/203-4-2-calculating-sensitivity-and-specificity-in-r/
# sensitivity(cm)
# specificity(cm)

## APPLYING K-FOLD CROSS VALIDATION FOR ACCURACY ----------------------------------------
folds = createFolds(training$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training[-x, ]
  test_fold = training[x, ]
  classifier_smote = glm(formula = TenYr.chd ~ .,
                         family = binomial,
                         data = training_fold)
  prob_pred_smote = predict(classifier_smote, type = 'response', newdata = test_fold[-15])
  y_pred_smote = ifelse(prob_pred_smote> 0.75, 1, 0)
  # cm = table(test_fold[, 15], y_pred_smote)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  # return(cm)
  return(accuracy)
})
cv
accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
accuracy # 85.28%

# cm = table(test_fold[,15], y_pred_smote)
# names(dimnames(cm)) <- c("predicted", "observed")
# confusionMatrix(cm, positive='1')




## APPLYING K-FOLD CROSS VALIDATION FOR CONFUSION MATRIX ----------------------------------------
folds = createFolds(training$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training[-x, ]
  test_fold = training[x, ]
  classifier_smote = glm(formula = TenYr.chd ~ .,
                         family = binomial,
                         data = training_fold)
  prob_pred_smote = predict(classifier_smote, type = 'response', newdata = test_fold[-15])
  y_pred_smote = ifelse(prob_pred_smote> 0.75, 1, 0)
  cm = table(test_fold[, 15], y_pred_smote)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(cm)
  # return(accuracy)
})
# cv
# accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
# accuracy # 85.28%

# cm = table(test_fold[, 15], y_pred_smote)
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')



## AUC / ROC ####### 
set.seed(123)
test[, 15] <- sort(test[, 15])
plot(x=test[, 15] , y=y_pred_smote)

## fit a logistic regression to the data...
glm.fit=glm(y_pred_smote ~ test[, 15], family=binomial)
lines(test[, 15], glm.fit$fitted.values)

roc(y_pred_smote, glm.fit$fitted.values, plot=TRUE)
par(pty = "s") 
roc(y_pred_smote, glm.fit$fitted.values, plot=TRUE)

roc(y_pred_smote, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE)

roc(y_pred_smote, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage")

roc(y_pred_smote, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4)

## We can calculate the area under the curve...
roc(y_pred_smote, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

## ...and the partial area under the curve.
roc(y_pred_smote, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90), auc.polygon = TRUE, auc.polygon.col = "#377eb822")

# AUC: 52.90%

























############## 4. KNN  ---------------------------------------------------------------------

# Classification accuracy of the kNN algorithm is found to be adversely affected by the presence of outliers in the experimental datasets

# KNN is an algorithm that is useful for matching a point with its closest k neighbors in a multi-dimensional space. It can be used for data that are continuous, discrete, ordinal and categorical which makes it particularly useful for dealing with all kind of missing data. (TRY WITH AND WITHOUT MISSING VALUES TO COMPARE)

KNNdataset <- chd

## ENCODING TARGET VARIABLE AS A FACTOR  ####### 
KNNdataset$TenYr.chd = factor(KNNdataset$TenYr.chd, levels = c(0, 1), labels = c(0, 1))

# Encoding categorical data
KNNdataset$Gender = factor(KNNdataset$Gender, levels = c(0, 1), labels = c(0, 1))
KNNdataset$Smoker = factor(KNNdataset$Smoker, levels = c(0, 1), labels = c(0, 1))
KNNdataset$BPMeds = factor(KNNdataset$BPMeds, levels = c(0, 1), labels = c(0, 1))
KNNdataset$Stroke = factor(KNNdataset$Stroke, levels = c(0, 1), labels = c(0, 1))
KNNdataset$Hypertensive = factor(KNNdataset$Hypertensive, levels = c(0, 1), labels = c(0, 1))
KNNdataset$Diabetes = factor(KNNdataset$Diabetes, levels = c(0, 1), labels = c(0, 1))

## FEATURES FOR SCALING NEEDS TO BE NUMERIC ####### 
KNNdataset$TenYr.chd<-as.numeric(KNNdataset$TenYr.chd)
KNNdataset$Age<-as.numeric(KNNdataset$Age)
KNNdataset$Gender<-as.numeric(KNNdataset$Gender)
KNNdataset$Smoker<-as.numeric(KNNdataset$Smoker)
KNNdataset$BPMeds<-as.numeric(KNNdataset$BPMeds)
KNNdataset$Stroke<-as.numeric(KNNdataset$Stroke)
KNNdataset$Hypertensive<-as.numeric(KNNdataset$Hypertensive)
KNNdataset$Diabetes<-as.numeric(KNNdataset$Diabetes)




## SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET ####### 
# library(caTools)
set.seed(123)
split = sample.split(KNNdataset$TenYr.chd, SplitRatio = 0.75)
training_set_knn = subset(KNNdataset, split == TRUE)
test_set_knn = subset(KNNdataset, split == FALSE)



## FEATURE SCALING ####### 
training_set_knn[-15] = scale(training_set_knn[-15])
test_set_knn[-15] = scale(test_set_knn[-15])

## FITTING K-NN TO THE TRAINING SET AND PREDICTING THE TEST SET RESULTS ####### 
# library(class)
y_pred = knn(train = training_set_knn[-15],
             test = test_set_knn[, -15],
             cl = training_set_knn[, 15],
             k = 5,
             prob = TRUE)

## CONFUSION MATRIX BEFORE K-FOLD CROSS VALIDATION ####### 
cm = table(test_set_knn[, 15], y_pred)
confusionMatrix(cm, positive='1')


## APPLYING K-FOLD CROSS VALIDATION FOR ACCURACY ----------------------------------------
folds = createFolds(training_set_knn$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set_knn[-x, ]
  test_fold = training_set_knn[x, ]
  y_pred = knn(train = training_fold[-15],
               test = test_fold[, -15],
               cl = training_fold[, 15],
               k = 5,
               prob = TRUE)
  # cm = table(test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  # return(cm)
  return(accuracy)
})
accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
accuracy # 0.8339623%
# cm
# names(dimnames(cm)) <- c("predicted", "observed")
# confusionMatrix(cm, positive='1')






## APPLYING K-FOLD CROSS VALIDATION FOR CONFUSION MATRIX ----------------------------------------
folds = createFolds(training_set_knn$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set_knn[-x, ]
  test_fold = training_set_knn[x, ]
  y_pred = knn(train = training_fold[-15],
               test = test_fold[, -15],
               cl = training_fold[, 15],
               k = 5,
               prob = TRUE)
  cm = table(test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(cm)
  # return(accuracy)
})
# accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
# accuracy # 0.922507%
cm
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')







## AUC / ROC ####### 
test_set_knn[, 15] <- sort(test_set_knn[, 15])
plot(x=test_set_knn[, 15] , y=y_pred)

## fit a logistic regression to the data...
glm.fit=glm(y_pred ~ test_set_knn[, 15], family=binomial)
lines(test_set_knn[, 15], glm.fit$fitted.values)


roc(y_pred, glm.fit$fitted.values, plot=TRUE)
par(pty = "s") 
roc(y_pred, glm.fit$fitted.values, plot=TRUE)

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE)

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage")

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4)

## We can calculate the area under the curve...
roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

## ...and the partial area under the curve.
# roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90), auc.polygon = TRUE, auc.polygon.col = "#377eb822")
# 


# AUC:52.20%



































############## 5. KNN WITH SMOTE  ----------------------------------------------------------
# KNN is susceptible to class imbalance, as described well here: https://www.quora.com/Why-does-knn-get-effected-by-the-class-imbalance

# Given that this dataset is unbalanced, i.e. there are more 0s (not predicited to get CHD in ten years) than 1s (preddicted to get CHD in ten years), the Synthetic Minority Oversampling Technique (SMOTE) is used to balance the classes in order to apply the KNN algorithm.

smote_dataset <- chd
## ENCODING TARGET VARIABLE AS A FACTOR  ####### 
smote_dataset$TenYr.chd = factor(smote_dataset$TenYr.chd, levels = c(0, 1), labels = c(0, 1))

# Encoding categorical data
smote_dataset$Gender = factor(smote_dataset$Gender, levels = c(0, 1), labels = c(0, 1))
smote_dataset$Smoker = factor(smote_dataset$Smoker, levels = c(0, 1), labels = c(0, 1))
smote_dataset$BPMeds = factor(smote_dataset$BPMeds, levels = c(0, 1), labels = c(0, 1))
smote_dataset$Stroke = factor(smote_dataset$Stroke, levels = c(0, 1), labels = c(0, 1))
smote_dataset$Hypertensive = factor(smote_dataset$Hypertensive, levels = c(0, 1), labels = c(0, 1))
smote_dataset$Diabetes = factor(smote_dataset$Diabetes, levels = c(0, 1), labels = c(0, 1))

## FEATURES FOR SCALING NEEDS TO BE NUMERIC ####### 
# smote_dataset$TenYr.chd<-as.numeric(smote_dataset$TenYr.chd)
smote_dataset$Age<-as.numeric(smote_dataset$Age)
smote_dataset$Gender<-as.numeric(smote_dataset$Gender)
smote_dataset$Smoker<-as.numeric(smote_dataset$Smoker)
smote_dataset$BPMeds<-as.numeric(smote_dataset$BPMeds)
smote_dataset$Stroke<-as.numeric(smote_dataset$Stroke)
smote_dataset$Hypertensive<-as.numeric(smote_dataset$Hypertensive)
smote_dataset$Diabetes<-as.numeric(smote_dataset$Diabetes)




## SMOTE ####### 
## Loading DMwr to balance the unbalanced class
library(DMwR)

## Smote : Synthetic Minority Oversampling Technique To Handle Class Imbalancy In Binary Classification
balanced.data <- SMOTE(TenYr.chd ~., smote_dataset, perc.over = 4800, k = 5, perc.under = 1000)

## SPLIT TRAIN/TEST SET #######  

set.seed(123)
split = sample.split(balanced.data$TenYr.chd, SplitRatio = 0.75)
training = subset(balanced.data, split == TRUE)
test = subset(balanced.data, split == FALSE)


## Let's check the count of unique value in the target variable
as.data.frame(table(balanced.data$TenYr.chd))


# Add this to original dataset and use as when needed
str(balanced.data)
summary(balanced.data)


## Let's check the count of unique value in the target variable
# as.data.frame(table(balanced.gd$TenYr.chd))


## FITTING K-NN WITH SMOTE TO THE TRAINING SET AND PREDICTING THE TEST SET RESULTS ####### 
# library(class)
y_pred_knn_smote = knn(train = training[, -15],
                   test = test[, -15],
                   cl = training[, 15],
                   k = 5,
                   prob = TRUE)

## CONFUSION MATRIX BEFORE K-FOLD CROSS VALIDATION  ####### 
cm = table(test[, 15], y_pred_knn_smote)
confusionMatrix(cm, positive='1')
# https://statinfer.com/203-4-2-calculating-sensitivity-and-specificity-in-r/
sensitivity(cm)
specificity(cm)


## APPLYING K-FOLD CROSS VALIDATION FOR ACCURACY ----------------------------------------
folds = createFolds(training$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training[-x, ]
  test_fold = training[x, ]
  y_pred_knn_smote = knn(train = training_fold[, -15],
                         test = test_fold[, -15],
                         cl = training_fold[, 15],
                         k = 5,
                         prob = TRUE)
  # cm = table(test_fold[, 15], y_pred_knn_smote)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  # return(cm)
  return(accuracy)
})
accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
accuracy # 0.922507%
# cm
# names(dimnames(cm)) <- c("predicted", "observed")
# confusionMatrix(cm, positive='1')


## APPLYING K-FOLD CROSS VALIDATION FOR CONFUSION MATRIX ----------------------------------------
folds = createFolds(training$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training[-x, ]
  test_fold = training[x, ]
  y_pred_knn_smote = knn(train = training_fold[, -15],
                         test = test_fold[, -15],
                         cl = training_fold[, 15],
                         k = 5,
                         prob = TRUE)
  cm = table(test_fold[, 15], y_pred_knn_smote)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(cm)
  # return(accuracy)
})
# accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
# accuracy # 0.922507%
cm
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')





## AUC / ROC ####### 
test[, 15] <- sort(test[, 15])
plot(x=test[, 15], y=y_pred_knn_smote)

## fit a logistic regression to the data...
glm.fit=glm(y_pred_knn_smote ~ test[, 15], family=binomial)
lines(test[, 15], glm.fit$fitted.values)


roc(y_pred_knn_smote, glm.fit$fitted.values, plot=TRUE)
par(pty = "s") 
roc(y_pred_knn_smote, glm.fit$fitted.values, plot=TRUE)

roc(y_pred_knn_smote, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE)

roc(y_pred_knn_smote, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage")

roc(y_pred_knn_smote, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4)

## We can calculate the area under the curve...
roc(y_pred_knn_smote, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

## ...and the partial area under the curve.
roc(y_pred_knn_smote, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90), auc.polygon = TRUE, auc.polygon.col = "#377eb822")




# AUC: 51.40%

















############## 6. NAIVE BAYES  ----------------------------------------------

nb_data <- chd # assign dataset new name
sum(is.na(nb_data))


## ENCODING TARGET VARIABLE AS A FACTOR  ####### 
nb_data$TenYr.chd = factor(nb_data$TenYr.chd, levels = c(0, 1), labels = c(0, 1))

# Encoding others as categorical data
nb_data$Gender = factor(nb_data$Gender, levels = c(0, 1), labels = c(0, 1))
nb_data$Smoker = factor(nb_data$Smoker, levels = c(0, 1), labels = c(0, 1))
nb_data$BPMeds = factor(nb_data$BPMeds, levels = c(0, 1), labels = c(0, 1))
nb_data$Stroke = factor(nb_data$Stroke, levels = c(0, 1), labels = c(0, 1))
nb_data$Hypertensive = factor(nb_data$Hypertensive, levels = c(0, 1), labels = c(0, 1))
nb_data$Diabetes = factor(nb_data$Diabetes, levels = c(0, 1), labels = c(0, 1))



## FEATURES SCALING ####### 
# Scaling is not required while modelling trees. Algorithms like Linear Discriminant Analysis(LDA), Naive Bayes are by design equipped to handle this and gives weights to the features accordingly. Performing a features scaling in these algorithms may not have much effect. But we can still do feature scalling and compare the results

## FEATURES FOR SCALING NEEDS TO BE NUMERIC ####### 
# nb_data$TenYr.chd<-as.numeric(nb_data$TenYr.chd)
nb_data$Age<-as.numeric(nb_data$Age)
nb_data$Gender<-as.numeric(nb_data$Gender)
nb_data$Smoker<-as.numeric(nb_data$Smoker)
nb_data$BPMeds<-as.numeric(nb_data$BPMeds)
nb_data$Stroke<-as.numeric(nb_data$Stroke)
nb_data$Hypertensive<-as.numeric(nb_data$Hypertensive)
nb_data$Diabetes<-as.numeric(nb_data$Diabetes)

# it is very much sensitive to outliers with the classical estimates of the location and scale parameters. Already done in the pre-processing part





## SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET ####### 
str(nb_data)
# Splitting the nb_data into the Training set and Test set
set.seed(123)
split = sample.split(nb_data$TenYr.chd, SplitRatio = 0.75)
training_set_nb_data = subset(nb_data, split == TRUE)
test_set_nb_data = subset(nb_data, split == FALSE)


## Feature Scaling ####### 
training_set_nb_data[-15] = scale(training_set_nb_data[-15])
test_set_nb_data[-15] = scale(test_set_nb_data[-15])

# Taking care of missing data. Already done in pre-processing part
# training_set_nb_data = na.omit(training_set_nb_data) 
# test_set_nb_data = na.omit(test_set_nb_data) 

## FIT NAIVE BAYES TO TRAINING SET ####### 
classifier = naiveBayes(x = training_set_nb_data[-15],
                        y = training_set_nb_data$TenYr.chd)

## PREDICTING THE TEST SET RESULTS ####### 
y_pred_nb = predict(classifier, newdata = test_set_nb_data[-15])

## CONFUSION MATRIX BEFORE K-FOLD CROSS VALIDATION ####### 
cm = table(test_set_nb_data[, 15], y_pred_nb)
cm
confusionMatrix(test_set_nb_data[, 15], y_pred_nb, positive='1')

## APPLYING K-FOLD CROSS VALIDATION FOR ACCURACY ----------------------------------------
folds = createFolds(training_set_nb_data$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set_nb_data[-x, ]
  test_fold = training_set_nb_data[x, ]
  classifier = naiveBayes(x = training_fold[-15],
                          y = training_fold$TenYr.chd)
  y_pred_nb = predict(classifier, newdata = test_fold [-15])
  # cm = table(test_fold[, 15], y_pred_nb)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  # return(cm)
  return(accuracy)
})
cv
accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
accuracy # 0.922507%







## APPLYING K-FOLD CROSS VALIDATION FOR CONFUSION MATRIX ----------------------------------------
folds = createFolds(training_set_nb_data$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set_nb_data[-x, ]
  test_fold = training_set_nb_data[x, ]
  classifier = naiveBayes(x = training_fold[-15],
                          y = training_fold$TenYr.chd)
  y_pred_nb = predict(classifier, newdata = test_fold [-15])
  cm = table(test_fold[, 15], y_pred_nb)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(cm)
  # return(accuracy)
})
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')









## AUC / ROC ####### 
set.seed(123)
test_set_nb_data[, 15] <- sort(test_set_nb_data[, 15])
plot(x=test_set_nb_data[, 15] , y=y_pred_nb)

## fit a logistic regression to the data...
glm.fit=glm(y_pred_nb ~ test_set_nb_data[, 15], family=binomial)
lines(test_set_nb_data[, 15] , glm.fit$fitted.values)

roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE)
par(pty = "s") 
roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE)

roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE)

roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage")

roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4)

## We can calculate the area under the curve...
roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

## ...and the partial area under the curve.
# roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90), auc.polygon = TRUE, auc.polygon.col = "#377eb822")



# AUC:52.51%


















############## 7. NAIVE BAYES WITH SMOTE ---------------------------------------------------

balanced.data <- chd # start with new re-named dataset

## ENCODING TARGET VARIABLE AS A FACTOR #########

balanced.data$TenYr.chd = factor(balanced.data$TenYr.chd, levels = c(0, 1), labels = c(0, 1))

# Encoding others as categorical data
balanced.data$Gender = factor(balanced.data$Gender, levels = c(0, 1), labels = c(0, 1))
balanced.data$Smoker = factor(balanced.data$Smoker, levels = c(0, 1), labels = c(0, 1))
balanced.data$BPMeds = factor(balanced.data$BPMeds, levels = c(0, 1), labels = c(0, 1))
balanced.data$Stroke = factor(balanced.data$Stroke, levels = c(0, 1), labels = c(0, 1))
balanced.data$Hypertensive = factor(balanced.data$Hypertensive, levels = c(0, 1), labels = c(0, 1))
balanced.data$Diabetes = factor(balanced.data$Diabetes, levels = c(0, 1), labels = c(0, 1))

## FEATURES FOR SCALING NEEDS TO BE NUMERIC ####### 
balanced.data$Age<-as.factor(balanced.data$Age)
balanced.data$Gender<-as.factor(balanced.data$Gender)
balanced.data$Smoker<-as.factor(balanced.data$Smoker)
balanced.data$BPMeds<-as.factor(balanced.data$BPMeds)
balanced.data$Stroke<-as.factor(balanced.data$Stroke)
balanced.data$Hypertensive<-as.factor(balanced.data$Hypertensive)
balanced.data$Diabetes<-as.factor(balanced.data$Diabetes)

## SMOTE ####### 
set.seed(123)
split = sample.split(balanced.data$TenYr.chd, SplitRatio = 0.75)
training = subset(balanced.data, split == TRUE)
test = subset(balanced.data, split == FALSE)


## Let's check the count of unique value in the target variable
as.data.frame(table(balanced.data$TenYr.chd))


# Add this to original dataset and use as when needed
# str(balanced.data)
# summary(balanced.data)


## FIT NAIVE BAYES SMOTE TO TRAINING SET ####### 
classifier = naiveBayes(x = training[-15],
                        y = training$TenYr.chd)

## PREDICTING THE TEST SET RESULTS ####### 
y_pred_nb = predict(classifier, newdata = test[-15])

## CONFUSION MATRIX MATRIX BEFORE K-FOLD CROSS VALIDATION ####### 
cm = table(balanced.data[, 15], y_pred_nb)
cm
confusionMatrix(balanced.data[, 15], y_pred_nb)

## APPLYING K-FOLD CROSS VALIDATION FOR ACCURACY ----------------------------------------
folds = createFolds(training$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training[-x, ]
  test_fold = training[x, ]
  classifier = naiveBayes(x = training_fold[-15],
                          y = training_fold$TenYr.chd)
  y_pred_nb = predict(classifier, newdata = test_fold[-15])
  # cm = table(test_fold[, 15], y_pred_knn_smote)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  # return(cm)
  return(accuracy)
})
accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
accuracy # 0.922507%


## APPLYING K-FOLD CROSS VALIDATION FOR CONFUSION MATRIX ----------------------------------------
folds = createFolds(training$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training[-x, ]
  test_fold = training[x, ]
  classifier = naiveBayes(x = training_fold[-15],
                          y = training_fold$TenYr.chd)
  y_pred_nb = predict(classifier, newdata = test_fold[-15])
  cm = table(test_fold[, 15], y_pred_nb)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(cm)
  # return(accuracy)
})
cm
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')



## AUC / ROC ####### 
set.seed(123)
test[, 15] <- sort(test[, 15])
plot(x=test[, 15] , y=y_pred_nb)

## fit a logistic regression to the data...
glm.fit=glm(y_pred_nb ~ test[, 15], family=binomial)
lines(test[,15] , glm.fit$fitted.values)



roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE)
par(pty = "s") 
roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE)

roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE)

roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage")

roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4)

## We can calculate the area under the curve...
roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

## ...and the partial area under the curve.
# roc(y_pred_nb, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90), auc.polygon = TRUE, auc.polygon.col = "#377eb822")



# AUC:52.13%









############## 8. NAIVE BAYES WITH PCA --------------------------------------------------------
## MULTI-COLLINEARITY PCA TEMPLATE ######### 

### pre-processing data for PCA:
PCA_dataset <- chd # assign dataset new name

# FEATURE TRANSFORMATION FOR PCA --------------------------------------------------------

# Encoding the target feature as factor
PCA_dataset$TenYr.chd = factor(PCA_dataset$TenYr.chd, levels = c(0, 1), labels = c(0, 1))

## Features needs to be numeric for PCA
# PCA_dataset$TenYr.chd<-as.numeric(PCA_dataset$TenYr.chd)
PCA_dataset$Age<-as.numeric(PCA_dataset$Age)
PCA_dataset$Gender<-as.numeric(PCA_dataset$Gender)
PCA_dataset$Smoker<-as.numeric(PCA_dataset$Smoker)
PCA_dataset$BPMeds<-as.numeric(PCA_dataset$BPMeds)
PCA_dataset$Stroke<-as.numeric(PCA_dataset$Stroke)
PCA_dataset$Hypertensive<-as.numeric(PCA_dataset$Hypertensive)
PCA_dataset$Diabetes<-as.numeric(PCA_dataset$Diabetes)

### TRAINING / TEST SET SPLIT FOR PCA 

set.seed(123) # keeps things consistent by using the same starting point
# Splitting the dataset into the Training set and Test set for PCA
split = sample.split(PCA_dataset$TenYr.chd, SplitRatio = 0.8)
training_set_pca = subset(PCA_dataset, split == TRUE)
test_set_pca = subset(PCA_dataset, split == FALSE)


str(PCA_dataset)
## Feature scaling for PCA
training_set_pca[-15] = scale(training_set_pca[-15])
test_set_pca[-15] = scale(test_set_pca[-15])


# APPLYING PCA ------------------------------------------------------------
pca = preProcess(x = training_set_pca[-15], method = 'pca', pcaComp = 3) 
# tinker with the pcaComp between 2-4 and see which is the most accurate
# NB: everytime you run the pca do it from the split test/train line 502
training_set_pca = predict(pca, training_set_pca)
training_set_pca = training_set_pca[c(2, 3, 4, 1)] # swap the colums

test_set_pca = predict(pca, test_set_pca)
test_set_pca = test_set_pca[c(2, 3, 4, 1)] # swap the colums

## FIT NAIVE BAYES TO TRAINING SET ####### 
classifier = naiveBayes(x = training_set_pca[-4],
                        y = training_set_pca$TenYr.chd)

## PREDICTING THE TEST SET RESULTS ####### 
y_pred = predict(classifier, newdata = test_set_pca[-4])

## CONFUSION MATRIX
cm = table(test_set_pca[, 4], y_pred)
cm
confusionMatrix(test_set_pca[, 4], y_pred)

## APPLYING K-FOLD CROSS VALIDATION FOR ACCURACY ----------------------------------------
folds = createFolds(training_set_pca$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set_pca[-x, ]
  test_fold = training_set_pca[x, ]
  classifier = naiveBayes(x = training_fold[-4],
                          y = training_fold$TenYr.chd)
  y_pred = predict(classifier, newdata = test_fold[-4])
  # cm = table(test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
accuracy # 85%


















## APPLYING K-FOLD CROSS VALIDATION FOR CONFUSION MATRIX  ----------------------------------------
folds = createFolds(training_set_pca$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set_pca[-x, ]
  test_fold = training_set_pca[x, ]
  classifier = naiveBayes(x = training_fold[-4],
                          y = training_fold$TenYr.chd)
  y_pred = predict(classifier, newdata = test_fold[-4])
  # cm = table(test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
# cv
# accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
# accuracy # 85%
cm
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')
















## AUC / ROC ####### 
set.seed(123)
test_set_pca[, 4] <- sort(test_set_pca[, 4])
plot(x=test_set_pca[, 4] , y=y_pred)

## fit a logistic regression to the data...
glm.fit=glm(y_pred ~ test_set_pca[, 4], family=binomial)
lines(test_set_pca[, 4] , glm.fit$fitted.values)

roc(y_pred, glm.fit$fitted.values, plot=TRUE)
par(pty = "s") 
roc(y_pred, glm.fit$fitted.values, plot=TRUE)

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE)

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage")

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4)

## We can calculate the area under the curve...
roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

## ...and the partial area under the curve.
# roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90), auc.polygon = TRUE, auc.polygon.col = "#377eb822")



# AUC:55.10%













############## 9. BAGGING & BOOSTING ---------------------------------------------------
# Decision Tree Classification

tree_dataset <- chd
#  Luckily, decision trees and boosted trees algorithms are immune to multicollinearity by nature (https://towardsdatascience.com/why-feature-correlation-matters-a-lot-847e8ba439c4)
## ENCODING THE TARGET FEATURE AS FACTOR ####### 

tree_dataset$TenYr.chd = factor(tree_dataset$TenYr.chd, levels = c(0, 1))

## FEATURES FOR NEEDS TO BE NUMERIC FOR FEATURE SCALING ####### 
# tree_dataset$Age<-as.numeric(tree_dataset$Age)
tree_dataset$Gender<-as.numeric(tree_dataset$Gender)
tree_dataset$Smoker<-as.numeric(tree_dataset$Smoker)
tree_dataset$BPMeds<-as.numeric(tree_dataset$BPMeds)
tree_dataset$Stroke<-as.numeric(tree_dataset$Stroke)
tree_dataset$Hypertensive<-as.numeric(tree_dataset$Hypertensive)
tree_dataset$Diabetes<-as.numeric(tree_dataset$Diabetes)


## SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET ####### 
set.seed(123)
split = sample.split(tree_dataset$TenYr.chd, SplitRatio = 0.75)
train = subset(tree_dataset, split == TRUE)
test = subset(tree_dataset, split == FALSE)

# Feature Scaling
train[-15] = scale(train[-15])
test[-15] = scale(test[-15])

## BAGGING #####
library(gbm)
library(xgboost)
library(caret)
library(ipred)
library(plyr)
library(rpart)
set.seed(123)
mod.bagging <- bagging(TenYr.chd ~.,
                       data=train,
                       control=rpart.control(maxdepth=5, minsplit=4))

bag.pred <- predict(mod.bagging, test)

confusionMatrix(bag.pred, test$TenYr.chd, positive = '1')

## AUC / ROC  ####### 
set.seed(123)
test[, 15] <- sort(test[, 15])
plot(x=test[, 15] , y=y_pred)

## fit a logistic regression to the data...
glm.fit=glm(y_pred ~ as.numeric(test[,15]), family=binomial)
lines(test[, 15], glm.fit$fitted.values)

roc(y_pred, glm.fit$fitted.values, plot=TRUE)
par(pty = "s") 
roc(y_pred, glm.fit$fitted.values, plot=TRUE)

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE)

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage")

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4)

## We can calculate the area under the curve...
roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

## ...and the partial area under the curve.
# roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90), auc.polygon = TRUE, auc.polygon.col = "#377eb822")


# AUC: 52.93%








## BOOSTING #####
set.seed(123)
train$TenYr.chd <- as.character(train$TenYr.chd)
head(train$TenYr.chd)
mod.boost <- gbm(formula = TenYr.chd ~ .,distribution = "bernoulli",
               data = train,  n.trees = 100,
                interaction.depth = 1,  shrinkage = 0.01,
                cv.folds = 1,  n.cores = 1,  verbose = FALSE)


mod.boost <- gbm(TenYr.chd ~ .,data=train, distribution="bernoulli",
                 n.trees=5000, interaction.depth =4, shrinkage=0.01)

summary(mod.boost)
#boost.pred <- predict(mod.boost, test)
boost.pred <- predict(mod.boost, test, n.trees =5000, type="response")

y_pred_num <- ifelse(boost.pred > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))
confusionMatrix(y_pred, test$TenYr.chd, positive = '1')



## AUC / ROC  ####### 
set.seed(123)
test[, 15] <- sort(test[, 15])
plot(x=test[, 15] , y=y_pred)

## fit a logistic regression to the data...
glm.fit=glm(y_pred ~ as.numeric(test[,15]), family=binomial)
lines(test[, 15], glm.fit$fitted.values)

roc(y_pred, glm.fit$fitted.values, plot=TRUE)
par(pty = "s") 
roc(y_pred, glm.fit$fitted.values, plot=TRUE)

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE)

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage")

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4)

## We can calculate the area under the curve...
roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

## ...and the partial area under the curve.
# roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90), auc.polygon = TRUE, auc.polygon.col = "#377eb822")


# AUC: 52.04%







############## 10. RANDOM FOREST ---------------------------------------------------

RM_dataset <- chd

## ENCODING THE TARGET FEATURE AS FACTOR ######### 
RM_dataset$TenYr.chd = factor(RM_dataset$TenYr.chd, levels = c(0, 1))

## FEATURES FOR NEEDS TO BE NUMERIC FOR FEATURE SCALING ####### 
RM_dataset$Age<-as.numeric(RM_dataset$Age)
RM_dataset$Gender<-as.numeric(RM_dataset$Gender)
RM_dataset$Smoker<-as.numeric(RM_dataset$Smoker)
RM_dataset$BPMeds<-as.numeric(RM_dataset$BPMeds)
RM_dataset$Stroke<-as.numeric(RM_dataset$Stroke)
RM_dataset$Hypertensive<-as.numeric(RM_dataset$Hypertensive)
RM_dataset$Diabetes<-as.numeric(RM_dataset$Diabetes)




## SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET ####### 
set.seed(123)
split = sample.split(RM_dataset$TenYr.chd, SplitRatio = 0.75)
RM_training_set = subset(RM_dataset, split == TRUE)
RM_test_set = subset(RM_dataset, split == FALSE)

# Feature Scaling
RM_training_set[-15] = scale(RM_training_set[-15])
RM_test_set[-15] = scale(RM_test_set[-15])

## FITTING RANDOM FOREST CLASSIFICATION TO THE TRAINING SET ####### 
set.seed(123)
classifier = randomForest(x = RM_training_set[-15],
                          y = RM_training_set$TenYr.chd,
                          ntree = 500)

## PREDICTING THE TEST SET RESULTS ####### 
y_pred = predict(classifier, newdata = RM_test_set[-15])

## MAKING THE CONFUSION MATRIX ####### 
cm = table(RM_test_set[, 15], y_pred)
cm
confusionMatrix(cm, positive='1')



#######################################
##
## Now let's fit the data with a random forest...
##
#######################################

## ROC for random forest
roc(y_pred, as.numeric(RM_test_set$TenYr.chd), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#4daf4a", lwd=4, print.auc=TRUE)

# AUC: 72.50%

#######################################
##
## Now layer logistic regression and random forest ROC graphs..
##
#######################################
roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

plot.roc(y_pred, as.numeric(RM_test_set$TenYr.chd), percent=TRUE, col="#4daf4a", lwd=4, print.auc=TRUE, add=TRUE, print.auc.y=40)
legend("bottomright", legend=c("Logisitic Regression", "Random Forest"), col=c("#377eb8", "#4daf4a"), lwd=4)

# Logist Regression AUC: 45.30%
# Random Forest AUC: 72.20%


# APPLYING K-FOLD CROSS VALIDATION FOR ACCURACY ----------------------------------------
folds = createFolds(RM_training_set$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  RM_training_fold = train_set_trees[-x, ]
  RM_test_fold = train_set_trees[x, ]
  classifier = randomForest(x = RM_training_fold[-15],
                            y = RM_training_fold$TenYr.chd,
                            ntree = 500)
  y_pred = predict(classifier, newdata = RM_test_fold[-15], type = 'class')
  cm = table(RM_test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
cv
accuracy = mean(as.numeric(cv)) # caldualte the accuracy on ten splits
accuracy # 85%













# APPLYING K-FOLD CROSS VALIDATION FOR CONFUSION MATRIX ----------------------------------------
folds = createFolds(RM_training_set$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  RM_training_fold = train_set_trees[-x, ]
  RM_test_fold = train_set_trees[x, ]
  classifier = randomForest(x = RM_training_fold[-15],
                            y = RM_training_fold$TenYr.chd,
                            ntree = 500)
  y_pred = predict(classifier, newdata = RM_test_fold[-15], type = 'class')
  cm = table(RM_test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(cm)
})
# cm = table(test_fold[, 15], y_pred)
cm
confusionMatrix(cm, positive='1')















#######################################
##
## Now let's fit the data with a random forest...
##
#######################################

## ROC for random forest
roc(y_pred, as.numeric(XG_test_set$TenYr.chd), plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#4daf4a", lwd=4, print.auc=TRUE)

# AUC: 72.50%

#######################################
##
## Now layer logistic regression and random forest ROC graphs..
##
#######################################
roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

plot.roc(y_pred, as.numeric(RM_test_set$TenYr.chd), percent=TRUE, col="#4daf4a", lwd=4, print.auc=TRUE, add=TRUE, print.auc.y=40)
legend("bottomright", legend=c("Logisitic Regression", "Random Forest"), col=c("#377eb8", "#4daf4a"), lwd=4)

# Logist Regression AUC: 45.30%
# Random Forest AUC: 72.20%




# SMOTE TEMPLATE -------------------------------------------------------------------
smote_dataset <- chd

split <- sample.split(smote_dataset$TenYr.chd, SplitRatio = 0.80)
train <- subset(smote_dataset, split == TRUE)
test <- subset(smote_dataset, split == FALSE)

## Let's check the count of unique value in the target variable
as.data.frame(table(smote_dataset$TenYr.chd))

## Loading DMwr to balance the unbalanced class
library(DMwR)

## Smote : Synthetic Minority Oversampling Technique To Handle Class Imbalancy In Binary Classification
balanced.data <- SMOTE(TenYr.chd ~., train, perc.over = 4800, k = 5, perc.under = 1000)

as.data.frame(table(balanced.data$TenYr.chd))

## Logistic Regression with SMOTE -------------------------------------------------------------------
library(caret)  
model <- glm (TenYr.chd ~., data = balanced.data, family = binomial)
summary(model)

## Predict the Values
predict <- predict(model, test, type = 'response')

## Create Confusion Matrix
table(test$TenYr.chd, predict > 0.5)

#ROCR Curve
library(ROCR)
ROCRpred <- prediction(predict, test$TenYr.chd)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))



############## 11. RANDOM FOREST WITH SMOTE -------------------------------------------------------------------
library(randomForest)  
library(e1071)  

rf = randomForest(TenYr.chd~.,  
                  ntree = 100,
                  data = balanced.data)
plot(rf)
varImp(rf)

## Important variables according to the model
varImpPlot(rf,  
           sort = T,
           n.var=25,
           main="Variable Importance")

predicted.response <- predict(rf, test)

confusionMatrix(data=predicted.response, reference=test$TenYr.chd, positive = '1')

## AUC / ROC  ####### 
set.seed(123)
test[,15] <- sort(test[,15] )
plot(x=test[,15]  , y=predicted.response)

## fit a logistic regression to the data...
glm.fit=glm(predicted.response ~ as.numeric(test[,15]), family=binomial)
lines(test[, 15], glm.fit$fitted.values)

roc(predicted.response, glm.fit$fitted.values, plot=TRUE)
par(pty = "s") 
roc(predicted.response, glm.fit$fitted.values, plot=TRUE)

roc(predicted.response, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE)

roc(predicted.response, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage")

roc(predicted.response, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4)

## We can calculate the area under the curve...
roc(predicted.response, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

## ...and the partial area under the curve.
# roc(predicted.response, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90), auc.polygon = TRUE, auc.polygon.col = "#377eb822")


# AUC: 52.60%
















########## 12. XGBoost Classifier ------------------------------------
# Extreme Gradient Boosting 
# (https://rpubs.com/awanindra01/xgboost)
# install.packages('BiocManager')
# load required libraries
if (!require("pacman")) install.packages("pacman")
pacman::p_load(pscl, ggplot2, ROCR,xgboost, magrittr, Matrix, readr, stringr,caret, car)

# dependent variable
chd$TenYr.chd = as.numeric(as.character(chd$TenYr.chd))

#training and Validation dataset
set.seed(123)
smp_size = floor(0.7 * nrow(chd))
train_ind = sample(seq_len(nrow(chd)), size = smp_size)

train = chd[train_ind, ]
val = chd[-train_ind, ]

#prepare training data
set.seed(123)
trainm = sparse.model.matrix(TenYr.chd ~., data = train)
train_label = train[,"TenYr.chd"]
train_matrix = xgb.DMatrix(data = as.matrix(trainm), label = train_label)

#prepare validation data
valm = sparse.model.matrix(TenYr.chd ~., data= val)
val_label = val[,"TenYr.chd"]
val_matrix = xgb.DMatrix(data = as.matrix(valm), label = val_label)

#parameters
xgb_params = list(objective   = "binary:logistic",
                  eval_metric = "error",
                  max_depth   = 3,
                  eta         = 0.01,
                  gammma      = 1,
                  colsample_bytree = 0.5,
                  min_child_weight = 1)

#model
bst_model = xgb.train(params = xgb_params, data = train_matrix,
                      nrounds = 1000)


#feature importance
imp = xgb.importance(colnames(train_matrix), model = bst_model)
xgb.plot.importance(imp)

#prediction & confusion matrix
p = predict(bst_model, newdata = val_matrix)
val$predicted = ifelse(p > 0.15,1,0)
confusionMatrix(table(val$predicted, val$TenYr.chd), positive = '1')

# Evaluation Curve
pred=prediction(p,val$TenYr.chd)
eval= performance(pred,"acc",)
plot(eval)


#Roc
roc=performance(pred,"tpr","fpr")
plot(roc,main="ROC curve")
abline(a=0,b=1)


#AUC
auc(val$predicted, val$TenYr.chd, method = "trapezoid")




############## 13. SVM ---------------------------------------------------
# Despite its popularity, SVM has a serious drawback, that is sensitivity to outliers in training samples. The penalty on misclassification is defined by a convex loss called the hinge loss, and the unboundedness of the convex loss causes the sensitivity to outliers.

## make sure to deal with outliers in the pre-processing stage

SVM_dataset <- chd

## ENCODING THE TARGET FEATURE AS FACTOR ####### 
SVM_dataset$TenYr.chd = factor(SVM_dataset$TenYr.chd, levels = c(0, 1))

## FEATURES FOR NEEDS TO BE NUMERIC FOR FEATURE SCALING ####### 
# XG_dataset$TenYr.chd = as.numeric(XG_dataset$TenYr.chd, levels = c(0, 1))
SVM_dataset$Age<-as.numeric(SVM_dataset$Age)
SVM_dataset$Gender<-as.numeric(SVM_dataset$Gender)
SVM_dataset$Smoker<-as.numeric(SVM_dataset$Smoker)
SVM_dataset$BPMeds<-as.numeric(SVM_dataset$BPMeds)
SVM_dataset$Stroke<-as.numeric(SVM_dataset$Stroke)
SVM_dataset$Hypertensive<-as.numeric(SVM_dataset$Hypertensive)
SVM_dataset$Diabetes<-as.numeric(SVM_dataset$Diabetes)


## SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET ####### 
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(SVM_dataset$TenYr.chd, SplitRatio = 0.75)
training_set = subset(SVM_dataset, split == TRUE)
test_set = subset(SVM_dataset, split == FALSE)

## Feature Scaling #####
training_set[-15] = scale(training_set[-15])
test_set[-15] = scale(test_set[-15])

## FITTING SVM CLASSIFICATION TO THE TRAINING SET ####### 
library(e1071)
classifier = svm(formula = TenYr.chd ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

## PREDICTING THE TEST SET RESULTS ####### 
y_pred = predict(classifier, newdata = test_set[-15])


## MAKING THE CONFUSION MATRIX ####### 
cm = table(test_set[, 15], y_pred)
cm
confusionMatrix(cm, positive='1')

## APPLYING K-FOLD CROSS VALIDATION FOR ACCURACY ----------------------------------------
folds = createFolds(training_set$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = TenYr.chd ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'linear')
  y_pred = predict(classifier, newdata = test_fold[-15])
  cm = table(test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
  return(confusionMatrix(cm, positive='1'))
})
cv
accuracy = mean(as.numeric(cv)) # calcualte the accuracy on ten splits
accuracy # 85%













## APPLYING K-FOLD CROSS VALIDATION FOR CONFUSION MATRIX ----------------------------------------
folds = createFolds(training_set$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = TenYr.chd ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'linear')
  y_pred = predict(classifier, newdata = test_fold[-15])
  cm = table(test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  # return(accuracy)
  return(confusionMatrix(cm, positive='1'))
})
cm
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')















## ROC method 1 ####### 
fit_glm <- glm(TenYr.chd ~ ., training_set, family=binomial(link="logit"))

glm_link_scores <- predict(fit_glm, test_set, type="link")

glm_response_scores <- predict(fit_glm, test_set, type="response")

score_data <- data.frame(link=glm_link_scores, 
                         response=glm_response_scores,
                         TenYr.chd= test_set$TenYr.chd,
                         stringsAsFactors=FALSE)

score_data %>% 
  ggplot(aes(x=link, y=response, col=TenYr.chd)) + 
  scale_color_manual(values=c("black", "red")) + 
  geom_point() + 
  geom_rug() + 
  ggtitle("ROC for KERNAL TEST")


# (https://www.youtube.com/watch?v=qcvAqAH60Yw)
plot(roc(test_set$TenYr.chd, glm_response_scores, direction="<"),
     col="blue", lwd=3,  legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", main="The turtle finds its way", print.auc=TRUE)

roc.df <- data_frame(
  tpp=roc.info$sensitivities*100,
  fpp=(1 - roc.info$specificities)*100,
  thresholds=roc.info$thresholds)
summary(roc.df)
head(roc.df)
tail(roc.df)

# ## ROC  ####### 
# response10 <- predictor10<- c()
# response10 <- c(response10, test_set$TenYr.chd)
# predictor10<- c(predictor10, y_pred)
# 
# roc6 <- plot.roc(response10, predictor10,  main="ROC for SVM",
#                  ylab="True Positive Rate",xlab="False Positive Rate", percent=TRUE, col="magenta")
# 
# print(roc6)








############## 14. GRID SEARCH PARAMETER TUNING WITH SVM -------------------------------------------------------------------
## ENCODING CATEGORICAL DATA ####### 
# Encoding the target feature as factor
chd$TenYr.chd = factor(chd$TenYr.chd, levels = c(0, 1), labels = c(0, 1))

# Encoding categorical data
chd$Gender = factor(chd$Gender, levels = c(0, 1), labels = c(0, 1))
chd$Smoker = factor(chd$Smoker, levels = c(0, 1), labels = c(0, 1))
chd$BPMeds = factor(chd$BPMeds, levels = c(0, 1), labels = c(0, 1))
chd$Stroke = factor(chd$Stroke, levels = c(0, 1), labels = c(0, 1))
chd$Hypertensive = factor(chd$Hypertensive, levels = c(0, 1), labels = c(0, 1))
chd$Diabetes = factor(chd$Diabetes, levels = c(0, 1), labels = c(0, 1))



## FEATURES FOR NEEDS TO BE NUMERIC FOR FEATURE SCALING ####### 
chd$Age<-as.numeric(chd$Age)
chd$Gender<-as.numeric(chd$Gender)
chd$Smoker<-as.numeric(chd$Smoker)
chd$BPMeds<-as.numeric(chd$BPMeds)
chd$Stroke<-as.numeric(chd$Stroke)
chd$Hypertensive<-as.numeric(chd$Hypertensive)
chd$Diabetes<-as.numeric(chd$Diabetes)



## TRAINNING / TEST SET SPILT #####
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(chd$TenYr.chd, SplitRatio = 0.75)
training_set = subset(chd, split == TRUE)
test_set = subset(chd, split == FALSE)

# Feature Scaling
training_set[-15] = scale(training_set[-15])
test_set[-15] = scale(test_set[-15])
# str(training_set)


## APPLYING GRID SEARCH TO FIND THE BEST PARAMETERS #####
# install.packages('caret')
# library(caret)
set.seed(123)
classifier = train(form = TenYr.chd ~ ., data = training_set, method = 'svmRadial')
classifier
classifier$bestTune



## FITTING NEW TUNING PARAMETERS TO SVM TRAINING SET #####
# install.packages('e1071')
library(e1071)
classifier = svm(formula = TenYr.chd ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial',
                 sigma = 0.06723858,
                 C = 1)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-15])

# Making the Confusion Matrix
cm = table(test_set[, 15], y_pred)
cm
## Applying k-Fold Cross Validation for accuracy #####
# install.packages('caret')
library(caret)
folds = createFolds(training_set$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = TenYr.chd ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial',
                   sigma = 0.06723858,
                   C = 1)
  y_pred = predict(classifier, newdata = test_fold[-15])
  cm = table(test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
accuracy

## Applying k-Fold Cross Validation for confusion matrix #####
folds = createFolds(training_set$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = TenYr.chd ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial',
                   sigma = 0.06772541,
                   C = 1)
  y_pred = predict(classifier, newdata = test_fold[-15])
  cm = table(test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(cm)
})
cm
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')



## AUC / ROC  ####### 
set.seed(123)
test_set[, 15] <- sort(test_set[, 15])
plot(x=test_set[, 15] , y=y_pred)

## fit a logistic regression to the data...
glm.fit=glm(y_pred ~ test_set[,15], family=binomial)
lines(test_set[,15], glm.fit$fitted.values)

roc(y_pred, glm.fit$fitted.values, plot=TRUE)
par(pty = "s") 


## We can calculate the area under the curve...
roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

## ...and the partial area under the curve.
# roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90), auc.polygon = TRUE, auc.polygon.col = "#377eb822")


# AUC:  67.44%




############## 15. KERNAL SVM ---------------------------------------------------
kernal_dataset <- chd

## ENCODING THE TARGET FEATURE AS FACTOR #######
kernal_dataset$TenYr.chd = factor(kernal_dataset$TenYr.chd, levels = c(0, 1))

## FEATURES FOR NEEDS TO BE NUMERIC FOR FEATURE SCALING ####### 
# XG_dataset$TenYr.chd = as.numeric(XG_dataset$TenYr.chd, levels = c(0, 1))
kernal_dataset$Age<-as.numeric(kernal_dataset$Age)
kernal_dataset$Gender<-as.numeric(kernal_dataset$Gender)
kernal_dataset$Smoker<-as.numeric(kernal_dataset$Smoker)
kernal_dataset$BPMeds<-as.numeric(kernal_dataset$BPMeds)
kernal_dataset$Stroke<-as.numeric(kernal_dataset$Stroke)
kernal_dataset$Hypertensive<-as.numeric(kernal_dataset$Hypertensive)
kernal_dataset$Diabetes<-as.numeric(kernal_dataset$Diabetes)



## SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET ####### 
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(kernal_dataset$TenYr.chd, SplitRatio = 0.75)
training_set = subset(kernal_dataset, split == TRUE)
test_set = subset(kernal_dataset, split == FALSE)

# Feature Scaling
training_set[-15] = scale(training_set[-15])
test_set[-15] = scale(test_set[-15])

## FITTING KERNAL SVM CLASSIFICATION TO THE TRAINING SET ####### 
# install.packages('e1071')
library(e1071)
classifier = svm(formula = TenYr.chd ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

## PREDICTING THE TEST SET RESULTS ####### 
y_pred = predict(classifier, newdata = test_set[-15])

## MAKING THE CONFUSION MATRIX ####### 
cm = table(test_set[, 15], y_pred)
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')


## APPLYING K-FOLD CROSS VALIDATION FOR ACCURACY ----------------------------------------
folds = createFolds(training_set$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = TenYr.chd ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-15])
  cm = table(test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
  return(cm)
})
cv
accuracy = mean(as.numeric(cv)) # calcualte the accuracy on ten splits
accuracy # 85%







## APPLYING K-FOLD CROSS VALIDATION FOR CONFUSION MATRIX ----------------------------------------
folds = createFolds(training_set$TenYr.chd, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = TenYr.chd ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-15])
  cm = table(test_fold[, 15], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
  return(cm)
})
cm
names(dimnames(cm)) <- c("predicted", "observed")
confusionMatrix(cm, positive='1')





## AUC / ROC  ####### 
set.seed(123)
test_set[, 15] <- sort(test_set[, 15])
plot(x=test_set[, 15] , y=y_pred)

## fit a logistic regression to the data...
glm.fit=glm(y_pred ~ test_set[,15], family=binomial)
lines(test_set[,15], glm.fit$fitted.values)

roc(y_pred, glm.fit$fitted.values, plot=TRUE)
par(pty = "s") 
roc(y_pred, glm.fit$fitted.values, plot=TRUE)

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE)

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage")

roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4)

## We can calculate the area under the curve...
roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE)

## ...and the partial area under the curve.
# roc(y_pred, glm.fit$fitted.values, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(100, 90), auc.polygon = TRUE, auc.polygon.col = "#377eb822")


# AUC: 52.60%











