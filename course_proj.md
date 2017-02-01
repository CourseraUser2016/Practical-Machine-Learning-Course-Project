# Practical Machine Learning


  
## Background 
  Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here (see the section on the Weight Lifting Exercise Dataset).
  
## Executive Summary
  
To predict the manner in which users of activity trackers did their exercise, we have compared the prediction accuracy of both random forests and decision trees. We have found that by dropping variables that had more than 60% NA's or low variance, that random forests were able to predict the "classe" variable with 99.18% accuracy on the test data and decision trees were able to predict with 74.23% accuracy.

  
## Setting the environment  
  

```r
#this function will install packages if they don't exist on a system
packages<-function(x){
  x<-as.character(match.call()[[2]])
  if (!require(x,character.only=TRUE)){
    install.packages(pkgs=x,
                     repos="http://cran.r-project.org",
                     dependencies = TRUE)
    require(x,character.only=TRUE)
  }
}

packages(caret)
packages(randomForest)
packages(rpart)
packages(rpart.plot)
packages(RColorBrewer)
packages(rattle)

#set working directory

if(!dir.exists("./data")){
    dir.create("./data")
}

setwd("./data")
#download files
if(!file.exists("pml-training.csv")){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
}

if(!file.exists("pml-testing.csv")){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","pml-testing.csv")
}

#Read training data and replace missing/error values
train_data <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""), header=TRUE)

#Read estnd replace missing/error values
test_data <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""), header=TRUE)

#Take a look at the data
str(train_data)
```

```
'data.frame':	19622 obs. of  160 variables:
 $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
 $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
 $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
 $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
 $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
 $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
 $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
 $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
 $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
 $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
 $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
 $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
 $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
 $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
 $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
 $ skewness_roll_belt.1    : num  NA NA NA NA NA NA NA NA NA NA ...
 $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
 $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
 $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
 $ max_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
 $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
 $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
 $ min_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
 $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
 $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
 $ amplitude_yaw_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
 $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
 $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
 $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
 $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
 $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
 $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
 $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
 $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
 $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
 $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
 $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
 $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
 $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
 $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
 $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
 $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
 $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
 $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
 $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
 $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
 $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
 $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
 $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
 $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
 $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
 $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
 $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
 $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
 $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
 $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
 $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
 $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
 $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
 $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
 $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
 $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
 $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
 $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
 $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
 $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
 $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
 $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
 $ kurtosis_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
 $ kurtosis_picth_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
 $ kurtosis_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
 $ skewness_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
 $ skewness_pitch_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
 $ skewness_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
 $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
 $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
 $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
 $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
 $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
 $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
 $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
 $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
 $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
 $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
 $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
 $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
 $ kurtosis_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
 $ kurtosis_picth_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
 $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
 $ skewness_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
 $ skewness_pitch_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
 $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
 $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
 $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
 $ max_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
 $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
 $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
 $ min_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
 $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
  [list output truncated]
```
  
  We can see that the first seven variables of the data are made up of metadata not relevant to the model and there are near 0 variance variables. We will remove this in a later step.
  
## Partitioning data for Cross-Validation
  

```r
#partition data for cross validation
inTrain <- createDataPartition(y=train_data$classe, p = 0.60, list=FALSE)
training <- train_data[inTrain,]
testing <- train_data[-inTrain,]

dim(training); dim(testing)
```

```
[1] 11776   160
```

```
[1] 7846  160
```
  
## Data Processing
  

```r
training <- training[,-c(1:7)]

nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[, nzv$nzv==FALSE]

training_clean <- training
for(i in 1:length(training)) {
  if( sum( is.na( training[, i] ) ) /nrow(training) >= .6) {
    for(j in 1:length(training_clean)) {
      if( length( grep(names(training[i]), names(training_clean)[j]) ) == 1)  {
        training_clean <- training_clean[ , -j]
      }   
    } 
  }
}

# Set the new cleaned up dataset back to the old dataset name
training <- training_clean

# Get the column names in the training dataset
columns <- colnames(training)

# Subset the test data on the variables that are in the training data set and remove the class variable
test_data <- test_data[colnames(training[, -53])]
dim(test_data)
```

```
[1] 20 52
```
  
##Cross-Validation: Prediction with Random Forest
  A random forest is run on the training set and the results of the test data are evaluated 

```r
set.seed(54321)
modFit <- randomForest(classe ~ ., data=training)
prediction <- predict(modFit, testing)
cm <- confusionMatrix(prediction, testing$classe)
print(cm)
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2232    5    0    0    0
         B    0 1509   13    0    0
         C    0    4 1353   23    2
         D    0    0    2 1262    4
         E    0    0    0    1 1436

Overall Statistics
                                         
               Accuracy : 0.9931         
                 95% CI : (0.991, 0.9948)
    No Information Rate : 0.2845         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.9913         
 Mcnemar's Test P-Value : NA             

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   0.9941   0.9890   0.9813   0.9958
Specificity            0.9991   0.9979   0.9955   0.9991   0.9998
Pos Pred Value         0.9978   0.9915   0.9790   0.9953   0.9993
Neg Pred Value         1.0000   0.9986   0.9977   0.9964   0.9991
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2845   0.1923   0.1724   0.1608   0.1830
Detection Prevalence   0.2851   0.1940   0.1761   0.1616   0.1832
Balanced Accuracy      0.9996   0.9960   0.9923   0.9902   0.9978
```


```r
overall.accuracy <- round(cm$overall['Accuracy'] * 100, 2)
sam.err <- round(1 - cm$overall['Accuracy'],2)
sam.err
```

```
Accuracy 
    0.01 
```
  
We see that the model is 99.18% accurate on a subset of the *training* data with an expected out of sample error rate of approximately 0.01%.  

```r
plot(modFit)
```

![](course_proj_files/figure-html/unnamed-chunk-2-1.png)<!-- -->
  
As can be seen, the random forest is converging to a solution with an error rate of less than 0.025 for all 5 classe.
  
## Cross-Validation: Prediction with a Decision Tree

```r
set.seed(54321)
modFit2 <- rpart(classe ~ ., data=training, method="class")
prediction2 <- predict(modFit2, testing, type="class")
cm2 <- confusionMatrix(prediction2, testing$classe)
print(cm2)
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1994  237   25   72   17
         B   64  893  126  114  122
         C   59  217 1098  197  179
         D   77  110   75  819   94
         E   38   61   44   84 1030

Overall Statistics
                                          
               Accuracy : 0.7436          
                 95% CI : (0.7337, 0.7532)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.6752          
 Mcnemar's Test P-Value : < 2.2e-16       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.8934   0.5883   0.8026   0.6369   0.7143
Specificity            0.9375   0.9327   0.8994   0.9457   0.9646
Pos Pred Value         0.8503   0.6770   0.6274   0.6970   0.8194
Neg Pred Value         0.9567   0.9042   0.9557   0.9300   0.9375
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2541   0.1138   0.1399   0.1044   0.1313
Detection Prevalence   0.2989   0.1681   0.2230   0.1498   0.1602
Balanced Accuracy      0.9154   0.7605   0.8510   0.7913   0.8394
```

```r
overall.accuracy2 <- round(cm2$overall['Accuracy'] * 100, 2)
sam.err2 <- round(1 - cm2$overall['Accuracy'],2)
sam.err2
```

```
Accuracy 
    0.26 
```
  
  The model is 74.23% accurate on the testing data partitioned from the training data. The expected out of sample error is roughly 0.26%.

  

```r
fancyRpartPlot(modFit2)
```

![](course_proj_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

## Prediction on the data 
  The Random Forest model gave an accuracy of 99.18%, which is much higher than the 74.23% accuracy from the Decision Tree. So we will use the Random Forest model to make the predictions on the test data to predict the way 20 participates performed the exercise.  

```r
final_prediction <- predict(modFit, test_data, type="class")
print(final_prediction)
```

```
 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
 B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
Levels: A B C D E
```
  
## Conclusion
  
  While Random Forests are much more computationally intensive, it can't be denied that they are much more reliable in their ability to predict a wide variety of variables. This is another example of why random forests are chosen over other machine learning algorithms. We have proven successfully that the behaviour of users of activity trackers can be predicted with an accuracy of 99%. 
