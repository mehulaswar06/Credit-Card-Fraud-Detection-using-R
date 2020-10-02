setwd("E://R Studio for Data Science/")
getwd()

#Importing Dataset from the respective folder.

credit_card<-read.csv("c:\\Users/MEHUL/Desktop/Projects/Data Science Intern Project/creditcard.csv")
#Glance at the structure of the dataset.
str(credit_card)
#Convert class to a factor variable
credit_card$Class<-factor(credit_card$Class,levels=c(0,1))


#Get the summary of the Data

summary(credit_card)

#Looking for the missing values
sum(is.na(credit_card))
#Output-There is no missing values in data

#-------------------------------------------#
#get the distribution of fraud and legitimate transactions in the dataset
table(credit_card$Class)
#Get the distribution of fraud and legit transaction in the dataset
prop.table(table(credit_card$Class))

#Pie Chart of credit card transactions
labels<-c("legit","fraud")
labels<-paste(labels,round(100*prop.table(table(credit_card$Class)),2))
labels<-paste0(labels,"%")
labels

pie(table(credit_card$Class),labels,col = c("orange","red"),
    main = "Pie Chart of Credit Card Transaction")
#------------------------------------------------------------------------#
#No Model Predictions

predictions<-rep.int(0,nrow(credit_card))
                     
predictions<-factor(predictions,levels=c(0,1))
predictions
                     
install.packages('caret')
install.packages('e1071')
library(caret)
confusionMatrix(data=predictions,reference=credit_card$Class)
                     
 #------------------------------------------------------------------#
#Creating smaller subset
library(dplyr)
                     
set.seed(1)
credit_card<-credit_card %>% sample_frac(0.1)
                     
table(credit_card$Class)
library(ggplot2)
                     
ggplot(data=credit_card,aes(x=V1 , y=V2,col=Class,)) +geom_point()+theme_bw()+scale_color_manual(values=c('dodgerblue2',"red"))
                 
#-------------------------------------------------------------------#


#Creating training and test sets for fraud detection model
#install.packages('caTools')
library(caTools)
set.seed(123)
data_sample=sample.split(credit_card$Class,SplitRatio=0.80)
                     
train_data=subset(credit_card,data_sample==TRUE)
                     
test_data=subset(credit_card,data_sample==FALSE)
                     
dim(train_data)
dim(test_data)
                     
 #-------------------------------------------------------------------------#
#Making the dataset balanced
#Random Over-Sampling(ROS)
table(train_data$Class)
n_legit<-22750
new_frac_legit<-0.50
new_n_total<-n_legit/new_frac_legit #=22750/0.50 
                     
#install.packages('ROSE')
library(ROSE)
                     
oversampling_result<-ovun.sample(Class ~.,
                                data=train_data,
                                method="over",
                                N=new_n_total,
                                seed=2019)
oversampled_credit<-oversampling_result$data
table(oversampled_credit$Class)
                     
ggplot(data=oversampled_credit,aes(x=V1,y=V2,col=Class))+
      geom_point(position=position_jitter(width=0.2))+
    theme_bw()+
    scale_color_manual(values=c('dodgerblue2','red'))
                     
#----------------------------------------------------------------------------------------#
                     
#Random under-Sampling(RUS)
                     
table(train_data$Class)
                     
n_fraud<-35
new_frac_fraud<-0.50
new_n_total<-n_fraud/new_frac_fraud  #=35/0.50
                     
#library(ROSE)
                     
undersampling_result<-ovun.sample(Class ~.,
                                  data=train_data,
                                  method = "under",
                                  N=new_n_total,
                                  seed=2019)
undersampled_credit<-undersampling_result$data
table(undersampled_credit$Class)
                     
 ggplot(data=undersampled_credit,aes(x=v1,y=v2,col=class))+
                       geom_point()+
                       these_bw()+
                       scale_color_manual(balues=c('dodgerblue2','red'))
 #----------------------------------------------------------------#
 #ROS and RUS 
 n_new<-nrow(train_data) #=22785
                     
fraction_fraud_new<-0.50
                     
sampling_result<-ovun.sample(Class ~.,
                            data=train_data,
                            method="both",
                            N=n_new,
                            p=fraction_fraud_new,
                             seed=2019)
 sampled_credit<-sampling_result$data
                     
table(sampled_credit$Class)
prop.table(table(sampled_credit$Class))
                     
ggplot(data=sampled_credit,aes(x=v1,y=v2,col=class))+
          geom_point(position=postion_jitter(width=0.2)) +
          theme_bw()+
          scale_color_manual(values=c('dodgerblue2','red'))
                     
 #--------------------------------------------------#
                     
install.packages("smotefamily")
                     
library(smotefamily)
                     
table(train_data$Class)
#Set the number and legitimate cases, and the desired percentage of liegitimate cases
 n0<-22750
 n1<-35
 r0<-0.6
                     
#Calculate the value for the dup_size parameter of SMOTE
ntimes<- ((1-r0)/r0)*(n0/n1) - 1
                     
smote_output=SMOTE(X=train_data[ ,-c(1,31)],
                  target=train_data$Class,
                  K=5,
                  dup_size=ntimes)
credit_smote<-smote_output$data
                     
colnames(credit_smote)[30]<-"Class"
prop.table(table(credit_smote$Class))
                     
#Class distribution for original dataset
ggplot(train_data,aes(x=V1,y=V2,color=Class))+
                       geom_point()+
                       scale_color_manual(values=c('dodgerblue2','red'))
                     
#Class Distribution for over-sampled dataset using SMOTE
                     
ggplot(credit_smote,aes(x=V1,y=V2,color=Class))+
            geom_point()+
            scale_color_manual(values=c('dodgerblue2','red'))
                     
#------------------------------------------------------------------#
#Decision Tree                   
#install.packages('rpart')
install.packages('rpart.plot')
library(rpart)
library(rpart.plot)
CART_model<-rpart(Class ~.,credit_smote)
rpart.plot(CART_model,extra=0,type=5,tweak=1.2)
                     
#predict fraud classes
predicted_val<-predict(CART_model,test_data,type="class")
#Build Confusion matrix
library(caret)
                     
confusionMatrix(predicted_val,test_data$Class)
#------------------------------------------------------------------------#
 #Prediction with original subset                   
predicted_val<-predict(CART_model,credit_card[-1],type='class')
confusionMatrix(predicted_val,credit_card$Class)
                     
#-------------------------------------------------------------------------
#Decision Tree without SMOTE
                     
CART_model<-rpart(Class ~. ,train_data[,-1])
                     
rpart.plot(CART_model,extra=0,type=5,tweak=1.2)
                     
#Predict fraud classes
predicted_val<-predict(CART_model,test_data[,-1],type='class')
                     
library(caret)
confusionMatrix(predicted_val,test_data$Class)
                     
#-------------------------------------------------------
#Prediction with original subset
predicted_val<-predict(CART_model,credit_card[,-1],type='class')
confusionMatrix(predicted_val,credit_card$Class)
                     
 #-----------------------------------------------#

   #By Mehul Aswar PRN-119A3006
                     