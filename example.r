## Scenario 1,MCAR, p=20, rho=0.5

library(MASS) # for mvrnorm function
library(pROC)
library(e1071) # for SVM
library(randomForest)
library(xgboost)
library(HDeconometrics) # for adalasso LR
library(caret) # for upSample, downSample
library(DMwR) # for SMOTE


source('CVgroup.R')
source('TS-adaLR.R')
source('calculate_performance.R')

## simulated MCAR data: 2000 samples, 20 covariates, 4 groups,Group 1 without missing, 8 missing patterns
#  covariate correlation coefficient 0.5

mydata<-read.csv(file = "data.csv",header = T, row.names = 1)

#covariates of each missing pattern
pp=20
mn=8
ppd=pp/4
ppp=seq((1+ppd),by=ppd,length=4) # from p1 to p5
mp<-list(c(1:ppp[4]),c(1:ppp[3]),c(1:ppp[2],(ppp[3]+1):ppp[4]),
         c(1:ppp[1],(ppp[2]+1):ppp[4]),c(1:ppp[2]),c(1:ppp[1],(ppp[2]+1):ppp[3]),
         c(1:ppp[1],(ppp[3]+1):ppp[4]),c(1:ppp[1]))

na2<-list()
for (i in 1:10) {
  na2[[i]]<-list()
}

main<-function(sed)
{
  set.seed(sed*12)
  #randomly select 10% data as test set, rest as training set
  index <-sample(nrow(mydata),nrow(mydata)*0.7)
  train<-mydata[index,]
  test<-mydata[-index,]
  
  # training data for each missing pattern
  train_list <- list()
  test_list<-list()
  # complete cases for each missing pattern
  for (i in 1:mn) {
    train_list[[i]]<-train[complete.cases(train[,mp[[i]]]),mp[[i]]]
  }
  for (i in 1:mn) {
    test_list[[i]]<-test[complete.cases(test[,mp[[i]]]),mp[[i]]]
  }
  ####  proposed: TS-adaLR  ####
  starttime.ts<-proc.time()
  train.ts<-train_list
  test.ts<-test_list
  
  resl.ts<-tsadalr(train.ts,test.ts,8,1)
  per.ts<-resl.ts[[2]]
  auc.ts<-resl.ts[[3]]
  r.ts<-calcper(per.ts,auc.ts)
  moin.ts<-resl.ts[[4]]
  
  endtime.ts<-proc.time()
  runtime.ts<-endtime.ts-starttime.ts
  ####  proposed: ROS-TS-adaLR  #### randomly oversampling
  starttime.ros<-proc.time()
  train.ros<-train_list
  test.ros<-test_list
  
  for (i in 1:mn) {
    train.ros[[i]]$y<-as.factor(train.ros[[i]]$y)
  }
  for (i in 1:mn) {
    train.ros[[i]]<-upSample(train.ros[[i]][,-1],y=train.ros[[i]]$y,yname = "y",list = F)
  } 
  for (i in 1:mn) {
    train.ros[[i]]$y<-as.numeric(paste(train.ros[[i]]$y))
  }
  for (i in 1:mn) {
    train.ros[[i]]<-train.ros[[i]][,c(ncol(train.ros[[i]]),1:(ncol(train.ros[[i]])-1))]
  }
  
  resl.ros<-tsadalr(train.ros,test.ros,8,1)
  per.ros<-resl.ros[[2]]
  auc.ros<-resl.ros[[3]]
  r.ros<-calcper(per.ros,auc.ros)
  moin.ros<-resl.ros[[4]]
  
  endtime.ros<-proc.time()
  runtime.ros<-endtime.ros-starttime.ros
  ####  proposed: SOMTE-TS-adaLR  ####
  starttime.smo<-proc.time()
  train.smo<-train_list
  test.smo<-test_list
  
  oversampled_data<-list()
  for (i in 1:mn) {
    current_data<-train.smo[[i]]
    current_data$y<-as.factor(current_data$y)
    oversampled_data[[i]] <- SMOTE(y ~., current_data, perc.over = 250, perc.under = 100)
  }
  train.smo<-oversampled_data
  for (i in 1:mn) {
    train.smo[[i]]$y<-as.numeric(paste(train.smo[[i]]$y))
  }
  resl.smo<-tsadalr(train.smo,test.smo,8,1)
  per.smo<-resl.smo[[2]]
  auc.smo<-resl.smo[[3]]
  r.smo<-calcper(per.smo,auc.smo)
  moin.smo<-resl.smo[[4]]
  
  endtime.smo<-proc.time()
  runtime.smo<-endtime.smo-starttime.smo
  
  
  r.all<-t(cbind(r.ts,r.ros,r.smo))
  moin.all<-c(moin.ts,moin.ros,moin.smo)
  runtime.all<-cbind(runtime.ts,runtime.ros,runtime.smo)[1,]
  result<-rbind(r.all,as.matrix(moin.all),as.matrix(runtime.all))
  return(result)
}

# Simulate 100 times, save results
r<-matrix(NA,nrow = 24,ncol = 100)
for (re in c(1:100)) {
  r[,re]<-as.numeric(main(re))
  print(re)
}

apply(r,1,mean)
write.csv(r,file = "result.csv")

# Save the performance (ACC, F, Recall, Specificity, and AUC) of 3 models 
rp<-r[-c(6,12,18:24),]
rpms<-cbind(apply(rp, 1, mean,na.rm=T),apply(rp, 1, sd,na.rm=T))

rpm<-round(rpms[,1],3)
rps<-round(rpms[,2],3)
rps<-paste0("(", sprintf("%.3f", rps), ")")
rps<-paste0("\t", rps, "\t")
r.m<-matrix(rpm, nrow = 3, byrow = TRUE)
r.s<-matrix(rps, nrow = 3, byrow = TRUE)


# Create empty matrix to store merged results
merged_matrix <- matrix(nrow = 6, ncol = 5)
# Alternately merge r.m and r.s
for (i in 1:3) {
  merged_matrix[2*i - 1, ] <- r.m[i, ]
  merged_matrix[2*i, ] <- r.s[i, ]
}
write.csv(merged_matrix,file = "final_result.csv")
