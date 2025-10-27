#### TS-adaLR- proposed method ####
tsadalr<-function(train,test,mn,sedts){
  tau = 1
  train.x <- list()
  train.y <- list()
  fglm.la <- list() # for lasso LR
  first.step.coef <-list()
  penal <-list() 
  cvfit <-list() 
  ll <- c()
  fglm <- list()
  prob.glm <-list()
  
  for (i in 1:mn) {
    train.x[[i]]<-train[[i]][,-1]
    train.y[[i]]<-train[[i]]$y
  }
  # adalasso LR & prediction
  for (i in 1:mn) {
    fglm.la[[i]]=ic.glmnet(as.matrix(train.x[[i]]),train.y[[i]],crit = "aic")
    first.step.coef[[i]]=coef(fglm.la[[i]])[-1]
    penal[[i]]=abs(first.step.coef[[i]])^(-tau)
    if(all(penal[[i]]==Inf)){
      penal[[i]][1]<-10^8
    }
    cvfit[[i]] = cv.glmnet(as.matrix(train.x[[i]]),train.y[[i]], alpha = 1)
    ll[i] = cvfit[[i]]$lambda.min
    fglm[[i]] = glmnet(as.matrix(train.x[[i]]),train.y[[i]], family = "binomial", 
                       alpha = 1, lambda = ll[i], penalty.factor = penal[[i]])
    # for 1st missing pattern (complete dataset), model group includes all candidate models
    prob.glm[[i]] = predict(fglm[[i]], newx = as.matrix(test[[i]][,-1]),type = "response")
  } 
  #Calculate AUC, ACC, precision (PRE), recall (REC), specificity (SPC) 
  roc.glm <-list()
  auc.glm <-list()
  per.glm <-list()
  thre.glm<-list()
  for (i in 1:mn) {
    roc.glm[[i]] <-roc(test[[i]]$y,as.numeric(prob.glm[[i]]),levels = c(0,1),direction='<')
    auc.glm[[i]] <-auc(roc.glm[[i]],partial.auc.focus=c("specificity",  "sensitivity"))
    per.glm[[i]]<-pROC::coords(roc.glm[[i]],x = "best",best.method ="youden", 
                               ret=c( "accuracy","precision", "recall","specificity"))
    thre.glm[[i]]<-pROC::coords(roc.glm[[i]],x = "best",best.method ="youden", ret=c( "threshold"))
  }
  ## For 2 thresholds: select the one with higher precision; set to 0.5 if inf
  for (i in 1:mn) {
    thre.glm[[i]]<-ifelse(nrow(thre.glm[[i]])==1,thre.glm[[i]],
                          ifelse(is.infinite(thre.glm[[i]][nrow(thre.glm[[i]]),])==T,
                                 0.5,thre.glm[[i]][nrow(thre.glm[[i]]),]))
  }
  ##For 2 per.glm: max follows rule (2nd by default); select 1st if per.glm=NaN; select 2nd (higher precision) if per.glm is numeric
  for (i in 1:mn) {
    per.glm[[i]]<-per.glm[[i]][!is.nan(apply(per.glm[[i]], 1, sum)),]
  }
  for (i in 1:mn) {
    if(nrow(per.glm[[i]])!=1){
      per.glm[[i]]<-per.glm[[i]][nrow(per.glm[[i]]),]
    }
  }
  ## convert probability to class prediction 
  pred.glm<-list()
  for (i in 1:mn) {
    pred.glm[[i]]<-ifelse(prob.glm[[i]]>thre.glm[[i]],1,0)
  }
  # Randomly split train-set complete cases into 4 folds
  set.seed(sedts)
  cvlist.glm <- CVgroup(k = 4,datasize = nrow(train[[1]]))
  #cv
  fglm.la.cv<-na2
  first.step.coef.cv<-na2
  penal.cv<-na2
  cvfit.cv<-na2
  ll.cv<-na2
  prob.glmcv <- matrix(NA, nrow=nrow(train[[1]]), ncol=mn)
  fglm.cv<-na2
  tau=1
  train.x.cv<-train[[1]][,-1]
  train.y.cv<-train[[1]]$y
  options(warn = -1)
  # cross-validation
  for (i in 1:mn) {
    for (k in 1:4){
      fglm.la.cv[[i]][[k]]=ic.glmnet(as.matrix(train[[1]][-cvlist.glm[[k]],mp[[i]]][,-1]),
                                     train.y.cv[-cvlist.glm[[k]]],crit = "aic")
      first.step.coef.cv[[i]][[k]]=coef(fglm.la.cv[[i]][[k]])[-1]
      penal.cv[[i]][[k]]=abs(first.step.coef.cv[[i]][[k]])^(-tau)
      if(all(penal.cv[[i]][[k]]==Inf)){
        penal.cv[[i]][[k]][1]<-10^8
      }
      cvfit.cv[[i]][[k]] = cv.glmnet(as.matrix(train[[1]][-cvlist.glm[[k]],mp[[i]]][,-1]),
                                     train.y.cv[-cvlist.glm[[k]]], alpha = 1)
      ll.cv[[i]][[k]] = cvfit.cv[[i]][[k]]$lambda.min
      fglm.cv[[i]][[k]] = glmnet(as.matrix(train[[1]][-cvlist.glm[[k]],mp[[i]]][,-1]),
                                 train.y.cv[-cvlist.glm[[k]]], family = "binomial", 
                                 alpha = 1, lambda = ll.cv[[i]][[k]], penalty.factor = penal.cv[[i]][[k]])
      prob.glmcv[cvlist.glm[[k]],i] = predict(fglm.cv[[i]][[k]], 
                                              newx = as.matrix(train[[1]][cvlist.glm[[k]],mp[[i]]][,-1]),
                                              type = "response")
    }
  }
  #Calculate threshold
  roc.glmcv <-na2
  thre.glmcv <-na2
  per.glmcv<-na2
  for (i in 1:mn) {
    for (k in 1:4){
      roc.glmcv[[i]][[k]] <-roc(train[[1]][cvlist.glm[[k]],]$y,prob.glmcv[cvlist.glm[[k]],i],levels = c(0,1),direction='<')
      thre.glmcv[[i]][[k]]<-coords(roc.glmcv[[i]][[k]],x = "best",best.method ="youden", 
                                   ret=c( "threshold"))
      per.glmcv[[i]][[k]]<-coords(roc.glmcv[[i]][[k]],x = "best",best.method ="youden", 
                                  ret=c( "accuracy","precision", "recall","specificity","threshold"))
    }
  }
  ## For 2 thresholds: select the one with higher precision
  for (i in 1:mn) {
    for (k in 1:4){
      thre.glmcv[[i]][[k]]<-ifelse(nrow(thre.glmcv[[i]][[k]])==1,thre.glmcv[[i]][[k]],
                                   ifelse(is.infinite(thre.glmcv[[i]][[k]][nrow(thre.glmcv[[i]][[k]]),])==T,
                                          0.5,thre.glmcv[[i]][[k]][nrow(thre.glmcv[[i]][[k]]),]))
    }
  }
  for (i in 1:mn) {
    for (k in 1:4){
      per.glmcv[[i]][[k]]<-if(class(per.glmcv[[i]][[k]])=="numeric"){per.glmcv[[i]][[k]]}
      else  {per.glmcv[[i]][[k]][,ncol(per.glmcv[[i]][[k]])]}
    }
  }
  # convert probability to class prediction 
  pred.glmcv<-prob.glmcv
  for (i in 1:mn) {
    for (k in 1:4){
      pred.glmcv[cvlist.glm[[k]],i]<-ifelse(prob.glmcv[cvlist.glm[[k]],i]> thre.glmcv[[i]][[k]],1,0)
    }
  }
  # link function:linear; LR, SVM, RF, Xgboost; 
  # 4-fold cross-validation
  # y predictions from each candidate model via cross-validation; each column contains predictions of a different candidate model
  # True y values
  y.train<-train[[1]]$y
  # Use predictions as covariates, true y as dependent variable; combine data and split into test set & training set at 3:7 ratio
  x.link<-as.data.frame(cbind(y.train,pred.glmcv))
  ind.link<-sample(nrow(x.link),0.3*nrow(x.link))
  train.link<-as.data.frame(x.link[-ind.link,])
  test.link<-as.data.frame(x.link[ind.link,])
  
  # for 1st missing pattern
  prob.glm.k1<-list()
  for (i in 1:mn) {
    prob.glm.k1[[i]] = predict(fglm[[i]], newx = as.matrix(test[[1]][,mp[[i]]][,-1]),
                               type = "response")
  }
  #Calculate AUC, ACC, precision (PRE), recall (REC), specificity (SPC)
  roc.glm.k1 <-list()
  thre.glm.k1<-list()
  for (i in 1:mn) {
    roc.glm.k1[[i]] <-roc(test[[1]]$y,as.numeric(prob.glm.k1[[i]]),
                          levels = c(0,1),direction='<')
    thre.glm.k1[[i]]<-coords(roc.glm.k1[[i]],x = "best",best.method ="youden", 
                             ret=c( "threshold"))
  }
  ## For 2 thresholds: select the one with higher precision; set to 0.5 if inf
  for (i in 1:mn) {
    thre.glm.k1[[i]]<-ifelse(nrow(thre.glm.k1[[i]])==1,thre.glm.k1[[i]],
                             ifelse(is.infinite(thre.glm.k1[[i]][nrow(thre.glm.k1[[i]]),])==T,
                                    0.5,thre.glm.k1[[i]][nrow(thre.glm.k1[[i]]),]))
  }
  ##convert probability to class prediction 
  pred.glm.k1<-list()
  for (i in 1:mn) {
    pred.glm.k1[[i]]<-ifelse(prob.glm.k1[[i]]>thre.glm.k1[[i]],1,0)
  }
  
  wtest<-as.data.frame(cbind(test[[1]]$y,do.call(cbind,pred.glm.k1)))
  wtest.x<-wtest[,-1]
  if(ncol(wtest.x)==8){
    names(wtest.x)<-c("V2","V3","V4","V5","V6","V7","V8","V9")
  }else{
    if(ncol(wtest.x)==4){
      names(wtest.x)<-c("V2","V3","V4","V5")
    }else{
    names(wtest.x)<-c("V2","V3","V4","V5","V6","V7","V8","V9","V10","V11")
    }
  }
  ######## link function linear regression  #####
  m1.link<-lm(y.train~.,data = train.link)
  pre1.link<-predict(m1.link,test.link)
  # Calculate auc
  m1.roc<-roc(test.link[,1],pre1.link,levels = c(0,1),direction='<')
  m1.auc<-auc(m1.roc,partial.auc.focus=c("specificity","sensitivity"))
  # prediction
  wpre1.link<-predict(m1.link,wtest.x)
  
  ###########  link function logistic regression #####
  m2.link<-glm(y.train~.,data = train.link,family =  binomial(link = "logit"))
  pre2.link<-predict(m2.link,test.link,type = "response")
  # Calculate auc
  m2.roc<-roc(test.link[,1],pre2.link,levels = c(0,1),direction='<')
  m2.auc<-auc(m2.roc,partial.auc.focus=c("specificity","sensitivey"))
  # prediction 
  wpre2.link<-predict(m2.link,wtest.x,type = "response")
  
  ########## link function SVM #######
  ## Convert y to factor
  train.link$y.train<-as.factor(train.link$y.train)
  test.link$y.train<-as.factor(test.link$y.train)
  m3.link<-svm(y.train~.,data=train.link,method="C-classification",kernal="radial")
  pre3.link<-predict(m3.link,newdata = test.link[,-1])
  # Calculate auc
  m3.roc<-roc(as.numeric(paste(test.link[,1])),as.numeric(pre3.link),levels = c(0,1),direction='<')
  m3.auc<-auc(m3.roc,partial.auc.focus=c("specificity","sensitivey"))
  # prediction 
  wpre3.link<-predict(m3.link,newdata = wtest.x)
  
  ####### link function random forest ##############
  m4.link<-randomForest(y.train~.,train.link)
  pre4.link<-predict(m4.link,test.link)
  # Calculate auc
  m4.roc<-roc(as.numeric(paste(test.link[,1])),as.numeric(pre4.link),levels = c(0,1),direction='<')
  m4.auc<-auc(m4.roc,partial.auc.focus=c("specificity","sensitivey"))
  # prediction 
  wpre4.link<-predict(m4.link,wtest.x)
  
  ###### link function xgboost ########
  m5.dtrain<-xgb.DMatrix(data = as.matrix(train.link[,-1]),label = as.numeric(paste(train.link[,1])))
  m5.dtest<-xgb.DMatrix(data = as.matrix(test.link[,-1]),label = test.link[,1])
  m5.link<-xgboost(m5.dtrain,nround = 30,objective = "binary:logistic")
  pre5.link<-predict(m5.link,m5.dtest)
  # Calculate auc
  m5.roc<-roc(test.link[,1],pre5.link,levels = c(0,1),direction='<')
  m5.auc<-auc(m5.roc,partial.auc.focus=c("specificity","sensitivey"))
  # prediction 
  wpre5.link<-predict(m5.link,as.matrix(wtest.x))
  
  # model selection
    mo.wpre<-cbind(wpre1.link,wpre2.link,wpre3.link,wpre5.link)
  mo.au<-c(m1.auc,m2.auc,m3.auc,m5.auc)
  mo.in<-which.max(mo.au) #index
  wres<-mo.wpre[,mo.in]

  roc.wglm <-roc(test[[1]]$y,wres,levels = c(0,1),direction='<')
  auc.wglm <-auc(roc.wglm,  partial.auc.focus=c("specificity", "sensitivity")) #0.7259
  per.wglm<-coords(roc.wglm, "best", ret=c( "accuracy","precision", "recall", "specificity"))
  if(nrow(per.wglm)!=1){
    per.wglm<-per.wglm[nrow(per.wglm),]
  }
  return(list(roc.wglm,per.wglm,auc.wglm,mo.in))
}
