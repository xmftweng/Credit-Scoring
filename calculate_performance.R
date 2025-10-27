# calculate the performance ###
calcper<-function(per,auc){
  f<-2*per[2]*per[3]/(per[2]+per[3])
  colnames(f)<-"f"
  r<-cbind(per,f,auc)
  r<-r[,c(1,5,3,4,6,2)]
  return(r)
}