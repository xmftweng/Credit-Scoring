# Randomly split data into k folds
CVgroup <- function(k,datasize){
  cvlist <- list()
  n <- rep(1:k,ceiling(datasize/k))[1:datasize]    
  temp <- sample(n,datasize)   
  z <- 1:k
  dataseq <- 1:datasize
  cvlist <- lapply(z,function(z) dataseq[temp==z])  
  return(cvlist)
}