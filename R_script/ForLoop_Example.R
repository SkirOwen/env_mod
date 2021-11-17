#########################################################
#  For loop example
################################

i_max <- 10 # define Max number of loops
i_max

A <- rep(NaN,i_max) 
A

for(i in 1:i_max)
{
  print(i)
  A[i] <- i*5
}

A