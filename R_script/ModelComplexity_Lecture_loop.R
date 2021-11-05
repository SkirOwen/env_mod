
library(AICcmodavg) # Provides a function to compute AICc

#######################################
# Loading and inspecting data 
######################
MyData <- read.csv(file=file.path("Data", "MyData2020.csv"))

head(MyData) # Inspect first rows of the data

# retrieve x and y data and create separated vectors 
y <- MyData$y
x <- MyData$x

# Plotting data
plot(x,y, main="Data points")

#######################################
# Fitting polynomial expressions using a for loop 
######################

i_max <- 3 # define highest order of the polynomial function you would like to consider

# Create empty vectors to store the selection criteria 
BIC.v <-  rep(NaN, i_max) 
AIC.v <-  rep(NaN, i_max) 
AICc.v <- rep(NaN, i_max) 

#i<-1 # Use to test the loop
for(i in 1:i_max){
  print(paste("Fitting polynomial of degree ", i, sep=""))
  model <- lm(y ~ poly(x, i, raw=TRUE)) # Notice that variable "i" denotes the degree of the polynoom.
  
  png(filename = file.path("Plots", paste("degree", i, ".png", sep="")))
    plot(x, y, main=paste("Degree ", i, sep = ""))
    #Display model fit on the plot 
    lines(x, model$fit, col='green', lwd=3)
  
    #Predicts 95% confidence model error bounds (P = 0.05)
    predicted.intervals <- predict(model, data.frame(x), interval='confidence', level=0.95)
  
    #Add model error bounds
    lines(x, predicted.intervals[, 2], col='green', lwd=2, lty = "longdash")
    lines(x, predicted.intervals[, 3], col='green', lwd=2, lty = "longdash")
  dev.off()
  
  #Model selection criteria computation
  #BIC
  BIC.v[i] <- BIC(model)
  #AIC
  AIC.v[i] <- AIC(model)
  #AICc
  AICc.v[i] <- AICc(model)
}

# Plots displying model fit criteria values
model.order <- seq(1:i_max)

plot(model.order, BIC.v, col='blue', ylab='BIC', xlab='Order', main='BIC')
lines(model.order, BIC.v, col='blue')

plot(model.order, AIC.v, col='red', ylab='AIC', xlab='Order', main='AIC')
lines(model.order, AIC.v, col='red')

plot(model.order, AICc.v, col='green', ylab='AICc', xlab='Order', main='AICc')
lines(model.order, AICc.v, col='green')

