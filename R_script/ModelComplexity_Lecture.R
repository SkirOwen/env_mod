
library(AICcmodavg) # Provides a function to compute AICc

#######################################
# Loading and inspecting data 
######################
MyData <- read.csv(file=file.path("Data","MyData2020.csv"))

head(MyData) # Inspect first rows of the data

# retrieve x and y data and create separated vectors 
y <- MyData$y
x <- MyData$x

# Plotting data
plot(x,y, main="Data points")

#######################################
# Fitting a Line (Polynomial of degree 1) to the data 
######################
model <- lm(y ~ x) # Fitting data to a linear model
# Which is equivalent to:
model <- lm(y ~ poly(x, 1, raw=TRUE)) 

summary(model) #Displays model fit

#Display model fit on a plot 
plot(x,y, main="Degree 1")
lines(x,model$fit,col='green',lwd=3)

#Predicts 95% confidence model error bounds (P = 0.05)
predicted.intervals <- predict(model,data.frame(x),interval='confidence',level=0.95)
#Add model error bounds
lines(x,predicted.intervals[,2],col='green',lwd=2, lty = "longdash")
lines(x,predicted.intervals[,3],col='green',lwd=2, lty = "longdash")

####
#Save plot using RStudio export button
###

#Model selection criteria computation
#BIC
BIC1 <- BIC(model)
#AIC
AIC1 <- AIC(model)
#AICc
AICc1 <- AICc(model)

#######################################
# Fitting a quadratic function (degree 2 polynomial) to the data 
######################
model <- lm(y ~ x+I(x^2)) # Fitting data to a quadratic model

# Which is equivalent to:
model <- lm(y ~ poly(x, 2, raw=TRUE)) 

summary(model) #Displays model fit

#Display model fit on a plot 
plot(x,y, main="Degree 1")
lines(x,model$fit,col='green',lwd=3)

#Predicts 95% confidence model error bounds (P = 0.05)
predicted.intervals <- predict(model,data.frame(x),interval='confidence',level=0.95)
#Add model error bounds
lines(x,predicted.intervals[,2],col='green',lwd=2, lty = "longdash")
lines(x,predicted.intervals[,3],col='green',lwd=2, lty = "longdash")

####
#Save plot using RStudio export button
###

#Model selection criteria computation
#BIC
BIC2 <- BIC(model)
#AIC
AIC2 <- AIC(model)
#AICc
AICc2 <- AICc(model)

#######################################
# Fitting a cubic model (degree 2 polynomial) to the data 
######################
model <- lm(y ~ x+I(x^2)+I(x^3)) # Fitting data to a cubic model

# Which is equivalent to:
model <- lm(y ~ poly(x, 3, raw=TRUE)) 

summary(model) #Displays model fit

#Display model fit on a plot 
plot(x,y, main="Degree 1")
lines(x,model$fit,col='green',lwd=3)

#Predicts 95% confidence model error bounds (P = 0.05)
predicted.intervals <- predict(model,data.frame(x),interval='confidence',level=0.95)
#Add model error bounds
lines(x,predicted.intervals[,2],col='green',lwd=2, lty = "longdash")
lines(x,predicted.intervals[,3],col='green',lwd=2, lty = "longdash")

####
#Save plot using RStudio export button
###

#Model selection criteria computation
#BIC
BIC3 <- BIC(model)
#AIC
AIC3 <- AIC(model)
#AICc
AICc3 <- AICc(model)

#######################################
# Selecting the best model 
######################

BIC.v<-c(BIC1,BIC2,BIC3) # Vector holding BIC values
BIC.v
AIC.v<-c(AIC1,AIC2,AIC3) # Vector holding AIC values
AIC.v
AICc.v<-c(AICc1,AICc2,AICc3) # Vector holding AICc values
AICc.v

# Plots displying model fit criteria values
model.order <- 1:3

plot(model.order,BIC.v,col='blue',ylab='BIC',xlab='Order',main='BIC')
lines(model.order,BIC.v,col='blue')

plot(model.order,AIC.v,col='red',ylab='AIC',xlab='Order',main='AIC')
lines(model.order,AIC.v,col='red')

plot(model.order,AICc.v,col='green',ylab='AICc',xlab='Order',main='AICc')
lines(model.order,AICc.v,col='green')