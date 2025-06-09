wine <- read.csv("wine-quality-white-and-red.csv")
head(wine)

if_red <- ifelse(wine$type == "white", 0,1) #changes wine type to white = 0, red = 1
others <- subset(wine, select = c(2:13))
wine_data <- cbind(if_red, others) #official data set used 
head(wine_data)

#correlation plot by color 
library(corrplot)
corrplot(cor(wine_data), method = "color")

#scatterplot matrix
pairs(wine_data)

#histogram of quality
hist(wine_data$quality,
     main = "Histogram of Quality",
     xlab = "Quality",
     breaks=5,
     col = "purple",
     border = "black")
