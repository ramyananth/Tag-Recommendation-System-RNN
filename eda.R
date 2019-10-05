install.packages('text2vec');
require('text2vec');

data_new = read.csv('./Data_1_1000.csv', stringsAsFactors = FALSE);

data = data_new[, c(12, 13, 15:109)]
cNames = colnames(data);
filtered_data = data.frame(Tag=character(), AvgRead=double(), claps=double(), visibility=double(), ratio=double(), stringsAsFactors = FALSE)
for (i in seq(3, ncol(data))) {
  index = which(data[, i] == 1);
  sum = sum(data[index, 2]);
  count = length(index)
  ratio = sum/count;
  avg = sum(data[index, 1])/count;
  filtered_data[nrow(filtered_data) + 1,] = list(Tag=cNames[i], AvgRead=avg, claps=sum, visibility=count, ratio=sum/count)
}
filtered_data[, 'normalClaps'] = normalizer(filtered_data[, 'claps']);
filtered_data[, 'normalVisibility'] = normalizer(filtered_data[, 'visibility']);

hist(filtered_data[, 'ratio'])

normalizer = function (data) {
  meanRatio = mean(data, na.rm = TRUE);
  (data - meanRatio)/sd(data, na.rm = TRUE)  
}

model = lm(filtered_data[, 'AvgRead'], filtered_data[, 'ratio'])
coeff=coefficients(model)
plot(filtered_data[, 'AvgRead'], filtered_data[, 'ratio'])
abline(coef = coeff)

library(philentropy)
data_for_dist = data_new[, 15:109]
jaccard_dist = distance(data_for_dist, method="jaccard")
model = hclust(jaccard_dist, method = "single")

km = kmeans(data_for_dist, centers = 2, algorithm = "Lloyd")
