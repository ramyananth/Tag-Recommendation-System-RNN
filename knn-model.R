install.packages(c('tm', 'SnowballC', 'wordcloud', 'topicmodels'))
library(tm)
library(SnowballC)
library(wordcloud)

install.packages('lsa')
require('lsa')

data1 = read.csv('./Random_Data_1_1000.csv', stringsAsFactors = FALSE)
data2 = read.csv('./Random_Data_1001_2000.csv', stringsAsFactors = FALSE)
data3 = read.csv('./Random_Data_2001_3000.csv', stringsAsFactors = FALSE)
data4 = read.csv('./Random_Data_3001_4000.csv', stringsAsFactors = FALSE)
data5 = read.csv('./Random_Data_4001_5000.csv', stringsAsFactors = FALSE)
data6 = read.csv('./Random_Data_5001_6000.csv', stringsAsFactors = FALSE)
data7 = read.csv('./Random_Data_6001_7000.csv', stringsAsFactors = FALSE)
data8 = read.csv('./Random_Data_7001_8000.csv', stringsAsFactors = FALSE)
data9 = read.csv('./Random_Data_8001_9000.csv', stringsAsFactors = FALSE)
data10 = read.csv('./Random_Data_9001_10000.csv', stringsAsFactors = FALSE)
data = rbind(data1, data2 ,data3, data4, data5, data6, data7, data8, data9, data10);
data = rbind(data1)
smp_size <- floor(0.90 * nrow(data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, 'content']
test <- data[-train_ind, 'content']

train_data_all = data[train_ind,];
test_data_all = data[-train_ind,];

myStopWords = processFile('./stopwords.txt')

processFile = function(filepath) {
  i = 1;
  con = file(filepath, "r")
  vec = vector();
  while ( TRUE ) {
    line = readLines(con, n = 1)
    if ( length(line) == 0 ) {
      break
    }
    vec[i] = line;
    i = i + 1;
  }
  
  close(con)
  vec;
}

#train the model on the training dataset
review_corpus = Corpus(VectorSource(train))
review_corpus = tm_map(review_corpus, content_transformer(tolower))
review_corpus = tm_map(review_corpus, removeNumbers)
review_corpus = tm_map(review_corpus, removePunctuation)
review_corpus = tm_map(review_corpus, removeWords, c("the", "and", "can", myStopWords, stopwords("en")))
review_corpus =  tm_map(review_corpus, stripWhitespace)

review_dtm <- DocumentTermMatrix(review_corpus)

review_dtm = removeSparseTerms(review_dtm, 0.99)


review_matrix = as.matrix(review_dtm)

data_matrix = matrix(nrow = nrow(review_matrix), ncol = nrow(review_matrix), dimnames = list(seq(1, nrow(review_matrix)), seq(1, nrow(review_matrix))));

for(i in seq(1, nrow(review_matrix))) {
  for(j in seq(1, nrow(review_matrix))) {
    data_matrix[i, j] = 1 - cosine(review_matrix[i, ], review_matrix[j, ])   
  }
}

cl = hclust(as.dist(data_matrix), method = "complete")
plot(cl)



# for test
for (t in seq(1, length(test))) {
  review_corpus_test = Corpus(VectorSource(test[t]))
  review_corpus_test = tm_map(review_corpus_test, content_transformer(tolower))
  review_corpus_test = tm_map(review_corpus_test, removeNumbers)
  review_corpus_test = tm_map(review_corpus_test, removePunctuation)
  review_corpus_test = tm_map(review_corpus_test, removeWords, c("the", "and", myStopWords, stopwords("en")))
  review_corpus_test =  tm_map(review_corpus_test, stripWhitespace)
  review_dtm_test <- DocumentTermMatrix(review_corpus_test)
  review_dtm_test = removeSparseTerms(review_dtm_test, 0.99) 
  
  x = review_dtm$dimnames$Terms
  y = review_dtm_test$dimnames$Terms
  
  y_matrix = as.matrix(review_dtm_test)
  
  new_y_matrix = matrix(ncol = length(x));
  
  for (i in seq(1, length(x))) {
    new_y_matrix[1, i] = tryCatch({
      if (y_matrix[1, x[i]] > -1) {
        y_matrix[1, x[i]]
      }
    }, error = function(error_condition) {
      num = 0
      return(num)
    });
  }
  dist_matrix = matrix(nrow = 1, ncol = nrow(review_matrix));
  
  for(j in seq(1, nrow(review_matrix))) {
    dist_matrix[1, j] = cosine(new_y_matrix[1, ], review_matrix[j, ])   
  }
  tag_predicted = knn_classifier(dist_matrix, 4);
  print(tag_predicted)
  true_tag_column = test_data_all[t, 15:109];
  true_tag_column_names = colnames(true_tag_column);
  true_tag = vector();
  for (i in seq(1, length(true_tag_column_names))) {
    if (true_tag_column[1, true_tag_column_names[i]] == 1) {
      true_tag[length(true_tag) + 1] = true_tag_column_names[i];
    }
  }
  matched = vector();
  matched = intersect(tag_predicted, true_tag);
  print(true_tag)
  percentCorrect = length(matched)/length(true_tag)
  print(paste("t:", t, "Matched: ", matched, "percent: ", percentCorrect))
}

test_model = function (k) {
  # for test
  total = 0;
  for (t in seq(1, length(test))) {
    review_corpus_test = Corpus(VectorSource(test[t]))
    review_corpus_test = tm_map(review_corpus_test, content_transformer(tolower))
    review_corpus_test = tm_map(review_corpus_test, removeNumbers)
    review_corpus_test = tm_map(review_corpus_test, removePunctuation)
    review_corpus_test = tm_map(review_corpus_test, removeWords, c("the", "and", myStopWords, stopwords("en")))
    review_corpus_test =  tm_map(review_corpus_test, stripWhitespace)
    review_dtm_test <- DocumentTermMatrix(review_corpus_test)
    review_dtm_test = removeSparseTerms(review_dtm_test, 0.99) 
    
    x = review_dtm$dimnames$Terms
    y = review_dtm_test$dimnames$Terms
    
    y_matrix = as.matrix(review_dtm_test)
    
    new_y_matrix = matrix(ncol = length(x));
    
    for (i in seq(1, length(x))) {
      new_y_matrix[1, i] = tryCatch({
        if (y_matrix[1, x[i]] > -1) {
          y_matrix[1, x[i]]
        }
      }, error = function(error_condition) {
        num = 0
        return(num)
      });
    }
    dist_matrix = matrix(nrow = 1, ncol = nrow(review_matrix));
    
    for(j in seq(1, nrow(review_matrix))) {
      dist_matrix[1, j] = cosine(new_y_matrix[1, ], review_matrix[j, ])   
    }
    tag_predicted = knn_classifier(dist_matrix, k);
    #print(tag_predicted)
    true_tag_column = test_data_all[t, 15:109];
    true_tag_column_names = colnames(true_tag_column);
    true_tag = vector();
    for (i in seq(1, length(true_tag_column_names))) {
      if (true_tag_column[1, true_tag_column_names[i]] == 1) {
        true_tag[length(true_tag) + 1] = true_tag_column_names[i];
      }
    }
    matched = vector();
    matched = intersect(tag_predicted, true_tag);
    #print(true_tag)
    percentCorrect = length(matched)/length(true_tag)
    #print(paste("t:", t, "Matched: ", matched, "percent: ", percentCorrect))
    total = (total + percentCorrect)/length(test);
  }
  #print(paste("Accuracy: ", total))
  total;
}



knn_classifier <- function(distance_matrix, k){
  output = vector(mode = "double", length = nrow(distance_matrix));
  # traverse the distance matrix row wise finding the top k similar sentences
  for (i in seq(nrow(distance_matrix))) {
    # fetches the ith row of the distance matrix
    thisRow = distance_matrix[i,];
    # sort the row and return their indexes
    sortedIndexes = sort(thisRow, index.return=TRUE)$ix
    # based on selected distance method return the top k values
    knnIndex = tail(sortedIndexes, k)
    #print(knnIndex)
    #print(data_matrix[1, knnIndex])
  }
  selectedTags = vector();
  for (i in seq(1, length(knnIndex))) {
    tags = data[train_ind[knnIndex[i]], 15:109];
    cNames = colnames(data[train_ind[knnIndex[i]], 15:109]);
    for (l in seq(1, length(cNames))) {
      if (!is.na(tags[1, cNames[l]]) && tags[1, cNames[l]] == 1) {
        selectedTags[length(selectedTags) + 1] = cNames[l];
      }
    } 
  }
  return(unique(selectedTags));
}

accuracies = vector();
for (i in seq(3, 10)) {
  print(paste("for k", i))
  accuracies[i] = test_model(i)
}

for_1 = 2.61330523173046e-09

#####################################################################################


article = "This fall, students at Santa Clara University will have a new way to get around campus — a self-driving shuttle.

The university has invited Auro Robotics to its palm tree-lined campus to test a prototype. The vehicle will be restricted to a top speed of 10 mph, and share the concrete and brick walkways with pedestrians.

Students at the private university in Silicon Valley’s shadow aren’t the only ones to experience the buzzed-about technology.

Earlier this week, two students at the University of Waterloo showed off the self-driving golf cart they built over summer break. As the cart did a loop around campus, including on a road with passenger cars, the university’s president, Feridun Hamdullahpur, rode shotgun. The University of Michigan also has plans to test a self-driving vehicle on campus roads, in addition to the miniature city it built on campus for testing.

While only a handful of U.S. states have laws welcoming self-driving vehicles, college campuses are a different story. The combination of curious, open-minded college communities and private land — in many ways, municipalities unto themselves — has made college campuses fertile ground for experimentations in self-driving vehicles.

“It’s a good focal point to have some conversations academically and let people see some of the advanced technologies they read about,” said Christopher Kitts, a robotics professor at Santa Clara University.

Earlier this year, Auro Robotics reached out to Santa Clara’s engineering department, which was excited by the chance to expose its campus to the technology.

The university did some preliminary tests with Auro Robotics last week and is awaiting final sign-off from its general counsel before beginning regular testing that could last much of the semester.

An Auro Robotics engineer will be riding in the cart as it travels around campus for safety’s sake, and to see how students respond to the cart.

Alex Rodrigues, left, and Michael Skupien pose with their cart outside the garage they built it in. (Varden Labs)

“We want to prove the viability of self-driving technology. And we want to learn the user experience around it,” said Auro Robotics chief executive Nalin Gupta. The start-up, which is incubating in the prestigious accelerator Y-Combinator, is developing self-driving shuttles for universities, resorts, large industrial sites and theme parks. “We believe these are the perfect starting points to enter into the days of self-driving mobility.”

It was drawn to those settings because of the lack of regulation, which will allow for easier testing and deployment of vehicles.

Auro’s prototype is outfitted with sensors that look 200 meters ahead of its path so that it can avoid obstacles.

“There will certainly be times when they operate on our campus, during the switching of classes and things like that, where they’re going to be in a sea of students,” Kitts said. “So being able to strategically operate in that environment is an interesting challenge they need to overcome.”

Alex Rodrigues, 19, and Michael Skupien, 20, retrofitted a golf cart with self-driving technology so it could drive autonomously around Waterloo’s campus. They received a $35,000 grant this summer and launched Varden Labs.

Like Auro Robotics, they’re targeting the self-driving shuttle space. This way they aren’t directly competing with deep-pocketed automakers and large tech companies, which are generally believed to be focused on the passenger car market.

Rodrigues is a believer that we’ll witness a birth of many new new vehicle forms.

“Today almost every person has a car to carry five people. It’s a huge waste,” said Rodrigues, noting that most trips are taken only by one person.

After teaming with Skupien to build the cart in his parent’s garage, they transported it back to campus. It certainly won’t be the last self-driving vehicle that finds its way to a college campus."


review_corpus_test = Corpus(VectorSource(article))
review_corpus_test = tm_map(review_corpus_test, content_transformer(tolower))
review_corpus_test = tm_map(review_corpus_test, removeNumbers)
review_corpus_test = tm_map(review_corpus_test, removePunctuation)
review_corpus_test = tm_map(review_corpus_test, removeWords, c("the", "and", myStopWords, stopwords("en")))
review_corpus_test =  tm_map(review_corpus_test, stripWhitespace)
review_dtm_test <- DocumentTermMatrix(review_corpus_test)
review_dtm_test = removeSparseTerms(review_dtm_test, 0.99) 

x = review_dtm$dimnames$Terms
y = review_dtm_test$dimnames$Terms

y_matrix = as.matrix(review_dtm_test)

new_y_matrix = matrix(ncol = length(x));

for (i in seq(1, length(x))) {
  new_y_matrix[1, i] = tryCatch({
    if (y_matrix[1, x[i]] > -1) {
      y_matrix[1, x[i]]
    }
  }, error = function(error_condition) {
    num = 0
    return(num)
  });
}
dist_matrix = matrix(nrow = 1, ncol = nrow(review_matrix));

for(j in seq(1, nrow(review_matrix))) {
  dist_matrix[1, j] = cosine(new_y_matrix[1, ], review_matrix[j, ])   
}
tag_predicted = knn_classifier(dist_matrix, 6);
print(tag_predicted)

