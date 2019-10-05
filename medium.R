install.packages('rvest');
install.packages('textcat');
require('rvest');
require('textcat');

install.packages('dplyr');
require('dplyr')

data = read.csv('./Medium_Clean.csv', stringsAsFactors = FALSE);

# for random selection of script =======
randomRows = function(df,n){
  return(df[sample(nrow(df),n),])
}

data = randomRows(data, 1000)
# for random selection of script ========


contentList = data.frame();
startIndex = 1;
endIndex = 1000;
for (i in seq(1, 1000)) {
  url = as.vector(data[i, 'url']);
  contentList[i, 'url'] = url;
  texts = tryCatch({
    webpage = read_html(url);
    pTags = html_nodes(webpage, 'p');
    pText = html_text(pTags);
    language = textcat::textcat(pText);
    if (!is.na(language[1]) && language[1] == 'english') {
      print(paste("Success on:", i, url))
      #logger(paste("Success on:", i, url))
      pText;
    } else {
      print(paste("Fail on:", i, url, "Language"))
      #logger(paste("Fail on:", i, url, "Language"));
      msg = NA;
      msg;
    }
  }, error = function(error_condition) {
    print(paste("Fail on:", i, url, "HTTP ERROR", error_condition))
    #logger(paste("Fail on:", i, url, "HTTP ERROR", error_condition));
    msg = NA;
    msg;
  });
  contentList[i, 'content'] = paste(texts, sep = " ", collapse = " ");
  if (i%%1000 == 0) {
    endIndex = i;
    new_data=merge(data[startIndex:endIndex,], contentList, by.y = "url");
    new_data_filter_1 = dplyr::filter(new_data, new_data['content'] != 'NA')
    print("Writing the file");
    write.csv(new_data_filter_1, file = paste("Data", "_", startIndex,"_", endIndex, ".csv", sep = ""))
    startIndex = i+1;
    contentList = data.frame();
  }
}

logger = function(message) {
  write(message, file="log.txt", append=TRUE)
}
