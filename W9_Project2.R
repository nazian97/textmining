## STQD6114: Project 2 ##

### Data Acquisition
# 30 article news from New Straits Times regarding financial issues were extracted
# and saved in .txt file format.

## Loading data into R
library(tm)

mytext <- DirSource("Unstructured/Project2")
docs <- VCorpus(mytext)

# get the details of the files
inspect(docs)

# print the first .txt file in the folder
writeLines(as.character(docs[[31]]))

## Pre-processing

# see the available transformations for data cleaning
getTransformations()

# create a custom transformation
toSpace <- content_transformer(function(x,pattern){return(gsub(pattern," ",x))})

# use the custom transformer to eliminate colons and hyphens etc
docs <- tm_map(docs, toSpace, "-")
docs <- tm_map(docs, toSpace, "–")
docs <- tm_map(docs, toSpace, "—")
docs <- tm_map(docs, toSpace, ":")
docs <- tm_map(docs, toSpace, "'")

# remove punctuation
docs <- tm_map(docs, removePunctuation)

# convert the words to lower case using content_transformer
docs <- tm_map(docs,content_transformer(tolower))

# strip digits
docs <- tm_map(docs, removeNumbers)

# remove stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs,removeWords,c("s","b","m","g","ve","q","will","also","afp","per","cent","can",
                                  "said","lot","bernama","sri")) # additional unmeaningful words

# convert to root word
docs <- tm_map(docs, content_transformer(gsub), pattern = "prices", replacement = "price")
docs <- tm_map(docs, content_transformer(gsub), pattern = "raised", replacement = "rise")
docs <- tm_map(docs, content_transformer(gsub), pattern = "raise", replacement = "rise")

# strip whitespace
docs <- tm_map(docs, stripWhitespace)

# convert to document term matrix
dtm <- DocumentTermMatrix(docs)


### Task 1: Perform topic modelling analysis using LDA

## 1. Using the dataset in Data Acquisition section, create five topics, k=5

#  use LDA() function from topicmodels package
library(topicmodels)
ap_lda <- LDA(dtm, k = 5, control = list(seed = 1234)) # set k=5 to create a five-topic LDA model.
ap_lda

## 2. Perform relevant analysis

# i.
# extract the per-topic-per-word probabilities
library(tidytext)
ap_topics <- tidy(ap_lda, matrix = "beta")
ap_topics # this has turned the model into a one-topic-per-term-per-row format

# use dplyr’s slice_max() to find the 10 terms that are most common within each topic
# use ggplot2 to visualize the 10 terms
library(ggplot2)
library(dplyr)

ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

ap_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()

# ii.
# find the greatest difference in beta between topic 2 & 3
library(tidyr)

beta_wide <- ap_topics %>%
  mutate(topic = paste0("topic", topic)) %>%
  pivot_wider(names_from = topic, values_from = beta) %>% 
  filter(topic5 > .004 | topic1 > .004) %>%
  mutate(log_ratio = log2(topic5 / topic1))

beta_wide %>%
  group_by(direction = log_ratio > 0) %>%
  slice_max(abs(log_ratio), n = 10) %>% 
  ungroup() %>%
  mutate(term = reorder(term, log_ratio)) %>%
  ggplot(aes(log_ratio, term)) +
  geom_col() +
  labs(x = "Log2 ratio of beta in topic 5 / topic 1", y = NULL)

# iii.
# per-document per topic probabilities
ap_documents <- tidy(ap_lda, matrix = "gamma")
ap_documents
# the model estimates that each word in the australia-hikes-interest-rates.txt 
# document has only a 0% probability of coming from topic 1
# now that we have these topic probabilities, we can see how well the unsupervised 
# learning did at distinguishing the five topics

# reseparate the document name into news category and index
ap_documents2 <- ap_documents %>%
  separate(document, c("document", "index"), sep = ".txt", convert = TRUE)
ap_documents3 <- ap_documents2%>%
  separate(document, c("category", "index"), sep = "_", convert = TRUE)

# reorder categories in order of topic 1-5 etc before plotting
ap_documents3 %>%
  mutate(title = reorder(category, gamma * topic)) %>%
  ggplot(aes(factor(index), gamma)) +
  geom_boxplot() +
  facet_wrap(~ category) +
  labs(x = "topic", y = expression(gamma))

# find the topic that was most associated with each index using slice_max()
index_classifications <- ap_documents3 %>%
  group_by(category, index) %>%
  slice_max(gamma) %>%
  ungroup()

# compare each to the “consensus” topic for each category (the most common topic among its chapters)
# and see which were most often misidentified.
news_topics <- index_classifications %>%
  count(category, topic) %>%
  group_by(category) %>%
  slice_max(n, n = 1) %>% 
  ungroup() %>%
  transmute(consensus = category, topic)

index_classifications %>%
  inner_join(news_topics, by = "topic") %>%
  filter(category != consensus) 

#iv.
# by word assignments: augment

# take the original document-word pairs and 
# find which words in each document were assigned to which topic
assignments <- augment(ap_lda, data = dtm)
assignments

# combine this assignments table with the consensus news titles 
# to find which words were incorrectly classified
assignments <- assignments %>%
  separate(document, c("category", "index"), 
           sep = "_", convert = TRUE) %>%
  inner_join(news_topics, by = c(".topic" = "topic"))


# most commonly mistaken words
wrong_words <- assignments %>%
  filter(category != consensus)

wrong_words %>%
  count(category, consensus, term, wt = count) %>%
  ungroup() %>%
  arrange(desc(n))


### Task 2: Perform text clustering

## 1. Construct data clustering by using k-means, hierarchical and HDBScan algorithms

# text representation
# present text data numerically, weighted TF-IDF
tdm.tfidf <- weightTfIdf(dtm)
# we remove A LOT of features. R is natively very weak with high dimensional matrix
tdm.tfidf <- removeSparseTerms(tdm.tfidf, 0.999)
# there is the memory-problem part
# - native matrix isn't "sparse-compliant" in the memory
# - sparse implementations aren't necessary compatible with clustering algorithms
tfidf.matrix <- as.matrix(tdm.tfidf)
inspect(dtm)

# cosine distance matrix (useful for specific clustering algorithms)
library(proxy)
dist.matrix = dist(tfidf.matrix, method = "cosine")
dim(dist.matrix)  # 31x31, 31 docs, 31 terms

## 2. Perform the relevant analysis on each of the clustering algorithms 

library(factoextra)
library(gridExtra)
a <- fviz_nbclust(tfidf.matrix, FUNcluster = kmeans, method = "silhouette") + theme_classic() 
b <- fviz_nbclust(tfidf.matrix, FUNcluster = cluster::pam, method = "silhouette") + theme_classic() 
c <- fviz_nbclust(tfidf.matrix, FUNcluster = cluster::clara, method = "silhouette") + theme_classic() 
d <- fviz_nbclust(tfidf.matrix, FUNcluster = hcut, method = "silhouette") + theme_classic() 
grid.arrange(a,b,c,d,ncol=2)

# running the clustering algorithms
truth.K=2
clustering.kmeans <- kmeans(tfidf.matrix, truth.K)
clustering.hierarchical <- hclust(dist.matrix, method = "ward.D2")
library(dbscan)
clustering.dbscan <- hdbscan(dist.matrix, minPts=10)

# kmeans
library(cluster)
clusplot(as.matrix(dist.matrix), clustering.kmeans$cluster, 
         color = T, shade = T, labels = 2, lines = 0)

layout(matrix(1:1,ncol=1))
truth.K=3 # repeat with 3 clusters
clustering.kmeans <- kmeans(tfidf.matrix, truth.K)
clusplot(as.matrix(dist.matrix), clustering.kmeans$cluster, 
         color = T, shade = T, labels = 2, lines = 0)

truth.K=4 # repeat with 4 clusters
clustering.kmeans <- kmeans(tfidf.matrix, truth.K)
clusplot(as.matrix(dist.matrix), clustering.kmeans$cluster, 
         color = T, shade = T, labels = 2, lines = 0)

# kmeans – determine the optimum number of clusters (elbow method)
# look for “elbow” in plot of summed intra-cluster distances (withinss) as fn of k
wss <- 2:30
for (i in 2:30) wss[i] <- sum(kmeans(dist.matrix,centers=i,nstart=25)$withinss)
plot(2:30, wss[2:30], type="b", xlab="Number of Clusters",ylab="Within groups sum of squares")

# hierarchical
summary(clustering.hierarchical)
plot(clustering.hierarchical)
rect.hclust(clustering.hierarchical,2)

# dbscan
plot(as.matrix(dist.matrix), col=clustering.dbscan$cluster + 1L)

# merge the results between the 3 previous clustering
master.cluster <- clustering.kmeans$cluster
slave.hierarchical <- cutree(clustering.hierarchical, k = truth.K)
slave.dbscan <- clustering.dbscan$cluster

# get an idea of what’s in each of these clusters
p_words <- colSums(tfidf.matrix) / sum(tfidf.matrix)
cluster_words <- lapply(unique(slave.hierarchical), function(x){
  rows <- tfidf.matrix[ slave.hierarchical == x , ]
  # for memory's sake, drop all words that don't appear in the cluster
  rows <- rows[ , colSums(rows) > 0 ]
  colSums(rows) / sum(rows) - p_words[ colnames(rows) ]
})


# create a summary table of the top 5 words defining each cluster
cluster_summary <- data.frame(cluster = unique(slave.hierarchical),
                              size = as.numeric(table(slave.hierarchical)),
                              top_words = sapply(cluster_words, function(d){
                                paste(
                                  names(d)[ order(d, decreasing = TRUE) ][ 1:5 ], 
                                  collapse = ", ")
                              }),
                              stringsAsFactors = FALSE)
                     

# plotting results
library(colorspace)
points <- cmdscale(dist.matrix, k=2)
palette <- diverge_hcl(truth.K)

layout(matrix(1:3,ncol=1))
plot(points, main="K-Means Clustering", col=as.factor(master.cluster),
     mai=c(0,0,0,0), mar=c(0,0,0,0),
     xaxt='n', yaxt='n', xlab='', ylab='')
plot(points, main="Hierarchical Clustering", col=as.factor(slave.hierarchical),
     mai=c(0,0,0,0), mar=c(0,0,0,0),
     xaxt='n', yaxt='n', xlab='', ylab='')
plot(points, main="Density-based Clustering", col=as.factor(slave.dbscan),
     mai=c(0,0,0,0), mar=c(0,0,0,0),
     xaxt='n', yaxt='n', xlab='', ylab='')

table(master.cluster)
table(slave.hierarchical)
table(slave.dbscan) # no cluster, probably not dense, data are noise

layout(matrix(1:1,ncol=1))


### Task 3: Sentiment analysis

# loading libraries
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(syuzhet)
library(ggplot2)
library(tidyr)
library(dplyr)
library(tidytext)

# loading dataset
data  <- read.csv("Unstructured/archive/Reviews.csv")
dataReviews <- head(data$Text,500)

# load the data as a corpus
reviews <- Corpus(VectorSource(dataReviews))

# replace punctuation and others with space
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
reviews <- tm_map(reviews, toSpace, "[[:punct:]]")
reviews <- tm_map(reviews, toSpace, "[[:cntrl:]]")
reviews <- tm_map(reviews, toSpace, "\\d+")
reviews <- tm_map(reviews, toSpace, "\n")
# remove punctuations
reviews <- tm_map(reviews, removePunctuation)
# convert the text to lower case
reviews <- tm_map(reviews, content_transformer(tolower))
# remove numbers
reviews <- tm_map(reviews, removeNumbers)
# remove english common stopwords
reviews <- tm_map(reviews, removeWords, stopwords("english"))
# specify your custom stopwords as a character vector
reviews <- tm_map(reviews, removeWords, c("s","t","b","d","can","don","br","ve",
                                          "m", "y","ie","isn","aren","f","will",
                                          "wasn","ll","doesn","wouldn","haven",
                                          "didn","lil","re","g","q","st","r","l",
                                          "x","k","im","hadn","w","c","http","www"))
# convert to root word
reviews <- tm_map(reviews, content_transformer(gsub), pattern = "works", replacement = "work")
reviews <- tm_map(reviews, content_transformer(gsub), pattern = "sweetness", replacement = "sweet")
reviews <- tm_map(reviews, content_transformer(gsub), pattern = "recommended", replacement = "recommend")
reviews <- tm_map(reviews, content_transformer(gsub), pattern = "loves", replacement = "love")
reviews <- tm_map(reviews, content_transformer(gsub), pattern = "loved", replacement = "love")
reviews <- tm_map(reviews, content_transformer(gsub), pattern = "likes", replacement = "like")
reviews <- tm_map(reviews, content_transformer(gsub), pattern = "liked", replacement = "like")
reviews <- tm_map(reviews, content_transformer(gsub), pattern = "convenience", replacement = "convenient")
# eliminate extra white spaces
reviews <- tm_map(reviews, stripWhitespace)

# print the cleaned data
writeLines(as.character(reviews))

# build a term-document matrix
dtm <- TermDocumentMatrix(reviews)
m <- as.matrix(dtm)

# sort by decreasing value of frequency
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)

# display the top 5 most frequent words
head(d, 10)

# barchart representation of the top 10 Words most appeared
barplot(d[1:10,]$freq, las = 2, names.arg = d[1:10,]$word,
        col ="tan", main ="Top 10 most frequent words",
        ylab = "Word frequencies")

# generate word cloud
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 5,
          max.words=100, random.order=FALSE, rot.per=0.40, 
          colors=brewer.pal(8, "Dark2"))

# regular sentiment score using get_sentiment() function
syuzhet_vector <- get_sentiment(dataReviews, method="syuzhet")
# see the first row of the vector
head(syuzhet_vector)
# see summary statistics of the vector
summary(syuzhet_vector)

# bing
bing_vector <- get_sentiment(dataReviews, method="bing")
head(bing_vector)
summary(bing_vector)

# affin
afinn_vector <- get_sentiment(dataReviews, method="afinn")
head(afinn_vector)
summary(afinn_vector)

# nrc
nrc_vector <- get_sentiment(dataReviews, method="nrc")
head(nrc_vector)
summary(nrc_vector)

# compare the first row of each vector using sign function
rbind(sign(head(syuzhet_vector)),sign(head(bing_vector)),
      sign(head(afinn_vector)),sign(head(nrc_vector)))


# most common positive and negative words

tidyword <- d[1:500,] %>% rename(n=freq)

bing_word_counts <- tidyword %>%
  inner_join(get_sentiments("bing")) %>%
  ungroup()
head(bing_word_counts)

# visualize the above 
bing_word_counts %>%
  group_by(sentiment) %>%
  slice_max(n, n = 10) %>% 
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(n, word, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(x = "Contribution to sentiment",
       y = NULL)

# emotion classification
# run nrc sentiment analysis to return data frame with emotions in each row
d <- get_nrc_sentiment(dataReviews)

# head(d,5) - to see top 5 lines of the get_nrc_sentiment dataframe
head (d,5)

# transpose
td<-data.frame(t(d))
# the function rowSums computes column sums across rows for each level of a grouping variable.
td_new <- data.frame(rowSums(td[2:253]))
# transformation and cleaning
names(td_new)[1] <- "count"
td_new <- cbind("sentiment" = rownames(td_new), td_new)
rownames(td_new) <- NULL
td_new2<-td_new[1:8,]



# plot 1 - count of words associated with each sentiment
quickplot(sentiment, data=td_new2, weight=count, geom="bar", fill=sentiment, ylab="count")+ggtitle("Survey sentiments")

# plot 2 - count of words associated with each sentiment, expressed as a percentage
barplot(
  sort(colSums(prop.table(d[, 1:8]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  las = 1, 
  main = "Emotions in Text", xlab="Percentage"
)
