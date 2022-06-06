## STQD6114 UNSTRUCTURED DATA ANALYTICS: PROJECT 1 ##

# Task 1: Find any website that have multiple pages regarding movies of 2 different genres

# i) Extract the information of the first 3 pages

library(rvest)

page_result = seq(from = 1, to = 101, by = 50)
romance <- paste0('https://www.imdb.com/search/title/?genres=romance&sort=num_votes,desc&start=', 
                  page_result, '&ref_=adv_nxt')
rom_title <- function(romance){
  url <- read_html(romance)
  nodes <- html_nodes(url, '.lister-item-header a')
  html_text(nodes)}
rom_title <- do.call(c,lapply(romance, rom_title))


rom_year <- function(romance){
  url <- read_html(romance)
  nodes <- html_nodes(url, '.text-muted.unbold')
  html_text(nodes)}
rom_year <- do.call(c,lapply(romance, rom_year))

rom_rating <- function(romance){
  url <- read_html(romance)
  nodes <- html_nodes(url, '.ratings-imdb-rating strong')
  html_text(nodes)}
rom_rating <- do.call(c,lapply(romance, rom_rating))

rom_votes <- function(romance){
  url <- read_html(romance)
  nodes <- html_nodes(url, '.sort-num_votes-visible span:nth-child(2)')
  html_text(nodes)}
rom_votes <- do.call(c,lapply(romance, rom_votes))

rom_director <- function(romance){
  url <- read_html(romance)
  nodes <- html_nodes(url, 'p:nth-child(5) a:nth-child(1)')
  html_text(nodes)}
rom_director <- do.call(c,lapply(romance, rom_director))


page_result = seq(from = 1, to = 101, by = 50)
horror <- paste0('https://www.imdb.com/search/title/?genres=horror&sort=num_votes,desc&start=', 
                 page_result, '&ref_=adv_nxt')
hor_title <- function(horror){
  url <- read_html(horror)
  nodes <- html_nodes(url, '.lister-item-header a')
  html_text(nodes)}
hor_title <- do.call(c,lapply(horror, hor_title))

hor_year <- function(horror){
  url <- read_html(horror)
  nodes <- html_nodes(url, '.text-muted.unbold')
  html_text(nodes)}
hor_year <- do.call(c,lapply(horror, hor_year))

hor_rating <- function(horror){
  url <- read_html(horror)
  nodes <- html_nodes(url, '.ratings-imdb-rating strong')
  html_text(nodes)}
hor_rating <- do.call(c,lapply(horror, hor_rating))

hor_votes <- function(horror){
  url <- read_html(horror)
  nodes <- html_nodes(url, '.sort-num_votes-visible span:nth-child(2)')
  html_text(nodes)}
hor_votes <- do.call(c,lapply(horror, hor_votes))

hor_director <- function(horror){
  url <- read_html(horror)
  nodes <- html_nodes(url, 'p:nth-child(5) a:nth-child(1)')
  html_text(nodes)}
hor_director <- do.call(c,lapply(horror, hor_director))


# ii) Build a dataset

movie_romance <- data.frame(rom_title, rom_year, rom_votes, rom_rating, rom_director)
names(movie_romance) <- c("Title", "Year", "Votes", "Rank", "Director")

movie_horror <- data.frame(hor_title, hor_year, hor_votes, hor_rating, hor_director)
names(movie_horror) <- c("Title", "Year", "Votes", "Rank", "Director")



# Task 2: Extract reviews from Twitter

library(rtweet)

health <- search_tweets(q = "vaccine AND exercise", n = 5, lang = "en", include_rts = FALSE)
health$text
health$source
health$display_text_width
users_data(health)
users_data(health)$screen_name
users_data(health)$name
users_data(health)$location
users_data(health)$description
users_data(health)$followers_count
users_data(health)$friends_count



sports <- search_tweets(q = "e-sports AND olympics", n = 5, lang = "en", include_rts = FALSE)
sports$text
sports$source
sports$display_text_width
users_data(sports)
users_data(sports)$screen_name
users_data(sports)$name
users_data(sports)$location
users_data(sports)$description
users_data(sports)$followers_count
users_data(sports)$friends_count



finance <- search_tweets(q = "inflation AND debt", n = 5, lang = "en", include_rts = FALSE)
finance$text
finance$source
finance$display_text_width
users_data(finance)
users_data(finance)$screen_name
users_data(finance)$name
users_data(finance)$location
users_data(finance)$description
users_data(finance)$followers_count
users_data(finance)$friends_count




# Task 3: Lyrics 

library(tm)
library(NLP)
library(stringr)

lyrics <- read.csv(file.choose(), header=F)

docs <- Corpus(VectorSource(lyrics))
writeLines(as.character(docs))

# i) data cleaning
docs <- tm_map(docs,removePunctuation)
docs <- tm_map(docs,content_transformer(tolower))
docs <- tm_map(docs,removeNumbers)
docs <- tm_map(docs,removeWords,stopwords("english"))
docs <- tm_map(docs,removeWords,c("ci", "im", "id", "can", "cant","cause","just","let","yet"))
docs <- tm_map(docs,stripWhitespace)
writeLines(as.character(docs))

# ii) convert to dtm and find frequency
dtm <- DocumentTermMatrix(docs)
inspect(dtm)
freq <- colSums(as.matrix(dtm))
length(freq)
ord <- order(freq,decreasing = T)
head(ord)
freq[head(ord)]

wf <- data.frame(names(freq),freq)
names(wf) <- c("term","freq")
head(wf)

library(ggplot2)
Subs <- subset(wf, freq>=3)
ggplot(Subs,aes(x=term, y=freq))+geom_bar(stat="identity")+
  theme(axis.text.x=element_text(angle=45,hjust=1))

library(wordcloud2)
wordcloud2(wf,shape="heart",size = 0.4,color="random-light", backgroundColor = "jetblack")

