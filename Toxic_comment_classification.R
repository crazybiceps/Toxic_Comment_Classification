# Loading Libraries                                                   ----
library(dplyr)
library(tidyr)
library(keras)
library(rebus)
library(tibble)
library(readr)
library(ggplot2)
library(stringr)
library(MLmetrics)
library(caret)
library(corrplot)
library(e1071)
library(tidytext)
library(stringr)
library(widyr)
library(reshape2)
library(caTools)
# library(qdap)
library(qdapRegex)

w <- read.csv("/home/kawal/D/jigsaw-toxic-comment-classification-challenge/train.csv")

# Converting text to sequences                                        ----

text <- w[ , 2] %>% 
  as.character() %>% 
  tolower()

data("stop_words")

maxlen    <- 100
max_words <- 12000

# text <- str_replace_all(text, "[^[:alnum:]]", " ") %>% 
#   rm_white()

tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(text)

# Using glove word embedding      ----
glove_dir <- "/home/kawal/D/glove.6B"
lines <- readLines(file.path(glove_dir , "glove.6B.100d.txt"))

embedding_index <- new.env(hash = T , parent = emptyenv())

for (i in 1:length(lines)) { 
  line <- lines[[i]]
  values <- strsplit(line , " ")[[1]]
  word <- values[[1]]
  embedding_index[[word]] <- as.double(values[-1])
}

embedding_dim <- 100

embedding_matrix <- array(0 , c(max_words , embedding_dim))

word_index <- tokenizer$word_index

for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embedding_index[[word]]
    if(!is.null(embedding_vector))
      embedding_matrix[index+1 , ] <- embedding_vector
  }
}

# w$comment_text <- str_replace_all(w$comment_text  , "[^[:alnum:]]", " " )

# column_predictions <- function( w , text , tokenizer , maxlen = 75 , 
#                                 max_words = 10000 , col_no = 3){

set.seed(3.14)
train_rows <- sample.split(w[ , col_no], SplitRatio = .7)

sequences <- texts_to_sequences(tokenizer , text[train_rows])

x_train   <- sequences %>% 
  pad_sequences(maxlen = maxlen)

sequences <- texts_to_sequences(tokenizer , text[!train_rows])

x_test    <- sequences %>%
  pad_sequences(maxlen = maxlen)

y_train <- w[train_rows ,  -c(1 , 2)]
y_test  <- w[!train_rows , -c(1 , 2)]

# x_train_validation_rows = sample.split(y_train[ , col_no - 2] , SplitRatio = .7)

# x_validation <- x_train[!(x_train_validation_rows) , ]
# x_train      <- x_train[x_train_validation_rows  , ]
# 
# y_validation <- y_train[!x_train_validation_rows , ]
# y_train      <- y_train[x_train_validation_rows  , ]

# Model                                                               ----
text_input <- layer_input(shape = list(NULL) , dtype = "int32" , name = "texts")

embedded_text <- text_input %>%
  layer_embedding(input_dim = max_words , output_dim = embedding_dim , input_length = maxlen , 
                  name = "embed")

model <- embedded_text %>%
  # layer_spatial_dropout_1d(rate = .5) %>% 
  layer_conv_1d(filters = 32 , kernel_size = 5 , activation = "relu" , name = "conv1") %>%
  # layer_max_pooling_1d(pool_size = 3) %>%
  # layer_spatial_dropout_1d(rate = .5) %>% 
  # # layer_conv_1d(filters = 64 , kernel_size = 3 , activation = "relu" , name = "conv2") %>%
  # # layer_max_pooling_1d(pool_size = 3) %>%
  # # layer_conv_1d(filters = 64 , kernel_size = 3 , activation = "relu" , name = "conv3") %>%
  # layer_max_pooling_1d(pool_size = 2) %>%
  # layer_conv_1d(filters = 64 , kernel_size = 3 , activation = "relu" , name = "conv4") %>%
  layer_global_max_pooling_1d()

toxic_pred <- model %>%
  layer_dense(units = 64 , activation = "relu" , kernel_regularizer = regularizer_l2(l  = .001)) %>%
  layer_dropout(rate = .5) %>%
  # layer_dense(units = 128 , activation = "relu" , kernel_regularizer = regularizer_l2(l  = .001)) %>% 
  # layer_dropout(rate = .5) %>% 
  layer_dense(units = 1 , activation = "sigmoid"  , name = "toxic")

toxic_model <- keras_model(
 text_input , toxic_pred
)

get_layer(toxic_model , name = "embed") %>% 
  set_weights(list(embedding_matrix)) %>% 
  freeze_weights()

# toxic_model_max75 <- load_model_hdf5(
#   filepath = "/home/kawal/D/jigsaw-toxic-comment-classification-challenge/toxic_model.hdf5")

toxic_model %>% 
  compile(
    optimizer = "adam" , 
    loss = "binary_crossentropy" ,
    metrics = c("acc")
  )

history_toxic <- toxic_model %>% 
  fit(
    x_train , 
    y_train[ , (col_no - 2) ] , 
    epochs = 25 , 
    batch_size = 512 ,
    validation_data = list(x_test , y_test[ , (col_no - 2)])
    # callbacks = callback_early_stopping(monitor = "val_loss" , patience = 3)
  )

save_model_hdf5( toxic_model , 
  filepath = "/home/kawal/D/jigsaw-toxic-comment-classification-challenge/toxic_model_max75.hdf5"
)

freeze_weights(toxic_model , from = "embed" , to = "conv3")

# Tidytext try for toxic                                              ----
train <- w[train_rows , ]

tidy_comments <- train[which(w[ , col_no] == 1) , ] %>% 
  select(c(id , comment_text)) %>% 
  mutate(comment_text = comment_text %>% tolower()) %>% 
  unnest_tokens(word , comment_text)


tidy_comments <- tidy_comments %>% 
  anti_join(stop_words) %>% 
  na.omit()

tidy_comments <- tidy_comments %>% 
  count(word , sort = T) %>%
  mutate(word = reorder( word , n )) 
  # %>% 
  # mutate(toxic_prob = n/nrow(tidy_comments)) 

non_toxic <- train[which(w[ , col_no] == 0) , ] %>% 
  select(c(id , comment_text)) %>% 
  mutate(comment_text = comment_text %>% tolower() %>% rm_white()) %>% 
  unnest_tokens(word , comment_text) %>% 
  anti_join(stop_words) %>% 
  count(word , sort = T) %>% 
  mutate(word = reorder(word , n))
  # %>% 
  # mutate(neg_toxic_prob = n / sum(n)) %>% 
  # mutate(neg_toxic_prob = ifelse(is.na(neg_toxic_prob) , 0 , neg_toxic_prob))
  
tidy_comments <- tidy_comments %>% 
  left_join(non_toxic , by = "word") %>% 
  mutate(n.y = ifelse(is.na(n.y) , 0 , n.y)) %>% 
  mutate(bayes_prob = (n.x / (n.x + n.y )))

tidy_comments <- tidy_comments %>% 
  mutate(bayes_prob = ifelse(n.x < 6 , 1 / nrow(tidy_comments) , bayes_prob ))

prediction <- function(train , tidy_comments = tidy_comments  , 
                       toxic_model = toxic_model , col_no = 3)
      {
      pred <- train %>% 
        mutate(comment_text = comment_text %>%  as.character() %>% tolower()) %>% 
        unnest_tokens(word , comment_text) %>% 
        anti_join(stop_words) %>% 
        left_join(tidy_comments) %>% 
        na.omit() %>% 
        mutate(bayes_prob = ifelse(is.na(bayes_prob) , 0 , bayes_prob)) %>% 
        group_by(id) %>% 
        summarise(n_words = n() , finalprob = sum(bayes_prob) / n_words ) %>% 
        right_join(train , by = "id") %>% 
        mutate(finalprob = ifelse(is.na(finalprob ) , 
                                  (sum(train[ , col_no]) / nrow(train)) , 
                                  finalprob))
      
      cat("Bayes_LogLoss")
      print(LogLoss(pred$finalprob , pred[ , (col_no + 2) ]))
      
      train_nn <- train %>%
        mutate(comment_text = as.character(comment_text) %>% tolower()) %>%
        mutate(toxic_prob = toxic_model %>%
                 predict(texts_to_sequences(tokenizer , comment_text ) %>%
                           pad_sequences(maxlen = maxlen)) %>%
                 as.vector())
      
      pred <- pred %>%
        left_join(train_nn %>% select(id , toxic_prob) )
      
      cat("NN_LogLoss")
      print(LogLoss(pred$toxic_prob , pred[ , (col_no + 2) ]))
      
      cat("Overall_LogLoss")
      print(LogLoss((.85*pred$toxic_prob + .15 * pred$finalprob ) , pred[ , (col_no + 2) ]))
}

prediction(train , tidy_comments , toxic_model , col_no = col_no)

# }
column_prediction(w , text , tokenizer , col_no = 6)