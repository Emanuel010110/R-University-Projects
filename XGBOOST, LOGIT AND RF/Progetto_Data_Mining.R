
con_train_1 <- read.csv("C:\\Users\\emanu\\Desktop\\Data_Mining\\Progetto\\training.csv")
con_test_1 <- read.csv("C:\\Users\\emanu\\Desktop\\Data_Mining\\Progetto\\test.csv")


########## 1) PREPROCESSING

### 1.1) TRASFORMAZIONE VARIABILI E ANALISI

# MISSING VALUES
sum(is.na(con_train_1)) # Non ci sono missing values
#con_train <- na.omit(con_train_1)
sum(is.na(con_test_1)) # Non ci sono missing values
#con_test <- na.omit(con_test_1)

# TOLGO ID
con_train <- subset(
  con_train_1,
  select = -ID)
con_test <- subset(
  con_test_1,
  select = -ID)

# ANALISI TIPO VARIABILI
str(con_train)
str(con_test)
summary(con_train)
summary(con_test)

# TRASFORMO LE VARIABILI CARATTERIALI IN FACTOR
qualitative_vars <- c("state","education","child","religion","area","contraceptive")
con_train[qualitative_vars] <- lapply(con_train[qualitative_vars], as.factor)
str(con_train)
summary(con_train)

qualitative_vars_2 <- c("state","education","child","religion","area")
con_test[qualitative_vars_2] <- lapply(con_test[qualitative_vars_2], as.factor)
str(con_test)
summary(con_test)

# TRASFORMO I FACTOR CHE SI POSSONO ORDINARE
con_train$education <- ordered(con_train$education,
                 levels = c("None","Low","Intermediate","High"))
con_train$child <- ordered(con_train$child,
                 levels = c("0","1","More than 1"))
str(con_train)
summary(con_train)

con_test$education <- ordered(con_test$education,
                               levels = c("None","Low","Intermediate","High"))
con_test$child <- ordered(con_test$child,
                           levels = c("0","1","More than 1"))
str(con_test)
summary(con_test)



### 1.2) DISTRIBUZIONI

library(ggplot2)
library(patchwork)

theme_set(theme_minimal(base_size = 11) + 
            theme(plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
                  axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)))
# Target
g_target <- ggplot(con_train, aes(x = contraceptive, fill = contraceptive)) +
  geom_bar(width = 0.6, show.legend = FALSE) +
  scale_fill_manual(values = c("#E69F00", "#56B4E9")) + 
  labs(title = "Target Variable Distribution", x = "Contraceptive Use", y = "Frequency")

# State
g_state <- ggplot(con_train, aes(x = state)) +
  geom_bar(fill = "steelblue", width = 0.7) +
  labs(title = "State Distribution", x = "State", y = "Frequency") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8)) 

# Age
g_age <- ggplot(con_train, aes(x = age)) +
  geom_histogram(binwidth = 2, fill = "#009E73", color = "white", alpha = 0.8) +
  labs(title = "Age Histogram", x = "Age", y = "Frequency")

# Children
g_child <- ggplot(con_train, aes(x = child, fill = child)) +
  geom_bar(width = 0.6, show.legend = FALSE) +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "Number of children", x = "Child Classes", y = "Frequency")

# Education
g_edu <- ggplot(con_train, aes(x = education, fill = education)) +
  geom_bar(width = 0.6, show.legend = FALSE) +
  scale_fill_brewer(palette = "Purples") +
  labs(title = "Level of Education", x = "Education", y = "Frequency")

(g_target + g_age) / (g_edu + g_child)
g_state



# 1.3)
# VEDO SE I LIVELLI PER IL TRAINING E IL TEST SONO GLI STESSI
# ALTRIMENTI NON VA BENE PER RF, BOOSTING
levels(con_train$education)
levels(con_test$education) #OK
levels(con_train$child)
levels(con_test$child) #OK
levels(con_train$religion)
levels(con_test$religion) #OK
levels(con_train$area)
levels(con_test$area) #OK
levels(con_train$state)
levels(con_test$state) #OK



########## 2) IMPLEMENTAZIONE MODELLI

### 2.1)
# DIVIDO ULTERIORMENTE IL TRAINING
set.seed(123)
train_index <- sample(1:nrow(con_train), 0.8*nrow(con_train))
train_set <- con_train[train_index, ]
val_set   <- con_train[-train_index, ]

# LOG-LOSS FUNCTION
log_loss <- function(y_true, y_pred) {
  eps <- 1e-5
  y_pred <- pmax(pmin(y_pred, 1 - eps), eps)
  -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
}



### 2.2) LOGISTIC REGRESSION
model_logit <- glm(contraceptive ~ ., 
                   data = train_set, 
                   family = binomial)
pred_logit <- predict(model_logit, val_set, type = "response") # MSE(train), non utile

y_true <- as.numeric(as.character(val_set$contraceptive))
log_loss(y_true, pred_logit)
# 0.4612508



### 2.3) SUBMISSION LOGIT
# Predizione sul test set
pred_test_logit <- predict(model_logit, newdata = con_test, type = "response")
# Applicazione clipping
eps <- 1e-5
pred_test_logit_clipped <- pmax(pmin(pred_test_logit, 1 - eps), eps)
# Creazione del dataframe per Kaggle
sub_logit <- data.frame(
  ID = con_test_1$ID,
  contraceptive = pred_test_logit_clipped
)
# Salvataggio CSV
write.csv(sub_logit, "submission_logit_TAFANI.csv", row.names = FALSE)



### 2.4) RANDOM FOREST
library(ranger)
set.seed(123)
p <- ncol(con_train) - 1 #Numero di predittori
mtry_con <- floor(sqrt(p))

model_ranger <- ranger(
  contraceptive ~ ., 
  data = train_set,
  num.trees = 1000,
  mtry = mtry_con,
  probability = TRUE,      # T Per la log-loss
  importance = "impurity"  
)

pred_ranger_res <- predict(model_ranger, data = val_set)
pred_ranger <- pred_ranger_res$predictions[, "1"] # Estraiamo la probabilità di 1

y_val <- as.numeric(as.character(val_set$contraceptive))
log_loss(y_val, pred_ranger)
# 0.4570855



### 2.5) SUBMISSION RF
pred_test_ranger_res <- predict(model_ranger, data = con_test)
# Probabilità 1
pred_test_ranger <- pred_test_ranger_res$predictions[, "1"]
# Clipping
eps <- 1e-5
pred_test_ranger_clipped <- pmax(pmin(pred_test_ranger, 1 - eps), eps)

sub_ranger <- data.frame(
  ID = con_test_1$ID,
  contraceptive = pred_test_ranger_clipped
)

write.csv(sub_ranger, "submission_RF_TAFANI.csv", row.names = FALSE)



### 2.6) XGBOOST
library(xgboost)

# TRASFORMO IL FACTOR IN NUMERICO
y_train <- as.numeric(as.character(train_set$contraceptive))
y_val   <- as.numeric(as.character(val_set$contraceptive))

# ONE-HOT ENCODING DEI PREDITTORI
# Funzione model.matrix trasforma i Factor in colonne di 0 e 1
# Target escluso

# PREPARO ESCLUDENDO L'INTERCETTA 
X_train_matrix <- model.matrix(contraceptive ~ . - 1, data = train_set)
X_val_matrix   <- model.matrix(contraceptive ~ . - 1, data = val_set)

# CREAZIONE DELLE DMATRIX PER XGBoost
dtrain <- xgb.DMatrix(data = X_train_matrix, label = y_train)
dval   <- xgb.DMatrix(data = X_val_matrix, label = y_val)

# LISTA PARAMETRI
params <- list(
  objective = "binary:logistic", # Classificazione binaria
  eval_metric = "logloss",       # Monitoro la log-loss
  max_depth = 6,                 # Profondità massima degli alberi
  eta = 0.05,                    # Learning rate
  subsample = 0.8,               # Usa l'80% dei dati per ogni albero
  colsample_bytree = 0.8         # Usa l'80% delle colonne per ogni albero
)

set.seed(123)
model_xgb <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,               
  watchlist = list(train = dtrain, val = dval), # Monitora l'errore in tempo reale
  early_stopping_rounds = 50,    # Blocco se la log-loss sul val non migliora per 50 alberi
  print_every_n = 50            
)

# PREVISIONE SUL VALIDATION SET
pred_xgb <- predict(model_xgb, dval)
log_loss(y_val, pred_xgb)
# 0.4485216

### 2.7) SUBMISSION
# ONE HOT ENCODIC SUL TEST SET
X_test_matrix <- model.matrix(~ . - 1, data = con_test)
dtest <- xgb.DMatrix(data = X_test_matrix)

pred_test_xgb1 <- predict(model_xgb, dtest)

# Clipping
eps <- 1e-5
pred_test_xgb1_clipped <- pmax(pmin(pred_test_xgb1, 1 - eps), eps)

sub_xgb1 <- data.frame(
  ID = con_test_1$ID,
  contraceptive = pred_test_xgb1_clipped
)

write.csv(sub_xgb1, "submission_xgb1_TAFANI.csv", row.names = FALSE)



### 2.8) XGBOOST (TENTATIVO 2)
# Parametri Ottimizzati
params_opt <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 4,                 # Da 6 a 4 per ridurre l'overfitting
  eta = 0.01,                    # Minore per maggiore precisione
  subsample = 0.8,
  colsample_bytree = 0.8
)

set.seed(123)
model_xgb_opt <- xgb.train(
  params = params_opt,
  data = dtrain,
  nrounds = 2000,                # Alzo visto l'abbassamento del learning rate
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 100,   
  print_every_n = 100
)

pred_xgb_opt <- predict(model_xgb_opt, dval)
log_loss(y_val, pred_xgb_opt)
# 0.448



# 2.9) SUBMISSION
X_test_matrix <- model.matrix(~ . - 1, data = con_test)
dtest <- xgb.DMatrix(data = X_test_matrix)

pred_test_xgb <- predict(model_xgb_opt, dtest)
# Clipping
eps <- 1e-5
pred_test_xgb_clipped <- pmax(pmin(pred_test_xgb, 1 - eps), eps)

sub_xgb <- data.frame(
  ID = con_test_1$ID,
  contraceptive = pred_test_xgb_clipped
)

write.csv(sub_xgb, "submission_FIN_xgb_TAFANI_EMANUEL.csv", row.names = FALSE)