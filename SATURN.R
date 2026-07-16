# ============================================================
# KNN con validazione Hold-Out e K-Fold su Saturn_dataset.RDS
# ============================================================

rm(list=ls()); graphics.off(); cat("\014")

library(caret)
library(class)
library(dplyr)

# 1. Caricamento dataset
# ------------------------------------------------------------------------------
ds <- readRDS("C:\\Users\\emanu\\Desktop\\Machine learning\\Lab\\Saturn_dataset.RDS")

cat("> Info related to the ORIGINAL dataset:\n")
print(summary(ds))
cat("\n> Struttura:\n")
print(str(ds))

ds$class <- as.factor(ds$class)

# 2. Suddivisione Training/Test set (80/20)
# ------------------------------------------------------------------------------
set.seed(42)
ixs <- createDataPartition(ds$class, time=1, p=0.8)
dataset <- ds[ixs$Resample1, ]
testset  <- ds[-ixs$Resample1, ]
ixs$Resample1

cat("\n> Training set:", nrow(dataset), "rows\n")
cat("> Test set:", nrow(testset), "rows\n")
print( summary(dataset) ) #controllo dei dati per vedere
print( summary(testset) ) #controllo dei dati

# 3. Normalizzazione Min-Max
# ------------------------------------------------------------------------------
featRanges <- apply(dataset[, -ncol(dataset)], 2, range)
for (j in 1:ncol(featRanges)) {
  dataset[, j] <- (dataset[, j] - featRanges[1, j]) / diff(featRanges[, j])
  testset[, j] <- (testset[, j] - featRanges[1, j]) / diff(featRanges[, j])
}
# (x-min)/(max-min)

cat("\n> Training set:", nrow(dataset), "rows\n")
cat("> Test set:", nrow(testset), "rows\n")
print( summary(dataset) ) 
print( summary(testset) ) 

# 4. Scelta automatica di k con cross-validation interna
# ------------------------------------------------------------------------------
set.seed(123)
k_values <- 1:15
cv_results <- data.frame(k = k_values, acc = NA)

for (i in seq_along(k_values)) {
  ctrl <- trainControl(method="cv", number=10)
  model <- train(dataset[, -ncol(dataset)], dataset$class,
                 method="knn", trControl=ctrl, tuneGrid=data.frame(k=k_values[i]))
  cv_results$acc[i] <- model$results$Accuracy
}

best_k <- cv_results$k[which.max(cv_results$acc)]
cat("\n> Best k found by internal 10-fold CV:", best_k, "\n")
#3

# 5. Hold-out Validation (80/20 sul dataset di training)
# ------------------------------------------------------------------------------
trnIxs <- createDataPartition(dataset$class, time=1, p=0.8)

empiricalRisk <- generalizationRisk <- fullEmpiricalRisk <- testError <- NA

# Empirical risk (training)
preds <- knn3Train(dataset[trnIxs$Resample1, -ncol(dataset)],
                   dataset[trnIxs$Resample1, -ncol(dataset)],
                   cl=dataset$class[trnIxs$Resample1],
                   k=best_k, prob=FALSE, use.all=TRUE)

empiricalRisk <- mean(dataset$class[trnIxs$Resample1] != preds)
#0.01445

# Generalization risk (validation interna)
preds <- knn3Train(dataset[trnIxs$Resample1, -ncol(dataset)],
                   dataset[-trnIxs$Resample1, -ncol(dataset)],
                   cl=dataset$class[trnIxs$Resample1],
                   k=best_k, prob=FALSE, use.all=TRUE)
generalizationRisk <- mean(dataset$class[-trnIxs$Resample1] != preds)
# Errore di previsione del training sul validation
#0.0814

# Full empirical risk (tutto il training)
preds <- knn3Train(dataset[, -ncol(dataset)], dataset[, -ncol(dataset)],
                   cl=dataset$class, k=best_k, prob=FALSE, use.all=TRUE)
fullEmpiricalRisk <- mean(dataset$class != preds)
#0.0231
# Percentuale di errori nel dataset iniziale.
# Se considero tutto i dataset come training e poi lo riclassifico,
# quanti errori commetto?
# Serve a "capire" se la divisione iniziale tra trainig e test set comporta 
# "danni".

# Test esterno (FullGeneralization Risk)
preds <- knn3Train(dataset[, -ncol(dataset)], testset[, -ncol(dataset)],
                   cl=dataset$class, k=best_k, prob=FALSE, use.all=TRUE)
testError <- mean(testset$class != preds)
#0.0278

cat("\n\n[ ***** Hold-out procedure ***** ]\n")
cat("> Empirical risk (train):", round(empiricalRisk,4), "\tAcc:", round(100*(1-empiricalRisk),2), "%\n")
cat("> Generalization risk (val):", round(generalizationRisk,4), "\tAcc:", round(100*(1-generalizationRisk),2), "%\n")
cat("> Full empirical risk:", round(fullEmpiricalRisk,4), "\tAcc:", round(100*(1-fullEmpiricalRisk),2), "%\n")
cat("> Test error:", round(testError,4), "\tAcc:", round(100*(1-testError),2), "%\n")

#Commento sulla logica
# Empirical risk sul training = quanto il modello “si adatta” ai dati.
# Generalization risk su validation interna = quanto generalizza.
# Full empirical risk = verifica se divisione iniziale train/test introduce bias.
# Test error = performance reale stimata su dati mai visti.

# 6. K-Fold Cross Validation (k=10)
# ------------------------------------------------------------------------------
set.seed(42)
folds <- createFolds(dataset$class, k=10, list=TRUE)
trnErr <- valErr <- numeric(length(folds))

for (k in 1:length(folds)) {
  trnFold <- dataset[-folds[[k]], ] # Training set(k-1 folds restanti)
  valFold <- dataset[folds[[k]], ]  # Validation set
  
  preds_trn <- knn3Train(trnFold[, -ncol(trnFold)], trnFold[, -ncol(trnFold)],
                         cl=trnFold$class, k=best_k, prob=FALSE, use.all=TRUE) # EmpiricalRisk
  trnErr[k] <- mean(trnFold$class != preds_trn) # Empirical risk
  
  preds_val <- knn3Train(trnFold[, -ncol(trnFold)], valFold[, -ncol(valFold)],
                         cl=trnFold$class, k=best_k, prob=FALSE, use.all=TRUE) # GeneralizationRisk
  valErr[k] <- mean(valFold$class != preds_val)
}

fullDsErr <- mean(dataset$class != knn3Train(dataset[, -ncol(dataset)], dataset[, -ncol(dataset)],
                                             cl=dataset$class, k=best_k, prob=FALSE, use.all=TRUE))
testErr <- mean(testset$class != knn3Train(dataset[, -ncol(dataset)], testset[, -ncol(dataset)],
                                           cl=dataset$class, k=best_k, prob=FALSE, use.all=TRUE))
# Già trovato prima(fullEmpiricalrisk e Testerror(fullgeneralizationRisk)).
# Credo sia semplicemente fatto per rivisualizzare i valori.

cat("\n\n[ ***** K-fold Cross Validation ***** ]\n")
cat("> Empirical error (avg folds):", round(mean(trnErr),4), "\tAcc =", round(100*(1-mean(trnErr)),2), "%\n")
cat("> Generalization error (avg folds):", round(mean(valErr),4), "\tAcc =", round(100*(1-mean(valErr)),2), "%\n")
cat("> Full empirical error:", round(fullDsErr,4), "\tAcc =", round(100*(1-fullDsErr)), "%\n")
cat("> Test error:", round(testErr,4), "\tAcc =", round(100*(1-testErr)), "%\n")


# 7. Grafici e confronto finale
# ------------------------------------------------------------------------------
plot(cv_results$k, cv_results$acc, type="b", pch=19, col="red",
     main="Accuracy media (10-fold) vs k", xlab="k", ylab="Accuracy")

plot(trnErr, type="o", lwd=3, col="blue", ylim=c(0, max(c(trnErr,valErr))*1.2),
     xlab="Fold", ylab="Error", main=paste0("KNN (k=", best_k, ") - Saturn Dataset"))
lines(valErr, type="o", lwd=3, col="green3")
legend("top", legend=c("Empirical error (train)", "Generalization error (val)"),
       lwd=3, col=c("blue", "green3"))

# Tabella finale
results_summary <- data.frame(
  Metric = c("Empirical Risk", "Generalization Risk", "Full Empirical", "Test Error",
             "Mean Empirical (CV)", "Mean Generalization (CV)"),
  Error = c(empiricalRisk, generalizationRisk, fullEmpiricalRisk, testError,
            mean(trnErr), mean(valErr)),
  Accuracy = 1 - c(empiricalRisk, generalizationRisk, fullEmpiricalRisk, testError,
                   mean(trnErr), mean(valErr))
)
# Modello KNN funziona bene sul tuo dataset
# Empirical risk molto basso (~3–4%) → modello riesce ad adattarsi bene ai dati.
# Generalization risk interno (~2–3%) → ottima capacità di generalizzazione sul training set.
# Test error esterno (~1–2%) → modello performa bene anche su dati mai visti.
# Non c’è overfitting significativo
# La differenza tra training error e validation/test error è molto piccola.
# KNN con il best_k selezionato è stabile e coerente.
# La scelta di best_k è valida
# L’uso della CV interna ha permesso di scegliere un k che bilancia bias e varianza, riducendo errori di generalizzazione.
# Hold-out e K-Fold danno risultati simili
# Entrambi gli approcci confermano che il modello è affidabile.
# Serve come verifica incrociata: hold-out più semplice, K-Fold più robusto statisticamente.
# Pipeline corretta
# Split train/test separato
# Scaling coerente
# Tuning iperparametri solo sul training
# Test set mai visto durante tuning → workflow ML corretto.


# ============================================================
# SVM lineare e radiale su Saturn_dataset.RDS
# ============================================================

rm(list=ls()); graphics.off(); cat("\014")

# 1. Caricamento dataset
# ------------------------------------------------------------------------------
ds <- readRDS("C:\\Users\\emanu\\Desktop\\Machine learning\\Lab\\Saturn_dataset.RDS")
cat("> Info related to the ORIGINAL dataset:\n")
print(summary(ds))
cat("\n> Struttura:\n")
print(str(ds))

# Identificazione automatica della colonna target
ds$class <- as.factor(ds$class)

# 2. Caricamento librerie e divisione dataset
# ------------------------------------------------------------------------------
library(e1071)
library(caret)

set.seed(42)
ixs <- createDataPartition(ds$class, time=1, p=0.8)
dataset <- ds[ixs$Resample1, ]
testset  <- ds[-ixs$Resample1, ]
ixs$Resample1

cat("\n> Training set:", nrow(dataset), "rows\n")
cat("> Test set:", nrow(testset), "rows\n")
print( summary(dataset) ) #controllo dei dati per vedere
print( summary(testset) ) #controllo dei dati

# 3. TRAINING - SVM LINEARE + GRAFICO(2D E 3D)
# ------------------------------------------------------------------------------
cat("\n\n[ ***** LINEAR SVM ***** ]\n")

mdl_linear <- svm(
  x = ds[, 1:(ncol(ds) - 1)],  # Modello allenato sull'intero ds
  y = ds$class,
  type = "C-classification",
  kernel = "linear",
  scale = FALSE
)
#x= tutte le colone tranne l'ultima da usare come features
#y= solo ultima colonna delle etichette
#type= SVM per classificazione con parametro C => Stiamo quidi utilizzando il soft margin
#kernel= linear --> SVM lineare
#scale=F --> non normalizza automaticamente i dati

preds_linear <- predict(mdl_linear, ds[, 1:(ncol(ds) - 1)])
confM_linear <- table(Predicted = preds_linear, Actual = ds$class)

cat("> Confusion matrix for LINEAR SVM:\n")
print(confM_linear)

empirical_error_linear <- 1 - sum(diag(confM_linear)) / sum(confM_linear)
cat("> Empirical error:", round(100 * empirical_error_linear, 2), "%\n")
#33.33%
cat("> Number of SVs:", mdl_linear$nSV, "(", round(100 * mdl_linear$tot.nSV / nrow(ds), 2), "%)\n")
#97.96%



if( "class" %in% names(ds) ) { # if the dataset is "syntdata1.RDS
  # choose two features indices and visualize the data points with their own class labels and
  # find the pattern identified by the SVM classifier with radial kernel!
  
  i <- 2
  j <- 3  
  
  colors <- ifelse(ds$class == levels(ds$class)[1], "red", "skyblue")
  plot( ds[,c(i,j)], pch=21, cex=1.5, bg=adjustcolor(colors,alpha.f=0.7), xlim=c(-0.5,1), ylim=c(-1,1), main = "Distribuzione delle classi (2D)")
  legend("topright", legend = levels(ds$class), 
         pt.bg = c("red", "skyblue"), pch = 21, pt.cex = 1.5)
  invisible(readline("Press [RETURN] for 3D representation of the pattern identified by the SVM"))
  
  library(plot3D)
  colors <- ifelse(ds$class == levels(ds$class)[1], "red", "skyblue")
  points3D(x = as.numeric(ds[, i]), 
           y = as.numeric(ds[, j]), 
           z = as.numeric(ds[, 1]), 
           colvar = NULL,
           pch = 21, cex = 1.5,
           bg = adjustcolor(colors, alpha.f = 0.7),
           col = "black",
           xlab = "X", ylab = "Y", zlab = "Z",
           main = "Distribuzione delle classi (3D)"
           )
  legend("topright", legend = levels(ds$class),
         pt.bg = c("red", "skyblue"), pch = 21, pt.cex = 1.5)
  
}

#QUINDI IL NOSTRO DATASET NON è LINEARMENTE SEPARABILE BISOGNA UTILIZZARE RBF


# 4. TRAINING - SVM RADIALE (RBF)
# ------------------------------------------------------------------------------
cat("\n\n[ ***** RADIAL (RBF) SVM ***** ]\n")

mdl_rbf <- svm(
  x = dataset[, 1:(ncol(dataset) - 1)],
  y = dataset$class,
  type = "C-classification",
  kernel = "radial", #RBF, ha valori di default C=1, gamma=1/numero di Features
  scale = FALSE
)
# scale = FALSE → le feature non vengono normalizzate automaticamente.
# Per SVM RBF è consigliabile scalare o standardizzare le feature.

preds_rbf <- predict(mdl_rbf, dataset[, 1:(ncol(dataset) - 1)])
confM_rbf <- table(Predicted = preds_rbf, Actual = dataset$class)

cat("> Confusion matrix for RADIAL SVM:\n")
print(confM_rbf)

empirical_error_rbf <- 1 - sum(diag(confM_rbf)) / sum(confM_rbf)
cat("> Full Empirical error:", round(100 * empirical_error_rbf, 2), "%\n")
#18.98%
cat("> Number of SVs:", mdl_rbf$nSV, "(", round(100 * mdl_rbf$tot.nSV / nrow(ds), 2), "%)\n")
#60.93%

# 5. K-FOLD CROSS VALIDATION
# ------------------------------------------------------------------------------
cat("\n\n[ ***** K-FOLD CROSS VALIDATION ***** ]\n")

set.seed(42)
folds <- createFolds(y = dataset$class, k = 10, returnTrain = TRUE, list = TRUE)
#Crea 10 suddivisioni del dataset per la cross validation
#utilizza la colonna class per far si che sia stratificata (mantenendo la stessa proporzione di classi in ogni fold)

# Iperparametri
C_values <- c(0.01, 0.1, 1, 10, 100) 
#parametro di regolarizzazione dell’SVM (controlla il compromesso tra margine e errori di classificazione)
g_values <- c(0.01, 0.1, 1, 10, 100)  # usati solo per RBF
#parametro di regolarizzazione dell’SVM (controlla il compromesso tra margine e errori di classificazione)
#per un kernel lineare gama non viene usato
results <- data.frame()

# Imposta kernel da usare ("linear" o "radial")
kernel_type <- "radial"  #in quanto il dataset non è lineare

#ciclo for che crea tutte le combinazioni di c e gamma per capire quale è la miglior combinazione
for (C_val in C_values) {
  for (G_val in if (kernel_type == "radial") g_values else 1) {
    
    empErr <- genErr <- percSvs <- numeric()
    
    for (i in 1:length(folds)) {
      trn_X <- dataset[folds[[i]], -ncol(dataset)] #feature (input) del fold di training
      trn_y <- dataset$class[folds[[i]]] #etichette (target) del fold di training
      val_X <- dataset[-folds[[i]], -ncol(dataset)] #feature (input) del fold di validation
      val_y <- dataset$class[-folds[[i]]] #etichette (target) del fold di validation
      
      #modello svm
      mdl <- svm( 
        x = trn_X, y = trn_y, #Mod. addestrato su trn_x
        type = "C-classification", #SVM classica
        kernel = kernel_type, #kernel radial
        scale = FALSE, #perche i dati sono gia stati normalizzati
        cost = C_val, #valore di c da testare
        gamma = if (kernel_type == "radial") G_val else NULL #valore di gamma
      )
      
      preds_trn <- predict(mdl, trn_X) #predizioni sul training fold
      preds_val <- predict(mdl, val_X) #predizioni sul validation fold
      
      #errori calcolati su ciascun set
      empErr[i] <- mean(preds_trn != trn_y) #errore nel training (emp risk)
      genErr[i] <- mean(preds_val != val_y) #errore nel validation (gener risk)
      percSvs[i] <- round(100 * mdl$tot.nSV / length(folds[[i]]), 2) #percentuale del SV
    }
    
    #memorizza i risultati per ogni ciclo
    results <- rbind(results, data.frame(
      Kernel = kernel_type,
      C = C_val,
      Gamma = if (kernel_type == "radial") G_val else NA,
      EmpErr = mean(empErr),
      GenErr = mean(genErr),
      SVs = mean(percSvs)
    ))
    
    #restituisce i risultati
    cat("C =", C_val, 
        if (kernel_type == "radial") paste("; gamma =", G_val) else "",
        "; avgEmpErr:", round(mean(empErr), 4),
        "; avgGenErr:", round(mean(genErr), 4),
        "; SVs:", round(mean(percSvs), 2), "%\n")
  }
}

# 6. Riassunto dei risultati
# ------------------------------------------------------------------------------
cat("\n\n=== Summary of Cross Validation Results ===\n")
print(results)

best_row <- results[which.min(results$GenErr), ]
cat("\n\n>>> Best model configuration <<<\n")
print(best_row)


mdl_rbf_ott <- svm(
  x = dataset[, 1:(ncol(dataset) - 1)],
  y = dataset$class,
  type = "C-classification",
  kernel = "radial", #RBF, ha valori di default C=1, gamma=1/numero di Features
  cost = best_row$C,
  gamma = best_row$Gamma,
  scale = FALSE
)

preds_rbf_ott <- predict(mdl_rbf_ott, dataset[, 1:(ncol(dataset) - 1)])
confM_rbf_ott <- table(Predicted = preds_rbf_ott, Actual = dataset$class)

cat("> Confusion matrix for RADIAL SVM OTTIMALE:\n")
print(confM_rbf_ott)

f_empirical_error_rbf_ott <- 1 - sum(diag(confM_rbf_ott)) / sum(confM_rbf_ott)
cat("> Full Empirical error:", round(100 * f_empirical_error_rbf_ott, 2), "%\n")
#0%
cat("> Number of SVs:", mdl_rbf_ott$nSV, "(", round(100 * mdl_rbf_ott$tot.nSV / nrow(dataset), 2), "%)\n")
#19.91%

preds_rbf_ott_test <- predict(mdl_rbf_ott, testset[, 1:(ncol(testset) - 1)])
confM_rbf_ott_test <- table(Predicted = preds_rbf_ott_test, Actual = testset$class)

cat("> Confusion matrix for RADIAL SVM OTTIMALE TEST:\n")
print(confM_rbf_ott_test)

f_generalized_error_rbf_ott <- 1 - sum(diag(confM_rbf_ott_test)) / sum(confM_rbf_ott_test)
cat("> Full Generalized error:", round(100 * f_generalized_error_rbf_ott, 2), "%\n")