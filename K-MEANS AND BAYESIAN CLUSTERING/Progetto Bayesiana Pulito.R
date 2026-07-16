###### 1) LIBRERIE ######-------------------------------------------------------
library(corrplot)

library(dplyr)

library(rstan)
library(coda)
library(ggplot2)
library(ggpubr)
library(bayestestR)

library(ggplot2)
library(scatterplot3d)
library(plotly)

###### 2.1) DATASET E PREPROCESSING ######----------------------------------------
spotify <- read.csv2("C:\\Users\\emanu\\Desktop\\Progetto Bayesiana\\spotify_songs.csv")

spotify <- subset(
  spotify, 
  select = -c(track_id, track_album_id, track_album_name, track_album_release_date, playlist_name, playlist_id)
)
# Abbiamo eliminato le variabili identificative o descrittive non utili ai fini 
# dell’analisi, perché non contengono informazione strutturale rilevante per il 
# clustering o per la modellizzazione della popularity.

qualitative_vars <- c("danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo")
spotify[qualitative_vars] <- lapply(spotify[qualitative_vars], as.numeric)
str(spotify)
# Abbiamo convertito le feature audio in formato numerico, perché il clustering 
# richiede variabili quantitative per il calcolo delle distanze e la regressione 
# richiede covariate correttamente tipizzate.

factor_vars <- c("playlist_genre", "playlist_subgenre")
spotify[factor_vars] <- lapply(spotify[factor_vars], as.factor)
str(spotify)
# Le variabili di genere e sottogenere non necessariamente entrano direttamente 
# nel clustering, ma possono essere usate per interpretare a posteriori i gruppi 
# individuati.

###-----###
sum(is.na(spotify)) 
spotify <- na.omit(spotify)
# Abbiamo verificato la presenza di valori mancanti e, per semplicità 
# metodologica, abbiamo rimosso le osservazioni incomplete. Questa scelta garantisce 
# coerenza nei modelli successivi, anche se può comportare perdita di 
# informazione e potenziale bias se i missing non sono completamente casuali.

str(spotify$playlist_genre)
summary(spotify$playlist_genre)
str(spotify$playlist_subgenre)
summary(spotify$playlist_subgenre)
str(spotify$track_popularity)
summary(spotify$track_popularity) 
# Min.  1st Qu.  Median   Mean  3rd Qu.   Max. 
# 0.00   24.00   45.00   42.49   62.00  100.00 
# Circa il 50% delle osservazioni ha popolarità <45, le altre >45 => dataset 
# piuttosto bilanciato sulla variabile popularity.
# Abbiamo effettuato una prima analisi descrittiva delle variabili categoriche 
# e della popularity, per verificarne distribuzione, supporto e possibili 
# criticità in vista della modellizzazione successiva

###-----###
par(mfrow = c(3,3))

for(i in names(spotify)){ #Scorre tutti i nomi delle colonne del dataset spotify
  
  if(is.numeric(spotify[[i]])){ #Controlla se la variabile corrente è numerica
    
    hist(spotify[[i]],
         main = paste("Histogram of", i),
         xlab = i)
    
  } else if(is.factor(spotify[[i]])){ #Controlla se la variabile corrente è factor
    
    barplot(table(spotify[[i]]),
            main = paste("Barplot of", i),
            las = 2)
  }
}
# Per le variabili quantitative abbiamo usato gli istogrammi, in modo da analizzare 
# la distribuzione marginale di ciascuna feature e identificare eventuali 
# asimmetrie, concentrazioni o comportamenti anomali.
# Per le variabili categoriche abbiamo rappresentato le frequenze mediante 
# barplot, così da verificare il bilanciamento dei livelli e il peso relativo 
# di ciascuna categoria.

###-----###
par(mfrow = c(1,1))
cor_matrix <- cor(spotify[sapply(spotify, is.numeric)], use = "complete.obs") # Matrice di correlazione (solo variabili numeriche)

corrplot(cor_matrix, method = "color", type = "upper", 
         tl.cex = 0.7, tl.col = "black", addCoef.col = "black", 
         number.cex = 0.6)
# Abbiamo costruito la matrice di correlazione tra le variabili numeriche per 
# individuare associazioni lineari e possibili ridondanze informative. 
# Questo è rilevante sia per l’interpretazione dei dati sia per valutare 
# eventuali problemi di multicollinearità o dominanza di alcune dimensioni 
# nel clustering.

###-----###
par(mfrow = c(3,3))
colors <- rainbow(length(names(spotify))) # Palette colori
j <- 1

for(i in names(spotify)){
  
  if(is.numeric(spotify[[i]])){
    
    boxplot(spotify[[i]],
            main = paste("Boxplot of", i),
            col = colors[j],
            border = "black",
            horizontal = TRUE)
    
  } else if(is.factor(spotify[[i]])){
    
    boxplot(track_popularity ~ spotify[[i]],
            data = spotify,
            main = paste("Track Popularity by", i),
            col = rainbow(length(levels(spotify[[i]]))),
            las = 2)
  }
  
  j <- j + 1
}
par(mfrow = c(1,1))
# Per le variabili numeriche abbiamo usato i boxplot per riassumere la distribuzione 
# in termini di quartili, dispersione e outlier, che sono particolarmente 
# rilevanti per metodi sensibili alle distanze come il clustering.
# Per le variabili categoriche abbiamo confrontato la distribuzione della 
# track popularity tra i diversi livelli. Questo ci permette di verificare 
# se esistano differenze sistematiche nella popularity tra gruppi, 
# concetto che si collega poi direttamente all’uso dei cluster come 
# variabile esplicativa.

##### 2.2) INTERPRETAZIONE GRAFICI #####----------------------------------------
# La distribuzione della popularity è concentrata in un intervallo intermedio, 
# suggerendo la presenza di strutture latenti piuttosto che una distribuzione 
# uniforme.
# danceability, energy, valence: Distribuzioni abbastanza simmetriche o 
# leggermente skewed. Molte canzoni con valori medio-alti
# Interpretazione: Spotify tende a contenere musica “mainstream”, più ballabile, 
# energetica. Queste variabili saranno probabilmente molto influenti nei cluster.

# Variabili molto asimmetriche: speechiness, acousticness, instrumentalness, 
# liveness. Fortemente skewed a destra. Molti valori vicino a 0.
# Interpretazione: La maggior parte delle canzoni NON è parlata (speechiness),
# NON è acustica pura, NON è strumentale.
# Problema importante. Queste variabili: hanno distribuzione molto sbilanciata,
# possono influenzare male il clustering.

# MATRICE DI CORRELAZIONE: Questa è molto importante.
# Osservazioni chiave: Correlazioni forti
# energy – loudness: 0.68 Molto alta
# energy – acousticness: -0.54 Forte negativa
# Interpretazione: Canzoni energiche più rumorose
# Canzoni acustiche → meno energiche
# Correlazioni moderate
# danceability – valence: ~0.33 Musica più “felice” → più ballabile
# Correlazioni basse con track_popularity Quasi tutte vicino a 0
# QUESTO È IMPORTANTISSIMO. Significa: Nessuna singola variabile spiega bene la 
# popularity. Ma allora perché clustering? Perché: la popularity potrebbe dipendere 
# da combinazioni di variabili, non da una sola.

##### 2.3) PREPROCESSING PT. 2 #####--------------------------------------------
spotify_scaled <- spotify
spotify_scaled[sapply(spotify, is.numeric)] <- scale(spotify[sapply(spotify, is.numeric)])

par(mfrow = c(3,3))
for(i in names(spotify_scaled)){
  
  if(is.numeric(spotify_scaled[[i]])){
    
    hist(spotify_scaled[[i]],
         main = paste("Histogram of", i),
         xlab = i)
    
  } else if(is.factor(spotify_scaled[[i]])){
    
    barplot(table(spotify_scaled[[i]]),
            main = paste("Barplot of", i),
            las = 2)
  }
}

par(mfrow = c(1,1))
cor_matrix <- cor(spotify_scaled[sapply(spotify_scaled, is.numeric)], use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.cex = 0.7, tl.col = "black", addCoef.col = "black", 
         number.cex = 0.6)
# La standardizzazione non altera la struttura di correlazione tra le variabili, 
# ma solo la loro scala.

set.seed(123)  # per riproducibilità

spotify_sample <- spotify_scaled %>%
  group_by(playlist_genre) %>% # Stratificazione in base al genere
  slice_sample(prop = 0.05) %>%  # modifica la proporzione desiderata (es. 5%)
  ungroup()

table(spotify_sample$playlist_genre)  # Verifica proporzioni
prop.table(table(spotify_sample$playlist_genre))
# Abbiamo utilizzato un campionamento stratificato per playlist_genre, 
# al fine di mantenere la rappresentatività delle diverse categorie nel 
# campione ridotto.
# Lo stratified sampling è stato effettuato rispetto al genere, pur non 
# essendo la variabile direttamente utilizzata nel clustering, 
# per preservare la struttura globale del dataset.

##### 3) VARIABLE SELECTION #####-----------------------------------------------

# 1. Definiamo target (Y) e covariate (X) dal subsample
# Usiamo track_popularity (che è continua e scalata)
y_stan <- spotify_sample$track_popularity

# Recuperiamo le 9 variabili numeriche che abbiamo scalato
spotify_sample_numeric <- subset(
  spotify_sample, 
  select = -c(track_name, track_artist, track_popularity, playlist_genre, playlist_subgenre)
)
X_covariates <- as.matrix(spotify_sample_numeric)

# Aggiungiamo l'intercetta 
X_stan <- cbind(intercept = rep(1, length(y_stan)), X_covariates)

# 2. Calcolo degli Iperparametri (EQUILIBRATI)
c_val <- 50         # soglia per distinguere: coefficiente ≈ 0 → spike, coefficiente ≠ 0 → slab
                    # più grande = più selezione (più aggressivo)
kappa_val <- 0.03   # Abbastanza grande da filtrare il rumore, ma non troppo severo
tau_val <- kappa_val / sqrt(2 * log(c_val) * c_val^2 / (c_val^2 - 1))

# 3. lista dati per Stan
data_STAN <- list(
  n = nrow(X_stan), 
  p = ncol(X_stan),  
  y = y_stan, 
  X = X_stan, 
  beta0 = rep(0, ncol(X_stan)), 
  alpha1 = 1,       # non forziamo nessuna aspettativa a priori
  alpha2 = 1, 
  c2 = c_val^2, 
  tau2 = tau_val^2
)

# 4. Compilazione ed esecuzione del Modello Stan
linear_variable_selection <- stan(
  file = "C:\\Users\\emanu\\Desktop\\Progetto Bayesiana\\spotify_SpSl.stan", 
  data = data_STAN,
  chains = 3, 
  iter = 12000, 
  warmup = 2000, 
  seed = 123
)

# 5. Estrazione dei parametri e diagnostica MCMC
rstan::traceplot(linear_variable_selection, pars = "beta", inc_warmup = TRUE)
rstan::traceplot(linear_variable_selection, pars = "beta", inc_warmup = FALSE)
params_list <- As.mcmc.list(linear_variable_selection, pars = "beta")
summary(params_list)
geweke.diag(params_list)
ci(params_list, method = "HDI")
ci(params_list, method = "ETI")
params_beta <- as.mcmc(extract(linear_variable_selection, pars = c("beta"))[[1]])
# I traceplot mostrano buon mixing e assenza di trend, indicando convergenza delle catene.
# Le variabili rilevanti sono quelle il cui intervallo credibile non include lo zero.
# VARIABILI IMPORTANTI: beta[3] → [-0.32, -0.17] NON include 0
# beta[5]=>[0.12, 0.26] ON include 0, beta[13]=>[-0.16, -0.06] NON include 0. Queste sono significative
# Borderline, beta[9], beta[10]. Includono 0=>incerte.
# I Geweke z-scores sono generalmente accettabili e non indicano problemi di convergenza rilevanti.

# 6. Calcolo delle Probabilità di Inclusione (PIP)
# Trasformiamo in 1 (Slab) o 0 (Spike) a seconda della soglia kappa
gamma_matrix <- ifelse(abs(params_beta) > kappa_val, 1, 0)

# Criterio Median Probability Model (PIP > 0.40) #probabilità media per ogni variabile, ovvero 
#la probabilità dei gamma=1
MPM_model <- as.numeric(colMeans(gamma_matrix) > 0.40)

#Criterio HPD 
#seleziona la riga dei gamma per ogni variabile che nelle 12.000 iterazioni si
#è verificata piu volte. e colora di grigio le variabili corrispondenti al valore 0 e blu 
#le variabili corrispondenti al valore 1.
unique_model <- unique(gamma_matrix, MARGIN = 1)
freq <- apply(unique_model, 1, function(b)
  sum(apply(gamma_matrix, MARGIN = 1, function(a) all(a == b))))
HPD_model <- unique_model[which.max(freq), ]

#Criterio HS
HS_model  <- as.numeric(colMeans(gamma_matrix) == 1)

#VISUALIZZAZIONE GRAFICI
# -- P1: Highest Posterior Probability --
p1 <- ggplot(data.frame(value = colMeans(gamma_matrix), 
                        idx = 1:ncol(X_stan),
                        var = factor(colnames(X_stan), levels = colnames(X_stan)),
                        HPD_model_inc = factor(HPD_model))) + 
  geom_bar(aes(y = value, x = var, fill = HPD_model_inc), stat="identity", alpha = 0.7, col = "black") + 
  geom_hline(mapping = aes(yintercept = .40), col = "red", lty = 2, linewidth = 0.8) +
  scale_fill_manual(values = c("0" = "gray", "1" = "steelblue")) +
  coord_flip() + 
  theme_minimal() + 
  theme(legend.position="none") + 
  ylab("PIP") + 
  xlab("") + 
  ggtitle("Highest Posterior Probability")

# -- P2: Median Probability Model --
p2 <- ggplot(data.frame(value = colMeans(gamma_matrix), 
                        idx = 1:ncol(X_stan),
                        var = factor(colnames(X_stan), levels = colnames(X_stan)),
                        MPM_model_inc = factor(MPM_model))) + 
  geom_bar(aes(y = value, x = var, fill = MPM_model_inc), stat="identity", alpha = 0.7, col = "black") + 
  geom_hline(mapping = aes(yintercept = .40), col = "red", lty = 2, linewidth = 0.8) +
  scale_fill_manual(values = c("0" = "gray", "1" = "steelblue")) +
  coord_flip() + 
  theme_minimal() + 
  # Nascondo il testo sull'asse Y qui per evitare sovrapposizioni e fare spazio ai grafici!
  theme(legend.position="none", axis.text.y = element_blank()) + 
  ylab("PIP") + 
  xlab("") + 
  ggtitle("Median Probability")
plot(p2)

# -- P3: Hard Shrinkage --
p3 <- ggplot(data.frame(value = colMeans(gamma_matrix), 
                        idx = 1:ncol(X_stan),
                        var = factor(colnames(X_stan), levels = colnames(X_stan)),
                        HS_model_inc = factor(HS_model))) + 
  geom_bar(aes(y = value, x = var, fill = HS_model_inc), stat="identity", alpha = 0.7, col = "black") + 
  geom_hline(mapping = aes(yintercept = .5), col = "red", lty = 2, linewidth = 0.8) +
  scale_fill_manual(values = c("0" = "gray", "1" = "steelblue")) +
  coord_flip() + 
  theme_minimal() + 
  theme(legend.position="none", axis.text.y = element_blank()) + 
  ylab("PIP") + 
  xlab("") + 
  ggtitle("Hard Shrinkage")

#STAMPA FINALE AFFIANCATA
ggarrange(p1, p2, p3, nrow = 1)

spotify_optim <- subset(
  spotify_sample_numeric, 
  select = c(duration_ms, liveness, instrumentalness, loudness, energy)
)

##### 4.1) CLUSTERING - K-MEANS - ELBOW METHOD #####----------------------------
spotify_elbow <- subset(
  spotify_scaled, 
  select = c(duration_ms, liveness, instrumentalness, loudness, energy)
)

# Calcolo inertia (within-cluster sum of squares)
inertias <- c()
K_range <- 2:10

for (k in K_range) {
  model <- kmeans(spotify_elbow, centers = k, nstart = 25)
  inertias <- c(inertias, model$tot.withinss)
}
df <- data.frame(K = K_range, Inertia = inertias) # plot

ggplot(df, aes(x = K, y = Inertia)) +
  geom_line(color = "steelblue") +
  geom_point(size = 3, color = "steelblue") +
  labs(title = "Elbow Method", x = "Numero di cluster K", y = "Inertia (tot.withinss)") +
  theme_minimal()
# K=3

##### 4.2) K-MEANS - K=3 #####--------------------------------------------------
ds <- spotify_optim[, c("loudness", "energy")]
plot(ds,
     pch=19,
     cex=1.5,
     xlab="loudness",
     ylab="energy")
# Si può notare correlazione

set.seed(13437885)
km.raw <- kmeans(spotify_optim, 3, iter.max=100, nstart=25)
print(km.raw)

clrs <- c("deepskyblue","green3","red")
plot(ds,
     pch=19,
     cex=1.5,
     xlab="loudness",
     ylab="energy")
for (i in 1:3) {
  points(ds[km.raw$cluster==i, ],
         bg=adjustcolor(clrs[i], alpha.f=0.8),
         pch=21,
         cex=1.5)
}
# Sembra ci siano due cluster abbastanza separati, ma poi ci sono 
# punti assegnati ad un nuovo cluster sopra.
# Questo non ci dice molto.

# Proviamo a prendere un'altra coppia...
ds <- spotify_optim[, c("duration_ms", "energy")]

set.seed(13437885)
km.raw <- kmeans(spotify_optim, 3, iter.max=100, nstart=25)
print(km.raw)
clrs <- c("deepskyblue","green3","red")
plot(ds,
     pch=19,
     cex=1.5,
     xlab="duration_ms",
     ylab="energy")
for (i in 1:3) {
  points(ds[km.raw$cluster==i, ],
         bg=adjustcolor(clrs[i], alpha.f=0.8),
         pch=21,
         cex=1.5)
}
# Stessa cosa di prima...

# Cambiamo approccio, usiamo i PCA ad unico scopo illustrativo
pca <- prcomp(spotify_optim)
plot(pca$x[,1:2], col=km.raw$cluster, pch=19)
# Qui c'è separazione dei tre cluster!

# SCATTERPLOT DI TUTTE LE VARIABILI
pairs(
  spotify_optim,
  pch = 19,
  col = "darkblue",
  cex = 0.6,
  main = "Scatterplot matrix delle variabili Spotify"
)

pca <- prcomp(spotify_optim, center = TRUE, scale. = FALSE)
# Varianza spiegata
summary(pca)
# Matrice delle direzioni
pca$rotation
# Coordinate dei punti nello spazio PCA
head(pca$x)
# summary(pca) = quanto conta ogni componente
# Ho: Proportion of Variance
#      PC1 = 0.3608
#      PC2 = 0.2216
#      PC3 = 0.1877
#      PC4 = 0.1764
#      PC5 = 0.0536
# Traduzione semplice:
# PC1 spiega il 36% dell’informazione
# PC2 il 22%
# PC3 il 19%
# Insieme:
# PC1 + PC2 = 58%
# PC1 + PC2 + PC3 = 77%
# Cosa significa: il grafico 3D (PC1, PC2, PC3) rappresenta 
# circa il 77% dei dati reali, quindi è affidabile ed
# è giusto usarlo.
# RIASSUMENDO: Le prime tre componenti spiegano circa il 77% 
# della varianza, quindi rappresentano bene la struttura dei 
# dati.

# pca$rotation = cosa significano le componenti
# Questa è la parte più importante
# PC1:
#     loudness  = -0.68
#     energy    = -0.68
#     liveness  = -0.22
# Interpretazione: PC1 è dominata da loudness ed energy
# Quindi: PC1 ≈ “intensità musicale”
# (canzoni forti e energetiche vs tranquille)

# PC2:
#     instrumentalness = -0.83
#     duration_ms      = -0.44
# Interpretazione:
# PC2 distingue tracce strumentali, durata
# Quindi: PC2 ≈ “tipo di brano / struttura”

# PC3:
#     duration_ms = +0.87
# Interpretazione: PC3 è praticamente la durata del brano

# PC4:
#     liveness = +0.94
# Rappresenta: quanto il brano è live

# PC5: 
#     loudness  = +0.69
#     energy    = -0.68
# Contrappone loudness vs energy
# (ma è poco importante → 5%)

# pca$x => Esempio: [1,] -1.27   0.60   -0.72 ...
# Significa: per la canzone 1:
#                             PC1 = -1.27 → bassa intensità
#                             PC2 = 0.60 → valore medio
#                             PC3 = -0.72 → durata più bassa
# Sono semplicemente coordinate nel nuovo spazio PCA
# COLLEGAMENTO AL GRAFICO
# Nel grafico 3D:
#                asse X = PC1 → intensità
#                asse Y = PC2 → tipo brano
#                asse Z = PC3 → durata
# Quindi sto vedendo come le canzoni si distribuiscono per:
# intensità, tipo, durata.

# INTERPRETAZIONE FINALE: La prima componente principale è 
# fortemente influenzata da loudness ed energy e rappresenta 
# l’intensità del brano. Le componenti successive catturano 
# altre caratteristiche come durata e liveness. Le prime tre 
# componenti spiegano circa il 77% della variabilità totale, 
# quindi la rappresentazione tridimensionale è significativa.

# Plot PCA 2D: PC1-PC2, PC1-PC3, PC2-PC3
clrs <- c("deepskyblue", "green3", "red") # Colori cluster
centers_pca <- predict(pca, newdata = km.raw$centers) # centroidi nello spazio PCA

par(mfrow = c(1, 3))
plot(pca$x[,1], pca$x[,2],
     col = clrs[km.raw$cluster],
     pch = 19,
     cex = 0.7,
     xlab = "PC1",
     ylab = "PC2",
     main = "PCA: PC1 vs PC2")
points(centers_pca[,1], centers_pca[,2],
       pch = 23, cex = 2, bg = clrs, col = "black", lwd = 2)

plot(pca$x[,1], pca$x[,3],
     col = clrs[km.raw$cluster],
     pch = 19,
     cex = 0.7,
     xlab = "PC1",
     ylab = "PC3",
     main = "PCA: PC1 vs PC3")
points(centers_pca[,1], centers_pca[,3],
       pch = 23, cex = 2, bg = clrs, col = "black", lwd = 2)

plot(pca$x[,2], pca$x[,3],
     col = clrs[km.raw$cluster],
     pch = 19,
     cex = 0.7,
     xlab = "PC2",
     ylab = "PC3",
     main = "PCA: PC2 vs PC3")
points(centers_pca[,2], centers_pca[,3],
       pch = 23, cex = 2, bg = clrs, col = "black", lwd = 2)

par(mfrow = c(1, 1))

# GRAFICO 3D INTERATTIVO
pca_df <- data.frame(
  PC1 = pca$x[,1],
  PC2 = pca$x[,2],
  PC3 = pca$x[,3],
  cluster = factor(km.raw$cluster)
)


fig_kms <- plot_ly(
  data = pca_df,
  x = ~PC1,
  y = ~PC2,
  z = ~PC3,
  type = "scatter3d",
  mode = "markers",
  color = ~cluster,
  colors = c("deepskyblue","green3","red"),
  marker = list(size = 4),
  text = ~paste(
    "Cluster:", cluster,
    "<br>PC1:", round(PC1, 2),
    "<br>PC2:", round(PC2, 2),
    "<br>PC3:", round(PC3, 2)
  ),
  hoverinfo = "text"
)

fig_kms


##### 5) CLUSTERING BAYESIANO #####------------------------------------------------
spotify_optim <- subset(
  spotify_sample_numeric,
  select = c(duration_ms, liveness, instrumentalness, loudness, energy)
)

# Per evitare problemi, forzo tutto a numerico
spotify_optim <- as.data.frame(lapply(spotify_optim, as.numeric))

# Riordino le colonne mettendo loudness per prima
# così il vincolo ordered[K] mu1 agisce su una variabile ben separante
spotify_bayes <- spotify_optim[, c("loudness", "energy", "duration_ms", "liveness", "instrumentalness")]

# Matrice dati
X <- as.matrix(spotify_bayes)

N <- nrow(X)
P <- ncol(X)
K <- 3

# 2. INIZIALIZZAZIONE DA K-MEANS
set.seed(123)
km_init <- kmeans(X, centers = K, nstart = 25)

# ordino i centroidi in base alla prima colonna (loudness)
ord <- order(km_init$centers[, 1])
centers_ord <- km_init$centers[ord, , drop = FALSE]

init_fun <- function() {
  list(
    w = rep(1 / K, K),
    mu1 = centers_ord[, 1],
    mu_rest = centers_ord[, -1, drop = FALSE],
    sigma2 = matrix(1, nrow = K, ncol = P)
  )
}

stan_data <- list(
  N = N,
  P = P,
  K = K,
  X = X,
  alpha = c(2, 2, 2),   # Dirichlet un po' più stabile del caso uniforme
  mu0 = rep(0, P),      # dati già standardizzati
  kappa0 = 2,
  a0 = 3,
  b0 = 2
)

fit_gmm <- stan(
  file = "C:\\Users\\emanu\\Desktop\\Progetto Bayesiana\\prova.stan",
  data = stan_data,
  chains = 2,
  iter = 2000,     # Riduci le iterazioni, 6000 sono tante se il modello converge
  warmup = 1000,   # Se i traceplot sono buoni, 1000 bastano
  seed = 123,
  init = init_fun,
  control = list(
    adapt_delta = 0.90, # 0.99 è estremo, prova 0.90 o 0.95
    max_treedepth = 12  # 15 è molto profondo, 12 è uno standard solido
  )
)

print(fit_gmm, pars = c("w", "mu"))

# ESTRAZIONE DELLA POSTERIOR
post <- rstan::extract(fit_gmm)



R <- dim(post$post_prob)[1]   # numero iterazioni MCMC
N <- dim(post$post_prob)[2]   # numero osservazioni
K <- dim(post$post_prob)[3]   # numero cluster

cat("Iterazioni:", R, "\n")
cat("Osservazioni:", N, "\n")
cat("Cluster:", K, "\n")

# 2. COSTRUZIONE DELLE PARTIZIONI (S)
#    S[r, i] = cluster assegnato al punto i nell'iterazione r
S <- matrix(0, nrow = R, ncol = N)

for (r in 1:R) {
  S[r, ] <- apply(post$post_prob[r, , ], 1, which.max)
}

# 3. POSTERIOR SIMILARITY MATRIX (M)
#    M[i,j] = probabilità che i e j siano nello stesso cluster

cat("Costruzione Posterior Similarity Matrix...\n")

M <- matrix(0, nrow = N, ncol = N)

for (r in 1:R) {
  W_r <- outer(S[r, ], S[r, ], FUN = "==") * 1
  M <- M + W_r
}

M <- M / R

# 4. CALCOLO BINDER LOSS
#    distanza tra M e ogni partizione W_r

cat("Calcolo Binder loss...\n")

distances <- numeric(R)

for (r in 1:R) {
  W_r <- outer(S[r, ], S[r, ], FUN = "==") * 1
  distances[r] <- sum((M - W_r)^2)
}

# 5. PARTIZIONE OTTIMA
best_r <- which.min(distances)

cat("Iterazione ottima:", best_r, "\n")

cluster_binder <- S[best_r, ]

cluster_bayes <- cluster_binder

# aggiungo al dataset
spotify_bayes$cluster_bayes <- factor(cluster_bayes)

# distribuzione cluster
cat("Distribuzione cluster:\n")
print(table(cluster_bayes))



table(cluster_bayes)

# 7. PCA CORRETTA
#    ATTENZIONE: la PCA va fatta SOLO sulle 5 variabili originali
spotify_bayes_plot <- spotify_bayes[, c("loudness", "energy", "duration_ms", "liveness", "instrumentalness")]

pca <- prcomp(spotify_bayes_plot, center = TRUE, scale. = FALSE)

summary(pca)
pca$rotation
head(pca$x)

# 8. PLOT PCA 2D
clrs <- c("deepskyblue", "green3", "red")

par(mfrow = c(1, 3))

plot(pca$x[,1], pca$x[,2],
     col = clrs[cluster_bayes],
     pch = 19,
     cex = 0.7,
     xlab = "PC1",
     ylab = "PC2",
     main = "Bayesian GMM: PC1 vs PC2")

plot(pca$x[,1], pca$x[,3],
     col = clrs[cluster_bayes],
     pch = 19,
     cex = 0.7,
     xlab = "PC1",
     ylab = "PC3",
     main = "Bayesian GMM: PC1 vs PC3")

plot(pca$x[,2], pca$x[,3],
     col = clrs[cluster_bayes],
     pch = 19,
     cex = 0.7,
     xlab = "PC2",
     ylab = "PC3",
     main = "Bayesian GMM: PC2 vs PC3")

par(mfrow = c(1, 1))

# 9. CENTROIDI BAYESIANI NELLO SPAZIO PCA
# media posteriori delle mu
mu_mean <- apply(post$mu, c(2, 3), mean)

# nomi colonne coerenti
colnames(mu_mean) <- colnames(spotify_bayes_plot)

# proiezione delle medie nello spazio PCA
centers_pca <- predict(pca, newdata = mu_mean)

par(mfrow = c(1, 3))

plot(pca$x[,1], pca$x[,2],
     col = clrs[cluster_bayes],
     pch = 19,
     cex = 0.7,
     xlab = "PC1",
     ylab = "PC2",
     main = "Bayesian GMM: PC1 vs PC2")
points(centers_pca[,1], centers_pca[,2],
       pch = 23, cex = 2, bg = clrs, col = "black", lwd = 2)

plot(pca$x[,1], pca$x[,3],
     col = clrs[cluster_bayes],
     pch = 19,
     cex = 0.7,
     xlab = "PC1",
     ylab = "PC3",
     main = "Bayesian GMM: PC1 vs PC3")
points(centers_pca[,1], centers_pca[,3],
       pch = 23, cex = 2, bg = clrs, col = "black", lwd = 2)

plot(pca$x[,2], pca$x[,3],
     col = clrs[cluster_bayes],
     pch = 19,
     cex = 0.7,
     xlab = "PC2",
     ylab = "PC3",
     main = "Bayesian GMM: PC2 vs PC3")
points(centers_pca[,2], centers_pca[,3],
       pch = 23, cex = 2, bg = clrs, col = "black", lwd = 2)

par(mfrow = c(1, 1))

# 11. PLOT PCA 3D INTERATTIVO
pca_df <- data.frame(
  PC1 = pca$x[,1],
  PC2 = pca$x[,2],
  PC3 = pca$x[,3],
  cluster = factor(cluster_bayes)
)

centers_df <- data.frame(
  PC1 = centers_pca[,1],
  PC2 = centers_pca[,2],
  PC3 = centers_pca[,3],
  cluster = factor(1:3)
)

fig <- plot_ly(
  data = pca_df,
  x = ~PC1,
  y = ~PC2,
  z = ~PC3,
  type = "scatter3d",
  mode = "markers",
  color = ~cluster,
  colors = clrs,
  marker = list(size = 4),
  text = ~paste(
    "Cluster:", cluster,
    "<br>PC1:", round(PC1, 2),
    "<br>PC2:", round(PC2, 2),
    "<br>PC3:", round(PC3, 2)
  ),
  hoverinfo = "text"
) %>%
  add_trace(
    data = centers_df,
    x = ~PC1,
    y = ~PC2,
    z = ~PC3,
    type = "scatter3d",
    mode = "markers",
    marker = list(size = 8, symbol = "diamond", color = "black"),
    text = ~paste("Centroide bayesiano cluster", cluster),
    hoverinfo = "text",
    inherit = FALSE
  ) %>%
  layout(
    title = "Bayesian GMM - PCA 3D interattiva",
    scene = list(
      xaxis = list(title = "PC1"),
      yaxis = list(title = "PC2"),
      zaxis = list(title = "PC3")
    )
  )

fig

# 12. SCATTERPLOT MATRIX COLORATA
panel.cluster <- function(x, y, ...) {
  points(x, y,
         pch = 19,
         cex = 0.6,
         col = clrs[cluster_bayes])
}

pairs(
  spotify_bayes_plot,
  lower.panel = panel.cluster,
  upper.panel = panel.cluster,
  diag.panel = NULL,
  main = "Scatterplot matrix con cluster bayesiani"
)

print(fit_gmm, pars = c("w", "mu"))



##### 6.1) CONFRONTO OSS. ASSEGNATE AI CLUSTER #####----------------------------

tab <- table(Kmeans = km.raw$cluster, Bayes = cluster_bayes)
tab
prop_tab <- prop.table(tab, margin = 1)
round(prop_tab, 3)

##### 6.2) REGRESSIONE 1(SOLO EFFETTO CLUSTERING) #####-------------------------

# DATASET PER REGRESSIONE
reg_df <- data.frame(
  track_popularity = spotify_sample$track_popularity,
  cluster = factor(cluster_bayes)
)

str(reg_df)
table(reg_df$cluster)
summary(reg_df$track_popularity)

boxplot(track_popularity ~ cluster,
        data = reg_df,
        col = c("deepskyblue", "green3", "red"),
        main = "Track popularity by Bayesian cluster",
        xlab = "Cluster",
        ylab = "Track popularity (standardized)")

aggregate(track_popularity ~ cluster, data = reg_df, mean)
aggregate(track_popularity ~ cluster, data = reg_df, sd)
aggregate(track_popularity ~ cluster, data = reg_df, median)

# REGRESSIONE LINEARE
mod_lm <- lm(track_popularity ~ cluster, data = reg_df)

summary(mod_lm)
confint(mod_lm)
anova(mod_lm)

# TEST GLOBALE
anova(mod_lm)
# Qui verifichi: H0: mu1=mu2=mu3
#         contro H1: almeno una media differisce
# Se il p-value è piccolo: La popularity media differisce significativamente 
# tra i cluster.


##### 6.2) REGRESSIONE 2(AGGIUNTA ALTRE VARIABILI) #####------------------------

reg_df2 <- data.frame(
  track_popularity = spotify_sample$track_popularity,
  cluster = factor(cluster_bayes),
  danceability = spotify_sample$danceability,
  speechiness = spotify_sample$speechiness,
  acousticness = spotify_sample$acousticness,
  valence = spotify_sample$valence,
  tempo = spotify_sample$tempo
)

mod2 <- lm(track_popularity ~ cluster + danceability + speechiness + acousticness + valence + tempo,
           data = reg_df2)

summary(mod2)
confint(mod2)
anova(mod2)

summary(spotify_sample)