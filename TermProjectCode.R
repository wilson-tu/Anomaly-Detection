
#working version 

library(dplyr)
library(lubridate)
library(ggplot2)
library(depmixS4)
library(factoextra)
library(ggplot2)

#1. the feature scaling 

# I. Why scaling features of a dataset is necessary?:
#Scaling features makes sure the features contribute equal amounts to model training

# II. What does normalization and standardization do to the data and the noise?:
#Normalization compresses the values linearly and is sensitive to outliers
#Min-Max: X' = (X - min(X))/(max(X) - min(X)) → [0,1]

#Standardization centers around 0 using SD=1 which preserves the outlier info better
#Z-score: X' = (X - μ)/σ


# III. Which feature scaling method do you choose for HMM training for anomaly detection purposes and why
# I chose standardization because it would preserve the reletive differences in anomalies 
#And would work better with outliers considering we are doign anomaly detection


scale_data <- function(df) {
  scaled_df <- as.data.frame(scale(df))
  attr(scaled_df, "scaled:center") <- NULL
  attr(scaled_df, "scaled:scale") <- NULL
  return(scaled_df)
}

#2 the feature engineering with PCA

# Notes: Based of the psa results i picked global intestity and global active power 
#i didnt use sub metering 3 cause results were better with 2 features instead of 3
# I preformed the pca and then based off the feature contributions to the Principal Component(PC1, ... PC7)
#that had the most proportion of variance I selected the 2 features with the most contirbution

perform_pca_feature_selection <- function(df) {
  pca <- prcomp(df, scale. = TRUE)
  
  # 1 Variance(bar graph)
  var_explained <- pca$sdev^2/sum(pca$sdev^2)
  print(
    ggplot(data.frame(PC = paste0("PC",1:length(var_explained)), 
                      Variance = var_explained[1:length(var_explained)]),
           aes(x = PC, y = Variance)) +
      geom_bar(stat = "identity", fill = "#4E79A7") +
      geom_text(aes(label = paste0(round(Variance*100,1),"%")), 
                vjust = -0.5, size = 3.5) +
      ylim(0, max(var_explained)*1.1) +
      labs(title = "PCA: Variance Explained per Component",
           x = "Principal Component",
           y = "Proportion of Variance Explained") +
      theme_minimal()
  )
  
  #2 Feature contributions to PC1 (bar graph)
  pc1_loadings <- data.frame(
    Feature = rownames(pca$rotation),
    Loading = abs(pca$rotation[,1])
  ) %>% arrange(desc(Loading))
  
  print(
    ggplot(pc1_loadings, aes(x = reorder(Feature, Loading), y = Loading)) +
      geom_bar(stat = "identity", fill = "#59A14F") +
      geom_text(aes(label = round(Loading, 3)), 
                hjust = -0.2, size = 3.5) +
      coord_flip() +
      labs(title = "Feature Contributions to PC1",
           x = "Features",
           y = "Absolute Loading Score") +
      theme_minimal() +
      theme(axis.text.y = element_text(size = 9))
  )
  
  # Get top 2 features
  top_features <- pc1_loadings$Feature[1:2]
  cat("Selected features based on PCA:", paste(top_features, collapse = ", "), "\n")
  
  return(top_features)
}

#3 HMM training and testing
train_and_evaluate_hmm <- function(train_data, test_data, features, n_states_range = seq(4, 20, by=4)) {
  #ntimes/observations per day
  train_dates <- as.Date(train_data$DateTime)
  ntimes <- as.numeric(table(train_dates))
  
  results <- data.frame(
    states = n_states_range,
    train_ll = NA,
    test_ll = NA,
    bic = NA
  )
  
  for (i in seq_along(n_states_range)) {
    n <- n_states_range[i]
    cat("\nTraining model with", n, "states...")
    
    #response formula
    responses <- lapply(features, function(f) as.formula(paste(f, "~ 1")))
    
    #model trainign with depmix like we did in the last assignment
    model <- depmix(response = responses,
                    data = train_data,
                    nstates = n,
                    family = rep(list(gaussian()), length(features)),
                    ntimes = ntimes)
    
    #using fit from the hint in part 1 of the assignment
    fitted_model <- fit(model, verbose = FALSE)
    
    #keeping the training metrics
    results$train_ll[i] <- logLik(fitted_model)/nrow(train_data)
    results$bic[i] <- BIC(fitted_model)/nrow(train_data)
    
    #test log-likelihood
    test_model <- setpars(
      depmix(response = responses,
             data = test_data,
             nstates = n,
             family = rep(list(gaussian()), length(features))),
      getpars(fitted_model)
    )
    results$test_ll[i] <- forwardbackward(test_model)$logLik/nrow(test_data)
    
    cat(" Done. Train LL:", round(results$train_ll[i], 3), 
        "Test LL:", round(results$test_ll[i], 3))
  }
  
  #plotting results
  plot_data <- results %>%
    select(states, train_ll, test_ll) %>%
    pivot_longer(-states, names_to = "dataset", values_to = "loglik")
  
  print(
    ggplot(plot_data, aes(x = states, y = loglik, color = dataset)) +
      geom_line() + geom_point() +
      labs(title = "Normalized Log-Likelihood by State Number",
           x = "Number of States", y = "Log-Likelihood")
  )
  
  print(
    ggplot(results, aes(x = states, y = bic)) +
      geom_line() + geom_point() +
      labs(title = "Normalized BIC by State Number",
           x = "Number of States", y = "BIC")
  )
  
  #lowest BIC with good test LL
  best_model_idx <- which.min(results$bic)
  return(list(
    results = results,
    best_n_states = results$states[best_model_idx]
  ))
}

#4 anomaly detection
establish_anomaly_threshold <- function(model, train_data, test_data, features, n_states) {
  #weekly chunks made from test data we have
  test_data$week <- cut(test_data$DateTime, breaks = "1 week")
  test_chunks <- split(test_data, test_data$week)
  
  #normalized LL for each chunk
  chunk_lls <- sapply(test_chunks, function(chunk) {
    if (nrow(chunk) > 0) {
      test_model <- setpars(
        depmix(response = lapply(features, function(f) as.formula(paste(f, "~ 1"))),
               data = chunk,
               nstates = n_states,
               family = rep(list(gaussian()), length(features))),
        getpars(model)
      )
      return(forwardbackward(test_model)$logLik/nrow(chunk))
    }
    return(NA)
  })
  
  #removing NA chunks from the data
  chunk_lls <- na.omit(chunk_lls)
  
  #threshold (mean - 3*SD)
  train_ll <- logLik(model)/nrow(train_data)
  threshold <- train_ll - 3*sd(chunk_lls)
  
  #plotting distribution
  plot_df <- data.frame(
    type = c(rep("Train", 1), rep("Test Chunks", length(chunk_lls))),
    loglik = c(train_ll, chunk_lls)
  )
  
  print(
    ggplot(data.frame(Week = seq_along(chunk_lls), LogLik = chunk_lls), 
           aes(x = Week, y = LogLik)) +
      geom_point(size = 3, color = "#0072B2") +
      geom_line(alpha = 0.5) +
      geom_hline(yintercept = threshold, 
                 color = "red", 
                 linetype = "dashed",
                 linewidth = 1) +
      geom_hline(yintercept = train_ll, 
                 color = "darkgreen", 
                 linetype = "solid",
                 linewidth = 0.8) +
      annotate("text", 
               x = length(chunk_lls)/2, 
               y = threshold - 0.05, 
               label = paste("Anomaly Threshold:", round(threshold, 3)),
               color = "red") +
      annotate("text",
               x = length(chunk_lls)/2,
               y = train_ll + 0.05,
               label = paste("Training Reference:", round(train_ll, 3)),
               color = "darkgreen") +
      labs(title = "Weekly Log-Likelihood Values with Threshold",
           x = "Week Number", 
           y = "Normalized Log-Likelihood",
           caption = "Red dashed line: Anomaly Threshold\nGreen solid line: Training Reference") +
      theme_minimal() +
      theme(plot.caption = element_text(hjust = 0))
  )
  
  return(threshold)
}

#I basically made a main so its easier to run everything


data <- read.csv("TermProjectData.txt", header = TRUE, sep = ",", na.strings = c("NA", "?"))
data$DateTime <- as.POSIXct(paste(data$Date, data$Time), format = "%d/%m/%Y %H:%M:%S")


numeric_cols <- c("Global_active_power", "Global_reactive_power", "Voltage",
                  "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3")
data <- data %>% 
  select(all_of(c("DateTime", numeric_cols))) %>%
  na.omit()

#standardization
scaled_data <- data %>%
  mutate(across(all_of(numeric_cols), scale))

# pca feature selection
top_features <- perform_pca_feature_selection(scaled_data[numeric_cols])

#the train and test sets (first 3 years for train, last year for test)
train_data <- scaled_data %>% filter(year(DateTime) %in% 2006:2008)
test_data <- scaled_data %>% filter(year(DateTime) == 2009)

#Mondays 2 - 6 AM
train_window <- train_data %>%
  filter(weekdays(DateTime) == "Monday",
         hour(DateTime) %in% 2:5)

test_window <- test_data %>%
  filter(weekdays(DateTime) == "Monday",
         hour(DateTime) %in% 2:5)

#train and evaluate HMMs
hmm_results <- train_and_evaluate_hmm(
  train_window,
  test_window,
  features = top_features,
  n_states_range = seq(4, 20, by=4)
)

#train final model with the best state number
final_model <- depmix(
  response = lapply(top_features, function(f) as.formula(paste(f, "~ 1"))),
  data = train_window,
  nstates = hmm_results$best_n_states,
  family = rep(list(gaussian()), length(top_features)),
  ntimes = as.numeric(table(as.Date(train_window$DateTime)))
)
final_fit <- fit(final_model, verbose = FALSE)

#setting anomaly threshold
threshold <- establish_anomaly_threshold(
  final_fit,
  train_window,
  test_window,
  features = top_features,
  n_states = hmm_results$best_n_states
)

cat("\n=== Final Results ===\n")
cat("Best number of states:", hmm_results$best_n_states, "\n")
cat("Anomaly threshold (normalized log-likelihood):", threshold, "\n")
hmm_results$results$train_ll
