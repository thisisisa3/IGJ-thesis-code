
library(readxl)
library(torch)
library(luz)
library(nn2poly)
library(ggplot2)
library(patchwork)
library(cowplot)
library(openxlsx)

options(torch.environment_viewer = FALSE)

data <- "C:/Users/belen/OneDrive/Escritorio/TFM ISA/DATA_eCOLI.xlsx"
df <- read_excel(data)
df <- df[, sapply(df, is.numeric)]

latent_dim <- 8
capa_int <- 19
batch_size <- 256
epochs <- 400
learning_rate <- 1e-3
max_degree <- 2

# scaling data 
X <- scale(as.matrix(df))

col_min <- apply(X, 2, min)
col_max <- apply(X, 2, max)


X <- sweep(X, 2, col_max - col_min, "/")
X <- 2 * X - 1  # Esto desplaza el rango de [0, 1] al rango [-1, 1]
X <- matrix(as.numeric(X), nrow = nrow(X), ncol = ncol(X))

set.seed(42)
idx <- sample(1:nrow(X), size = floor(0.8 * nrow(X)))
X_train <- X[idx, ]
X_val <- X[-idx, ]
input_dim <- ncol(X)

X_train_tensor <- torch_tensor(X_train, dtype = torch_float())
X_val_tensor   <- torch_tensor(X_val,   dtype = torch_float())

train_ds <- tensor_dataset(X_train_tensor, X_train_tensor)
val_ds   <- tensor_dataset(X_val_tensor, X_val_tensor)

train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE, num_workers = 0)
val_dl   <- dataloader(val_ds,   batch_size = batch_size, shuffle = FALSE, num_workers = 0)

# MODEL SETUP WITH CONSTRAINTS

autoencoder_model <- luz_model_sequential(
  nn_linear(input_dim, capa_int),  
  nn_tanh(),                        
  nn_linear(capa_int, latent_dim), 
  nn_tanh(),
  nn_linear(latent_dim, capa_int), 
  nn_tanh(),
  nn_linear(capa_int, input_dim)   
)

autoencoder_model <- luz::setup(module = autoencoder_model, loss = nn_mse_loss(), optimizer = optim_adam)
autoencoder_constrained <- add_constraints(autoencoder_model, type = "l1_norm")

# TRAINING

fitted_autoencoder <- fit(autoencoder_constrained, data = train_dl, valid_data = val_dl, epochs = epochs)

# NN2POLY REPRESENTATION EXTRACTION

cat("\n--- Extracting Polynomial Representation via nn2poly ---\n")

# Extract the raw torch sequential model from the luz training wrapper
torch_autoencoder <- fitted_autoencoder$model

# Put the model in evaluation mode (freezes weights/constraints)
torch_autoencoder$eval()

# Compute the final Taylor polynomial representation
# Note: 'max_degree' is used in the stable CRAN version of nn2poly to denote 'max_order'
final_poly <- nn2poly(
  object = torch_autoencoder,
  max_degree = max_degree,       # Changes the max polynomial order to 3
  keep_layers = FALSE   # Only returns the final output layer polynomials
)

# print(final_poly)

setwd("C:/Users/belen/OneDrive/Escritorio/TFM ISA")

# Predict using both models on your validation data
ae_preds   <- as.matrix(with_no_grad({ torch_autoencoder(X_val_tensor) }))
poly_preds <- predict(final_poly, X_val)


# Plot: Comparison between Autoencoder and Polynomial approximation

p_diagonal <- nn2poly:::plot_diagonal(
  x_axis = as.vector(ae_preds),
  y_axis = as.vector(poly_preds),
  xlab   = "Autoencoder Predictions",
  ylab   = "Polynomial Predictions",
  title  = "Fidelity Check: Autoencoder vs. Polynomial Model"
) +
  theme(
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5), # Título principal 
    axis.title.x = element_text(size = 16),                           # Título del eje X
    axis.title.y = element_text(size = 16),                           # Título del eje Y
    axis.text = element_text(size = 12)                               # Números de los 
  )

ggsave(
  filename = "C:/Users/belen/OneDrive/Escritorio/TFM ISA/fidelity_plot_deg2_ep400.png",
  plot = p_diagonal,
  width = 8,
  height = 6,
  dpi = 300
)


mse_poly <- mean((X_val - poly_preds)^2)
cat("Results for max_order=", max_degree, "\n")

cat("Polynomial MSE:", mse_poly, "\n")

mse_fidelity <- mean((ae_preds - poly_preds)^2)
cat("Fidelity MSE:", mse_fidelity, "\n")

# str(final_poly, max.level = 2)
# dim(final_poly$values)
# length(final_poly$labels)


# EXPORT TOP 100 MOST IMPORTANT POLYNOMIAL TERMS

term_name <- function(idx){

  if(length(idx) == 1 && idx == 0)
    return("1")

  tab <- table(idx)

  paste(
    sapply(names(tab), function(v){

      p <- tab[[v]]

      if(p == 1)
        paste0("x", v)
      else
        paste0("x", v, "^", p)

    }),
    collapse = " "
  )
}

poly_df <- data.frame(
  Term = sapply(final_poly$labels, term_name),
  final_poly$values,
  check.names = FALSE
)

colnames(poly_df)[-1] <- paste0(
  "Output_",
  seq_len(ncol(final_poly$values))
)

# Importance = L2 norm across all outputs
poly_df$Importance <- sqrt(rowSums(final_poly$values^2))

# Keep only the 100 most important terms
poly_df <- poly_df[
  order(poly_df$Importance, decreasing = TRUE),
]

poly_df <- head(poly_df, 100)

# Export to Excel
openxlsx::write.xlsx(
  poly_df,
  file = "C:/Users/belen/OneDrive/Escritorio/TFM ISA/poly_deg2_ep400.xlsx",
  rowNames = FALSE
)
