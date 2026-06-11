
library(readxl)
library(torch)
library(luz)
library(nn2poly)

options(torch.environment_viewer = FALSE)

# =========================================================
# DATA & PARAMETERS
# =========================================================
data <- "C:/Users/belen/OneDrive/Escritorio/TFM ISA/DATA_eCOLI.xlsx"
df <- read_excel(data)
df <- df[, sapply(df, is.numeric)]

latent_dim <- 8
capa_int <- 19
batch_size <- 256
epochs <- 700
learning_rate <- 1e-3

# =========================================================
# PREPROCESSING & TENSORS
# =========================================================
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

# =========================================================
# MODEL SETUP WITH CONSTRAINTS
# =========================================================
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

# =========================================================
# TRAINING
# =========================================================
fitted_autoencoder <- fit(
  autoencoder_constrained,
  data = train_dl,
  valid_data = val_dl,
  epochs = epochs
)

# Save model
luz_save(fitted_autoencoder, "modelo_temporal.pt")

# =========================================================
# TRAINING HISTORY PLOT
# =========================================================

library(ggplot2)
setwd("C:/Users/belen/OneDrive/Escritorio/TFM ISA")

metrics <- fitted_autoencoder$records$metrics

train_loss <- sapply(metrics$train, function(x) x$loss)
valid_loss <- sapply(metrics$valid, function(x) x$loss)

loss_df <- data.frame(
  Epoch = seq_along(train_loss),
  Train_MSE = train_loss,
  Validation_MSE = valid_loss
)

print(loss_df)

p <- ggplot(loss_df, aes(x = Epoch)) +

  geom_line(
    aes(y = Train_MSE, color = "Training"),
    linewidth = 1.4
  ) +

  geom_line(
    aes(y = Validation_MSE, color = "Validation"),
    linewidth = 1.4
  ) +

  coord_cartesian(
    ylim = c(0, 0.3)
  )+

  labs(
    title = "Training and Validation MSE",
    x = "Epoch",
    y = "MSE",
    color = NULL
  ) +

  scale_color_manual(
    values = c(
      "Training" = "#1f77b4",
      "Validation" = "#ff7f0e"
    )
  ) +

  scale_x_continuous(
  breaks = c(100, 200, 250, 400, 600)
  ) +

  theme_bw(base_size = 18) +

  theme(
    plot.title = element_text(
      size = 22,
      face = "bold",
      hjust = 0.5
    ),

    axis.title = element_text(
      size = 18,
      face = "bold"
    ),

    axis.text = element_text(
      size = 14
    ),

    legend.position = "top",
    legend.text = element_text(size = 14)
  )

print(p)

ggsave(
  "C:/Users/belen/OneDrive/Escritorio/TFM ISA/train_val_e700.png",
  plot = p,
  width = 8,
  height = 8,
  dpi = 600,
  bg = "white"
)