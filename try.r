
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
epochs <- 400
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
fitted_autoencoder <- fit(autoencoder_constrained, data = train_dl, valid_data = val_dl, epochs = epochs)


# Guardamos el modelo entrenado temporalmente por si decides usarlo
luz_save(fitted_autoencoder, "modelo_temporal.pt")

# =========================================================
# CLEAN + TRUE SQUARE PLOT
# =========================================================

library(ggplot2)

p <- ggplot(loss_df, aes(x = Epoch)) +

  # Curves
  geom_line(aes(y = Train_MSE, color = "Train MSE"),
            linewidth = 1.6) +
  geom_line(aes(y = Validation_MSE, color = "Validation MSE"),
            linewidth = 1.6) +

  # TRUE square scaling (same x and y physical scale)
  coord_fixed(
    ratio = max(loss_df$Epoch) / 0.30,
    ylim = c(0, 0.30)
  ) +

  labs(
    title = "Autoencoder Training History",
    x = "Epoch",
    y = "MSE Loss",
    color = NULL
  ) +

  # Professional colors
  scale_color_manual(values = c(
    "Train MSE" = "#1f77b4",
    "Validation MSE" = "#ff7f0e"
  )) +

  # Better ticks
    scale_x_continuous(
  breaks = seq(
    0,
    max(loss_df$Epoch),
    by = 50
  ),
  expand = c(0.01,0)
)+
  scale_y_continuous(
    breaks = seq(0,0.30,0.05),
    expand = c(0,0)
  ) +

  # Cleaner theme
  theme_classic(base_size = 18) +

  theme(

    # ----- BIGGER TEXT -----
    plot.title = element_text(
      size = 24,
      face = "bold",
      hjust = 0.5,
      margin = margin(b = 18)
    ),

    axis.title.x = element_text(
      size = 20,
      face = "bold",
      margin = margin(t = 14)
    ),

    axis.title.y = element_text(
      size = 20,
      face = "bold",
      margin = margin(r = 14)
    ),

    axis.text = element_text(
      size = 17,
      color = "black"
    ),

    # ----- AXES -----
    axis.line = element_line(
      linewidth = 1.2,
      color = "black"
    ),

    axis.ticks = element_line(
      linewidth = 1
    ),

    axis.ticks.length = unit(0.25, "cm"),

    # ----- GRID -----
    panel.grid.major = element_line(
      color = "grey85",
      linewidth = 0.5,
      linetype = "dashed"
    ),
    panel.grid.minor = element_blank(),

    # ----- LEGEND -----
    legend.position = "top",

    legend.text = element_text(
      size = 16
    ),

    legend.key.width = unit(1.4, "cm"),

    legend.background = element_blank(),

    # Margins
    plot.margin = margin(15,15,15,15)
  )

print(p)

# Save high-quality figure
ggsave(
  "curva_aprendizaje_square_TFM.png",
  plot = p,
  width = 8,
  height = 8,
  dpi = 600,
  bg = "white"
)

