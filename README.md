# IGJ-thesis-code

This repository contains the Python and R code used to produce all analyses, figures, and results presented in the Master's Thesis "Data Analysis of Antibiotic Resistance in Escherichia coli".

There are four main scripts:

- **PCA_FSA.ipynb**: Displays the antibiotic correlation matrix, Principal Component Analysis (PCA), and Forward Stepwise Algorithm (FSA) results.
- **AUTOENCODER.ipynb**: Contains all analyses and results related to the autoencoder model.
- **try.r**: Implements the autoencoder using Torch in RStudio and includes training, validation, and testing procedure.
- **aenn2app.r**: Applies the NN2Poly method to the selected autoencoder and presents the resulting analyses and outputs.

Additional files included in the repository:

- **DATA_eCOLI.xlsx**: Preprocessed dataset used throughout the analyses.
- **DATA_eCOLI_raw.xlsx**: Preprocessed dataset prior to logarithmic transformation.
- **Escherichia coli_2018.xlsx**: Source dataset from which the project was initiated, based on the data preparation performed by Ana Azcue.
- **poly_deg1_ep400.xlsx**: Contains the 100 most significant coefficients for each of the 20 final polynomials generated using the NN2Poly methodology up to degree 1, using 400 epochs.
- **poly_deg2_ep400.xlsx**: Contains the 100 most significant coefficients for each of the 20 final polynomials generated using the NN2Poly methodology up to degree 2, using 400 epochs.
- **poly_deg3_ep400.xlsx**: Contains the 100 most significant coefficients for each of the 20 final polynomials generated using the NN2Poly methodology up to degree 3, using 400 epochs.
