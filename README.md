# Enhancing Risk Assessment of Listeria monocytogenes Using Machine Learning and Genomic Data

# Overview
This project utilizes machine learning (ML) techniques and Next Generation Sequencing (NGS) data to enhance the predictive accuracy of microbial risk assessment (MRA) for Listeria monocytogenes. By analyzing genomic data, the project aims to identify key virulence factors and stress resistance genes that contribute to the pathogenicity of this foodborne pathogen, thereby improving food safety and public health outcomes.

# Problem Definition
Current microbial risk assessment models often fail to account for the genetic diversity across strains of Listeria monocytogenes. This project aims to overcome the limitations of traditional methods by integrating machine learning with NGS data, allowing for more precise predictions of the risk of illness caused by specific strains of Listeria monocytogenes based on their genetic makeup.

# Key Features
Data Collection: The project uses genomic data from 207 Listeria monocytogenes strains, sourced from various food products and environments. These data are publicly available via NCBI (SRA dataset).
Bioinformatics Workflow: Genome assembly, gene prediction, and virulence gene detection are performed using tools like Velvet, Prodigal, and VFDB BLAST.
Machine Learning Models: Various regression models (e.g., Random Forest, Ridge Regression, Support Vector Regression) are trained to predict the likelihood of illness based on genomic data.

# Methodology

Bioinformatics Workflow
Data Collection: 207 strains from various food sources (e.g., fish, meat, vegetables, etc.) were collected.
Genome Assembly: Velvet was used to assemble the genomes.
Gene Prediction: Prodigal was used to identify open reading frames (ORFs) and predict proteins.
Virulence Gene Detection: The predicted proteins were aligned against the Virulence Factor Database (VFDB) to identify virulence genes in each strain.
Matrix Construction: A matrix of virulence gene presence/absence was created for each strain.

Machine Learning Workflow
Data Preprocessing: Missing values were filled, and data were normalized using MinMaxScaler.
Feature Selection: Various methods, including PCA and feature importance scores, were used to select the most relevant features.
Model Selection: Several regression models were tested, including Random Forest, Lasso, Ridge, and Support Vector Regression, among others.
Model Evaluation: The models were evaluated based on Mean Squared Error (MSE) and R² metrics.

# Results
The project conducted several trials with different feature selection and extraction techniques:

Without Feature Selection (with PCA): Random Forest achieved the best performance with an MSE of 467.37 and an R² of 0.1022.

With Feature Selection (without PCA): Lasso Regression performed best before feature selection (MSE = 518.85), while Ridge Regression was the best after feature selection.

With Feature Selection and PCA: Random Forest remained the best model before feature selection (MSE = 467.37), but the performance slightly decreased after feature selection (MSE = 482.80).

Neural Network (NN)

Without PCA: The best performance was achieved with a neural network, resulting in an R² score of 0.045.

With PCA: When Principal Component Analysis (PCA) was applied, the best performance was slightly reduced, resulting in an MSE of 413.62 and an R² score of 0.067.

XGBoost

Best Performance: XGBoost, when using the top 31 features, achieved the highest performance with an R² score of 0.16.

# Requirements

Python 3.x

Tools (Velvet, Prodigal, VFDB BLAST)

Necessary libraries (scikit-learn, numpy, pandas, matplotlib, itertools, seaborn, tensorflow, keras, keras_tuner, xgboost)

NCBI SRA dataset (access for genomic data)

# Installation
Install Python 3.x (if not already installed).

Install required libraries:
Copy code

pip install -r requirements.txt

Download genomic data from the NCBI SRA dataset.

# Usage

Prepare the genomic data as described in the Bioinformatics Workflow section.
Run the preprocessing script to clean and normalize the data.
Train and evaluate machine learning models by running the train_model.py script.
View the results in the output logs.
# Conclusion
By incorporating machine learning into microbial risk assessment, this project provides a more accurate and comprehensive model for predicting the risks associated with Listeria monocytogenes. These insights can be used to develop targeted interventions that improve food safety and public health.

# Acknowledgments
The genomic data used in this project were provided by NCBI (SRA dataset).
Special thanks to the authors of the related papers and tools (Velvet, Prodigal, VFDB BLAST, etc.).
License
This project is licensed under the MIT License - see the LICENSE file for details.
