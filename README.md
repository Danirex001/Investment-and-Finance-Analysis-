# Investment and finance Analysis
This project focuses on analyzing a financial dataset to uncover investment patterns, group individuals by investment behavior, and apply a machine learning classification model to predict investor categories based on demographic and financial traits.

## **Table of contents**

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Overview

The script performs:

Exploratory Data Analysis (EDA) on gender, financial goals, and investment purposes.
Filtering of top wealth creators based on conditions.
Categorization of investors into lightweight, heavyweight, and middleweight.
A machine learning model (Random Forest Classifier) that predicts the type of investor based on age and investment preferences.

## Features

Grouping and summarizing data by gender and age.
Applying custom logic to classify investors.
One-hot encoding of categorical variables for model training.
Predicting investor type using a trained classifier.
## Installation

Clone the repository:
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Install required packages:
pip install pandas numpy matplotlib scikit-learn
## Usage

Ensure your dataset is correctly loaded as fin or finance DataFrame.
Run the Python script:
python your_script.py
Outputs include:
Summarized statistics by gender
Value counts of objectives
Percentage of top wealth creators
Investor classification
Model predictions
## Dataset

The dataset should contain the following columns:

Gender
Gold
Purpose
Objective
mutual_Funds
Avenue
Investment_Avenues
age
Note: Make sure column names are consistent and correctly cased.

## Results

The trained Random Forest model can predict an investor’s category based on:

Age
Investment preferences
Financial objectives
You can evaluate model performance using accuracy score and classification report.

## Technologies Used

Python
Pandas
NumPy
Matplotlib
Scikit-learn

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for improvements or suggestions.

## License

This project is licensed under the MIT License.


