* Econ 6343 - Problem Set 8
* Name: Mahendra Jamankar

clear
set more off
capture log close

log using PS8stata.log, replace

import delimited "nlsy.csv", clear

***************************************************************************
* Q1. Basic regression model
reg logwage black hispanic female schoolt gradhs grad4yr, robust

* Q2. Correlation matrix of ASVAB variables
correlate asvab*

* Q3. Regression with all ASVAB variables
reg logwage black hispanic female schoolt gradhs grad4yr asvab*, robust
***************************************************************************
* Q4. PCA Analysis
* First, conduct PCA on ASVAB variables
pca asvab*
* Predict first principal component
predict pc1, score
* Run regression with first principal component
reg logwage black hispanic female schoolt gradhs grad4yr pc1, robust
**************************************************************************

* Q5. Factor Analysis
* Conduct factor analysis on ASVAB variables
factor asvab*, factors(1)
* Predict factor score
predict factor1
* Run regression with factor score
reg logwage black hispanic female schoolt gradhs grad4yr factor1, robust
***************************************************************************

* Q6. Maximum Likelihood Estimation of measurement system

* First, standardize ASVAB variables for better convergence
foreach var of varlist asvab* {
    egen std_`var' = std(`var')
}

* Estimate the measurement system using GSEM
gsem (std_asvab* <- black hispanic female ability@1) ///
     (logwage <- black hispanic female schoolt gradhs grad4yr ability), ///
     latent(ability) ///
     cov(ability*ability@1) ///
     method(ml) difficult
	 
* Display results
estimates store model1
estimates table model1, star stats(N r2)
