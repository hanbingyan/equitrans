# Equilibrium Transport

## Packages
We use [POT](https://pythonot.github.io/) for optimal transport in one period. 
The implementations of Jenks natural breaks and total variation denoising in 1d are based 
on other authors' work, given in the files `jenks.py` and `tv1d.py`. 

Thanks to the authors of these packages!

## Usage
Code for two job markets is given in two folders `Executives` and `California`, 
which can work separately. 

## Illustrative examples

`MV_precommit.py` and `MV_equilibrium.py` are for the toy example on dynamic matching 
between supply and demand.

## Executives
### Data
With access to WRDS, we download financial statements data from Compustat North America and 
executive compensation data from Execucomp.

In Compustat, we choose the following query variables. For five-year data, set the date 
as 2017 to 2022. Name the output file as `finstat_y17.csv`.
* Global Company Key (GVKEY)
* Ticker Symbol (TIC)
* GIC Groups (GGROUP)
* GIC Industries (GIND)
* GIC Sectors (GSECTOR)
* GIC Sub-Industries (GSUBIND)
* Data Year Fiscal (FYEAR)
* Sales/Turnover (Net) (SALE)
* Market Value Total Fiscal (MKVALT)

In Execucomp, we choose the following query variables. For five-year data, set the date 
as 2017 to 2022. Name the output file as `ceo_y17.csv`. 
* Compustat's Global Company Key (GVKEY)
* (Current) Ticker Symbol (TICKER)
* EXEC_FULLNAME (EXEC_FULLNAME)
* Date Became CEO (BECAMECEO)
* Title (TITLE)
* Total Compensation (Salary + Bonus + Others) (TDC1)
* Year (YEAR)

Credit rating data are obtained from Compustat Daily Updates - Ratings on date 2017-01 only. Name the 
output file as `rating_y17.csv`. Variables include 
* Company Name (CONM)
* Ticker Symbol (TIC)
* CUSIP (CUSIP)
* S&P Domestic Long Term Issuer Credit Rating (SPLTICRM)
* S&P Subordinated Debt Rating (SPSDRM)
* S&P Domestic Short Term Issuer Credit Rating (SPSTICRM)

**For validation, `790firms_gvkey.txt` gives the GVKEY of 790 firms after cleaning 2017-2021 data.
Readers can also use this file in WRDS queries.**

### Replication steps

1. With data obtained, use `CleanData_5Years.ipynb` to clean and merge data. 
The output is `merged_y17_avgmanager.csv`. 
2. Select a suitable number of groups with `GroupNumberSelection.py`. There are two flag variables,
`search_mode` and `even_split`. If `search_mode` is true, it will consider all candidates of group numbers
and generate correlation pickle files to plot the group number screening figure. If
`search_mode` is false, only one candidate of group numbers is used and it generates data files
called `classified_group_{}.csv` and `wage_beta_group_{}.csv`. `even_split` controls whether to use
even splits.
3. Estimate transition matrices. We need to download the data for a longer time horizon 
such as 2000-2022. Name the output files from WRDS as `ceo_20years.csv` and `finstat_20years.csv`. 
Then clean the data with `CleanData_12Years_TransMat.py` and remember to set `n_group` as the
one obtained from Step 2. Next, run `Calc_TransMatrix.py` with the same `n_group` to estimate
transition matrices.
4. `Executives_equi.py` calculates theoretical equilibrium transport with different alphas. 
The flag variable `bench_zero` means to use perfectly matched data (=True) or bootstrap real
data (=False). Also, set `n_groups` the same as above.
5. `Executives_cali.py` generates the Sinkhorn distances between theoretical and real transport
plans. Variables `bench_zero` and `n_groups` should be the same as in Step 4.
6. Figures and tables are obtained by `Tables_Statistics.ipynb` and `Tables_Executives_Alpha.ipynb`.


## California

### Data
The University of California (UC) Compensation data are from 
[Government Compensation in California website](https://publicpay.ca.gov/Reports/RawExport.aspx).
Put 2013-2021 files into a folder called `UniversityOfCalifornia`. The dataset in the Year 2013 uses 
different job titles. For example, 'Assoc Prof-Ay-B/E/E' is called 
'Associate Prof-Ay-B/E/E'. We use `Clean_Y2013_Data.ipynb` to unify the job titles. 
US News Rankings are obtained from [Andrew G. Reiter's website](https://andyreiter.com/datasets/).
We have added `USNews_Ranking.csv` for readers' reference.

`Clean_UC_Salary.ipynb` cleans and merges compensations and rankings data. Readers can also
use the output `uc_salary.csv` directly. Wages are also rankings in this file.

### Replication steps
1. Estimate transition matrices with `UC_transmatrix.py`.
2. Similarly, `UC_equi.py` calculates theoretical equilibrium transport with different alphas. 
The flag variable `bench_zero` means to use perfectly matched data (=True) or bootstrap real
data (=False).
3. `UC_cali.py` generates the Sinkhorn distances between theoretical and real transport
plans. Variables `bench_zero` should be the same as in Step 2.
4. Figures and tables are obtained by `Results_Figs_Tables.ipynb` and 
`Descriptive_Statistics.ipynb`.

## Cobb-Douglas 
It contains code for the model following Gabaix and Landier (2008).
