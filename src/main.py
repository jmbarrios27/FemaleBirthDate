# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(10,8))
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import warnings
warnings.filterwarnings('ignore')

# Reading Data
birth_df = pd.read_csv('C:\\Users\\Asus\\Desktop\\FemaleBirthdate-main\\femaleBithdates.csv')

# Transforming Date column into datetime format
birth_df['Date'] = pd.to_datetime(birth_df['Date'])

# Checking Data information and NaN values
print('Data information')
print(birth_df.describe())
print()
print('Rows vs Columns', birth_df.shape)
print()
print('Nan Values for each column')
print(birth_df.isna().sum())

# Data Augmentation, Creating Columns to get more data
birth_df['Month'] = birth_df.Date.dt.month
birth_df['Day'] = birth_df.Date.dt.day

# Creating Dataframes grouping by columns
births_by_date = birth_df.groupby(by='Date').sum()
births_by_month = birth_df.groupby(by='Month').sum()
births_by_day = birth_df.groupby(by='Day').sum()

# Dropping Some Columns
births_by_date = births_by_date.drop(columns=['Month', 'Day'])
births_by_month = births_by_month.drop(columns=['Day'])
births_by_day = births_by_day.drop(columns=['Month'])

# Plot Tendency for each day of the year
births_by_date.plot(color='g')
plt.title('BIRTHS BEHAVIOR THROUGH THE YEAR')
plt.ylabel('Number of Births per Date')
plt.show()

# Plot Tendency for each Month
births_by_month.plot(color='y',marker='o')
plt.title('BIRTHS BEHAVIOR PER MONTH')
plt.ylabel('Number of Births per Month')
plt.xticks(births_by_month.index)
plt.show()

# Plot Tendency for each day of the minth
births_by_day.plot(color='r',linestyle='-',marker='o')
plt.title('BIRTHS BEHAVIOR THROUGH DAYS OF THE MONTHS')
plt.ylabel('Number of Births per Day')
plt.xticks(births_by_day.index)
plt.show()

# Lets give a window of 30 days for the year
# Window time
birth_window = birth_df
birth_window = birth_window.drop(columns=['Month','Day'])
birth_df_mean = birth_window
birth_df_mean.set_index('Date', inplace=True)
birth_df_mean = birth_df_mean.rolling( window= 30).mean()
birth_df_mean.plot(color='navy')
plt.title('MEAN OF BIRTHS ON A WINDOW TIME OF 30 DAYS')
plt.xlabel('Window Time')
plt.ylabel('Mean of Births')
plt.show()


# Time Series Analysis
sm.tsa.seasonal_decompose(birth_window.Births).plot()
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(birth_window.Births)[1])
plt.figure(figsize=(20,20))
plt.show()


plot_acf(birth_window)

plot_pacf(birth_window)

# Creating variables with the values and the index (date and number of births)
x, y = birth_window.index , birth_df['Births']

model2 = ARIMA(y, order=(0,1,2))
model2_fit = model2.fit(disp=0)
print(model2_fit.summary())

residuals = pd.DataFrame(model2_fit.resid)

model2_sse = sum((residuals**2).values)
model2_aic = model2_fit.aic


df_comp = birth_window.copy()
df_comp = df_comp.asfreq('d')
df_comp.head()

q = [1,2]
p = [1,7]
d = [0,1]
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter for SARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

rest_dict = {}

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(diff(y),order=param,seasonal_order=param_seasonal)
            results = mod.fit(maxiter=5, method='powell')
#             print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
            rest_dict[param] = {param_seasonal: results.aic}
        except:
            continue
print(rest_dict)

mod =  SARIMAX(diff(y),
               order=(1, 0, 1),
               seasonal_order=(7, 1, 2, 12))

results = mod.fit(maxiter=100, method='powell')
print(results.summary().tables[1])

print(results)

results.plot_diagnostics(figsize=(18, 8))
plt.show()

