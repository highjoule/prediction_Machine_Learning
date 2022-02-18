# This model review 3 different models that predicts if an operation of a financial company is fraudulent



## The current code includes cleaning process of information, data modelling, display of model results and performance. Methods used include Random Forest, KMeans-cluster and linear regression


```python
import pandas as pd

#We load the information downloaded using dataframe from pandas.

filename = 'data.xls'
#parse_dates is used to used these columns as dates
df0 = pd.read_csv(filename,index_col = 0,parse_dates=['activated_date','last_payment_date'])
df2 = df0[['cust_id','activated_date','last_payment_date','balance','cash_advance','credit_limit']]
```


```python
#the index in this question is set based on the column 'activated_date'
df = df0.set_index('activated_date')
```


```python
df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cust_id</th>
      <th>last_payment_date</th>
      <th>balance</th>
      <th>balance_frequency</th>
      <th>purchases</th>
      <th>oneoff_purchases</th>
      <th>installments_purchases</th>
      <th>cash_advance</th>
      <th>purchases_frequency</th>
      <th>oneoff_purchases_frequency</th>
      <th>purchases_installments_frequency</th>
      <th>cash_advance_frequency</th>
      <th>cash_advance_trx</th>
      <th>purchases_trx</th>
      <th>credit_limit</th>
      <th>payments</th>
      <th>minimum_payments</th>
      <th>prc_full_payment</th>
      <th>tenure</th>
      <th>fraud</th>
    </tr>
    <tr>
      <th>activated_date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-10-06</th>
      <td>C10001</td>
      <td>2020-09-09</td>
      <td>40.900749</td>
      <td>0.818182</td>
      <td>95.40</td>
      <td>0.00</td>
      <td>95.40</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>1000.0</td>
      <td>201.802084</td>
      <td>139.509787</td>
      <td>0.000000</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2019-10-06</th>
      <td>C10002</td>
      <td>2020-07-04</td>
      <td>3202.467416</td>
      <td>0.909091</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>6442.945483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>4</td>
      <td>0</td>
      <td>7000.0</td>
      <td>4103.032597</td>
      <td>1072.340217</td>
      <td>0.222222</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2019-10-06</th>
      <td>C10003</td>
      <td>2020-09-17</td>
      <td>2495.148862</td>
      <td>1.000000</td>
      <td>773.17</td>
      <td>773.17</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>12</td>
      <td>7500.0</td>
      <td>622.066742</td>
      <td>627.284787</td>
      <td>0.000000</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2019-10-06</th>
      <td>C10004</td>
      <td>2020-08-24</td>
      <td>1666.670542</td>
      <td>0.636364</td>
      <td>1499.00</td>
      <td>1499.00</td>
      <td>0.00</td>
      <td>205.788017</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>1</td>
      <td>1</td>
      <td>7500.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2019-10-06</th>
      <td>C10005</td>
      <td>2020-10-20</td>
      <td>817.714335</td>
      <td>1.000000</td>
      <td>16.00</td>
      <td>16.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1200.0</td>
      <td>678.334763</td>
      <td>244.791237</td>
      <td>0.000000</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-07-31</th>
      <td>C19186</td>
      <td>2020-11-03</td>
      <td>28.493517</td>
      <td>1.000000</td>
      <td>291.12</td>
      <td>0.00</td>
      <td>291.12</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>6</td>
      <td>1000.0</td>
      <td>325.594462</td>
      <td>48.886365</td>
      <td>0.500000</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-07-31</th>
      <td>C19187</td>
      <td>2020-09-06</td>
      <td>19.183215</td>
      <td>1.000000</td>
      <td>300.00</td>
      <td>0.00</td>
      <td>300.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>6</td>
      <td>1000.0</td>
      <td>275.861322</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-07-31</th>
      <td>C19188</td>
      <td>2020-06-03</td>
      <td>23.398673</td>
      <td>0.833333</td>
      <td>144.40</td>
      <td>0.00</td>
      <td>144.40</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.000000</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0</td>
      <td>5</td>
      <td>1000.0</td>
      <td>81.270775</td>
      <td>82.418369</td>
      <td>0.250000</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-07-31</th>
      <td>C19189</td>
      <td>2020-07-19</td>
      <td>13.457564</td>
      <td>0.833333</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>36.558778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>2</td>
      <td>0</td>
      <td>500.0</td>
      <td>52.549959</td>
      <td>55.755628</td>
      <td>0.250000</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-07-31</th>
      <td>C19190</td>
      <td>2020-10-14</td>
      <td>372.708075</td>
      <td>0.666667</td>
      <td>1093.25</td>
      <td>1093.25</td>
      <td>0.00</td>
      <td>127.040008</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>2</td>
      <td>23</td>
      <td>1200.0</td>
      <td>63.165404</td>
      <td>88.288956</td>
      <td>0.000000</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>8950 rows × 20 columns</p>
</div>




```python
#The dataframe is group by month
dmonth = df.resample('M')
#Balance is chosen to calculate mean and median
dmb = dmonth['balance']
dmb.mean(),dmb.median()
```




    (activated_date
     2019-10-31    2482.234166
     2019-11-30    1848.704323
     2019-12-31    2018.788906
     2020-01-31    1854.535889
     2020-02-29    1747.350977
     2020-03-31    1554.973023
     2020-04-30    1483.183191
     2020-05-31    1214.333732
     2020-06-30     939.997996
     2020-07-31     649.717622
     Name: balance, dtype: float64,
     activated_date
     2019-10-31    1524.409377
     2019-11-30    1082.071173
     2019-12-31    1162.588384
     2020-01-31    1175.749847
     2020-02-29     994.841733
     2020-03-31     828.954823
     2020-04-30     910.141912
     2020-05-31     734.557681
     2020-06-30     472.791862
     2020-07-31     221.291290
     Name: balance, dtype: float64)




```python
#The mean and median is shown 
plt.plot(dmb.mean())
plt.xticks(rotation=45)
plt.grid(b=True, which='major', color='#A79E9E', linestyle='-', linewidth=0.4)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()
```


    
![](https://github.com/highjoule/prediction_random_forest/blob/main/images/output_5_0.png)
    



```python
# The median and the mean is shown in the next code
fig, (ax1, ax2) = plt.subplots(1, 2,sharey=True,figsize=(10, 2))
ax1.plot(dmb.mean())
ax1.set_title('Balance mean')
ax2.plot(dmb.median())
ax2.set_title('Balance median')

for ax in fig.axes:
    ax.grid(b=True, which='major', color='#A79E9E', linestyle='-', linewidth=0.4)
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.sca(ax)
    plt.xticks(rotation=45)
plt.show()
```


    


![](https://github.com/highjoule/prediction_random_forest/blob/main/images/output_6_0.png)    



```python
#The dataframe is group by year
dy = df.resample('Y')
#Balance is chosen to calculate mean and median
dyb = dy['balance']
dyb.mean(),dyb.median()
```




    (activated_date
     2019-12-31    2099.681459
     2020-12-31    1342.353111
     Name: balance, dtype: float64,
     activated_date
     2019-12-31    1209.733106
     2020-12-31     723.893169
     Name: balance, dtype: float64)




```python
# The filter below gets the clients with an activated date and last payment date in 2020
df2.set_index('cust_id',inplace=True)
dates= (df2['activated_date']>'2020-01-01')&(df2['last_payment_date']>'2020-01-01')
df_2020 = df2.loc[dates]
```


```python
# We set a new index and remove the characters
new_index=df_2020.index.str.replace('C', '')
```


```python
df_2020.set_index(new_index,inplace=True)
df_2020
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activated_date</th>
      <th>last_payment_date</th>
      <th>balance</th>
      <th>cash_advance</th>
      <th>credit_limit</th>
    </tr>
    <tr>
      <th>cust_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12738</th>
      <td>2020-01-02</td>
      <td>2020-05-24</td>
      <td>623.955961</td>
      <td>2119.702403</td>
      <td>1200.0</td>
    </tr>
    <tr>
      <th>12739</th>
      <td>2020-01-02</td>
      <td>2020-09-14</td>
      <td>2123.483416</td>
      <td>0.000000</td>
      <td>7500.0</td>
    </tr>
    <tr>
      <th>12740</th>
      <td>2020-01-02</td>
      <td>2020-05-27</td>
      <td>939.348677</td>
      <td>0.000000</td>
      <td>1000.0</td>
    </tr>
    <tr>
      <th>12741</th>
      <td>2020-01-02</td>
      <td>2020-08-01</td>
      <td>1372.486104</td>
      <td>315.120089</td>
      <td>3000.0</td>
    </tr>
    <tr>
      <th>12742</th>
      <td>2020-01-02</td>
      <td>2020-08-11</td>
      <td>123.058670</td>
      <td>0.000000</td>
      <td>3500.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19186</th>
      <td>2020-07-31</td>
      <td>2020-11-03</td>
      <td>28.493517</td>
      <td>0.000000</td>
      <td>1000.0</td>
    </tr>
    <tr>
      <th>19187</th>
      <td>2020-07-31</td>
      <td>2020-09-06</td>
      <td>19.183215</td>
      <td>0.000000</td>
      <td>1000.0</td>
    </tr>
    <tr>
      <th>19188</th>
      <td>2020-07-31</td>
      <td>2020-06-03</td>
      <td>23.398673</td>
      <td>0.000000</td>
      <td>1000.0</td>
    </tr>
    <tr>
      <th>19189</th>
      <td>2020-07-31</td>
      <td>2020-07-19</td>
      <td>13.457564</td>
      <td>36.558778</td>
      <td>500.0</td>
    </tr>
    <tr>
      <th>19190</th>
      <td>2020-07-31</td>
      <td>2020-10-14</td>
      <td>372.708075</td>
      <td>127.040008</td>
      <td>1200.0</td>
    </tr>
  </tbody>
</table>
<p>6272 rows × 5 columns</p>
</div>



### In the next lines and prediction method with k-means, random forest and linear regression is analyzed


```python
#A new data frame is build based on the custumer ID
df3 = df0.set_index('cust_id')
df3
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activated_date</th>
      <th>last_payment_date</th>
      <th>balance</th>
      <th>balance_frequency</th>
      <th>purchases</th>
      <th>oneoff_purchases</th>
      <th>installments_purchases</th>
      <th>cash_advance</th>
      <th>purchases_frequency</th>
      <th>oneoff_purchases_frequency</th>
      <th>purchases_installments_frequency</th>
      <th>cash_advance_frequency</th>
      <th>cash_advance_trx</th>
      <th>purchases_trx</th>
      <th>credit_limit</th>
      <th>payments</th>
      <th>minimum_payments</th>
      <th>prc_full_payment</th>
      <th>tenure</th>
      <th>fraud</th>
    </tr>
    <tr>
      <th>cust_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C10001</th>
      <td>2019-10-06</td>
      <td>2020-09-09</td>
      <td>40.900749</td>
      <td>0.818182</td>
      <td>95.40</td>
      <td>0.00</td>
      <td>95.40</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>1000.0</td>
      <td>201.802084</td>
      <td>139.509787</td>
      <td>0.000000</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C10002</th>
      <td>2019-10-06</td>
      <td>2020-07-04</td>
      <td>3202.467416</td>
      <td>0.909091</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>6442.945483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>4</td>
      <td>0</td>
      <td>7000.0</td>
      <td>4103.032597</td>
      <td>1072.340217</td>
      <td>0.222222</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C10003</th>
      <td>2019-10-06</td>
      <td>2020-09-17</td>
      <td>2495.148862</td>
      <td>1.000000</td>
      <td>773.17</td>
      <td>773.17</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>12</td>
      <td>7500.0</td>
      <td>622.066742</td>
      <td>627.284787</td>
      <td>0.000000</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C10004</th>
      <td>2019-10-06</td>
      <td>2020-08-24</td>
      <td>1666.670542</td>
      <td>0.636364</td>
      <td>1499.00</td>
      <td>1499.00</td>
      <td>0.00</td>
      <td>205.788017</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>1</td>
      <td>1</td>
      <td>7500.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C10005</th>
      <td>2019-10-06</td>
      <td>2020-10-20</td>
      <td>817.714335</td>
      <td>1.000000</td>
      <td>16.00</td>
      <td>16.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1200.0</td>
      <td>678.334763</td>
      <td>244.791237</td>
      <td>0.000000</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>C19186</th>
      <td>2020-07-31</td>
      <td>2020-11-03</td>
      <td>28.493517</td>
      <td>1.000000</td>
      <td>291.12</td>
      <td>0.00</td>
      <td>291.12</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>6</td>
      <td>1000.0</td>
      <td>325.594462</td>
      <td>48.886365</td>
      <td>0.500000</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C19187</th>
      <td>2020-07-31</td>
      <td>2020-09-06</td>
      <td>19.183215</td>
      <td>1.000000</td>
      <td>300.00</td>
      <td>0.00</td>
      <td>300.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>6</td>
      <td>1000.0</td>
      <td>275.861322</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C19188</th>
      <td>2020-07-31</td>
      <td>2020-06-03</td>
      <td>23.398673</td>
      <td>0.833333</td>
      <td>144.40</td>
      <td>0.00</td>
      <td>144.40</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.000000</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0</td>
      <td>5</td>
      <td>1000.0</td>
      <td>81.270775</td>
      <td>82.418369</td>
      <td>0.250000</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C19189</th>
      <td>2020-07-31</td>
      <td>2020-07-19</td>
      <td>13.457564</td>
      <td>0.833333</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>36.558778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>2</td>
      <td>0</td>
      <td>500.0</td>
      <td>52.549959</td>
      <td>55.755628</td>
      <td>0.250000</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C19190</th>
      <td>2020-07-31</td>
      <td>2020-10-14</td>
      <td>372.708075</td>
      <td>0.666667</td>
      <td>1093.25</td>
      <td>1093.25</td>
      <td>0.00</td>
      <td>127.040008</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>2</td>
      <td>23</td>
      <td>1200.0</td>
      <td>63.165404</td>
      <td>88.288956</td>
      <td>0.000000</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>8950 rows × 20 columns</p>
</div>




```python

```


```python
#The dataframe has NaN values, therefore minimum payments is discarted in part of the analysis, thus ignoring this variable
df3.isnull().sum()
```




    activated_date                       13
    last_payment_date                     9
    balance                               2
    balance_frequency                     0
    purchases                             0
    oneoff_purchases                      0
    installments_purchases                0
    cash_advance                        112
    purchases_frequency                   0
    oneoff_purchases_frequency            0
    purchases_installments_frequency      0
    cash_advance_frequency                0
    cash_advance_trx                      0
    purchases_trx                         0
    credit_limit                          1
    payments                              0
    minimum_payments                    321
    prc_full_payment                      0
    tenure                                0
    fraud                                 0
    dtype: int64




```python
#column called minimum payments is discarted
df3 = df3.drop('minimum_payments', 1)
```


```python
df3.isnull().sum()
```




    activated_date                       13
    last_payment_date                     9
    balance                               2
    balance_frequency                     0
    purchases                             0
    oneoff_purchases                      0
    installments_purchases                0
    cash_advance                        112
    purchases_frequency                   0
    oneoff_purchases_frequency            0
    purchases_installments_frequency      0
    cash_advance_frequency                0
    cash_advance_trx                      0
    purchases_trx                         0
    credit_limit                          1
    payments                              0
    prc_full_payment                      0
    tenure                                0
    fraud                                 0
    dtype: int64




```python
df3.shape
```




    (8950, 19)




```python
#The remaining data is cleaned and no NaN values are included in the prediction model
df3_clean = df3.dropna()
df3_clean.isnull().sum()
```




    activated_date                      0
    last_payment_date                   0
    balance                             0
    balance_frequency                   0
    purchases                           0
    oneoff_purchases                    0
    installments_purchases              0
    cash_advance                        0
    purchases_frequency                 0
    oneoff_purchases_frequency          0
    purchases_installments_frequency    0
    cash_advance_frequency              0
    cash_advance_trx                    0
    purchases_trx                       0
    credit_limit                        0
    payments                            0
    prc_full_payment                    0
    tenure                              0
    fraud                               0
    dtype: int64




```python
#from an original data set with 8950 rows the new data is reduced to 8814 rows
df3_clean.shape
```




    (8814, 19)




```python
#x and y sets are stored to fit in the model
X = df3_clean.iloc[:,2:18].values
y = df3_clean['fraud'].values
```


```python
#the data is split in an train set, which will allow us to measure its performance
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```


```python
X_train.shape,X_test.shape,y_train.shape,y_test.shape
```




    ((6169, 16), (2645, 16), (6169,), (2645,))




```python
X_train[0,:]
```




    array([5.55131027e+02, 1.00000000e+00, 1.58645000e+03, 1.11160000e+03,
           4.74850000e+02, 0.00000000e+00, 1.00000000e+00, 4.16667000e-01,
           1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.10000000e+01,
           5.50000000e+03, 3.21928576e+03, 4.16667000e-01, 1.20000000e+01])



#### Random forest prediction method


```python
#In this secction Random fprest classifier is used to predict future values
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=5, n_estimators=20)

```


```python

model.fit(X_train, y_train)

```




    RandomForestClassifier(n_estimators=20, random_state=5)




```python
predicted = model.predict(X_test)

```


```python
#with features_importances we can know the importance of each variable
model.feature_importances_
```




    array([0.06776457, 0.00470264, 0.22117273, 0.13264667, 0.0505921 ,
           0.08567389, 0.0071573 , 0.02816976, 0.02632343, 0.03619133,
           0.03927568, 0.08795447, 0.03203428, 0.17072535, 0.00759797,
           0.00201782])




```python
#preparation of the variables names
df3_clean.columns[2:-1]
```




    Index(['balance', 'balance_frequency', 'purchases', 'oneoff_purchases',
           'installments_purchases', 'cash_advance', 'purchases_frequency',
           'oneoff_purchases_frequency', 'purchases_installments_frequency',
           'cash_advance_frequency', 'cash_advance_trx', 'purchases_trx',
           'credit_limit', 'payments', 'prc_full_payment', 'tenure'],
          dtype='object')




```python
#plot that shows the top 20 important features
feat_importances = pd.Series(model.feature_importances_, index=df3_clean.columns[2:-1])
feat_importances.nlargest(20).plot(kind='bar',figsize=(10,10))
plt.title("Top 20 important features")
plt.show()
```



![](https://github.com/highjoule/prediction_random_forest/blob/main/images/output_30_0.png)
    



```python
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score
print(f'Accuracy Score:\n{accuracy_score(y_test, predicted):0.3f}')

```

    Accuracy Score:
    0.996
    

### Bonus MiniBatch Means


```python
from sklearn.cluster import MiniBatchKMeans

# Define K-means model 
kmeans = MiniBatchKMeans(n_clusters=2, random_state=42).fit(X_train)

# Obtain predictions and calculate distance from cluster centroid
X_test_clusters = kmeans.predict(X_test)
X_test_clusters_centers = kmeans.cluster_centers_
dist = [np.linalg.norm(x-y) for x, y in zip(X_test, X_test_clusters_centers[X_test_clusters])]

# Create fraud predictions based on outliers on clusters 
km_y_pred = np.array(dist)
km_y_pred[dist >= np.percentile(dist, 95)] = 1
km_y_pred[dist < np.percentile(dist, 95)] = 0

```


```python
X_test.shape,km_y_pred.shape,X.shape,y.shape

```




    ((2645, 16), (2645,), (8814, 16), (8814,))




```python
X_test_clusters_centers[X_test_clusters].shape
```




    (2645, 16)




```python
#This method gives a roc_auc_score of 92.8%
roc_auc_score(y_test, km_y_pred)

```




    0.928095238095238




```python
#The next plt shows the distribution of the test data set and the predicted data in a scatter distributio
plt.scatter(X_test[km_y_pred == 0, 0], X_test[km_y_pred== 0, 15], label="Not Fraud", alpha=0.5, linewidth=0.15)
plt.scatter(X_test[km_y_pred == 1, 0], X_test[km_y_pred == 1, 15], label="Fraud", alpha=0.5, linewidth=0.15, c='r')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2aeefef79d0>




![](https://github.com/highjoule/prediction_random_forest/blob/main/images/output_37_1.png)
    

    



```python
#The next plt shows the distribution of the original data set and the clasification data in a scatter distribution
plt.scatter(X[y == 0, 0], X[y == 0, 15], label="Not Fraud", alpha=0.5, linewidth=0.15)
plt.scatter(X[y == 1, 0], X[y == 1, 15], label="Fraud", alpha=0.5, linewidth=0.15, c='r')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2aef02c9c70>




 
![](https://github.com/highjoule/prediction_random_forest/blob/main/images/output_38_1.png)   
    



```python

```


```python
from sklearn.metrics import confusion_matrix

#the values of the confusion matrix are below
cm = confusion_matrix(y_test,  km_y_pred)
cm
```




    array([[2510,  115],
           [   2,   18]], dtype=int64)



### Bonus Simple regression method


```python
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

```


```python
model = LinearRegression()

```


```python
model.fit(X_train, y_train)

```




    LinearRegression()




```python
model.coef_
```




    array([ 1.66899934e-06, -1.10210087e-02,  1.31814820e-04, -1.10319650e-04,
           -1.05122646e-04,  7.22581615e-06,  2.29670162e-02, -5.42940815e-02,
           -3.61222591e-02, -5.91127082e-02,  1.87112880e-03,  9.97778954e-05,
           -1.70119484e-06,  3.80687819e-06, -6.52439311e-03, -1.25234617e-03])




```python
y_predicted = model.predict(X_test)

```


```python
#The correlation value of r2 has a very poor performace in multi-variable problem like this
r2_score(y_test, y_predicted)

```




    0.24677778976214026




```python

```
