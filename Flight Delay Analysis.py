#!/usr/bin/env python
# coding: utf-8

# In[109]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# In[2]:


df = pd.read_csv(r"E:\Data Storage\Flight\flights.csv" , low_memory = False)


# In[3]:


df.columns


# In[82]:


df['MONTH'].unique()


# In[4]:


pd.set_option('display.max_columns', None)
df.head()


# In[5]:


dimf1 = pd.read_csv(r"E:\Data Storage\Flight\airports.csv")


# In[6]:


dimf1.head()


# In[7]:


dimf1.shape


# In[8]:


dimf1["COUNTRY"].unique()


# In[9]:


sttxct = dimf1[['CITY','STATE']].drop_duplicates()


# In[10]:


sttxct.to_csv("E:\Data Storage\Flight\state_x_city.csv", index = False)


# In[11]:


dimf2 = pd.read_csv(r"E:\Data Storage\Flight\airlines.csv")


# In[12]:


dimf2.head()


# In[13]:


dimf2.shape


# In[14]:


df.shape


# In[85]:


df["DATE"] = pd.to_datetime(df[['YEAR','MONTH','DAY']])


# In[16]:


df["DAY_OF_WEEK_UPTD"] = ["Monday" if t == 1 else
                         "Tuesday" if t ==2 else
                         "Wednesday" if t==3 else
                         "Thursday" if t ==4 else
                         "Friday" if t ==5 else
                         "Saturday" if t == 6 else
                         "Sunday" for t in df["DAY_OF_WEEK"]]


# In[17]:


df.head()


# In[34]:


Raw_df = df[['DATE','DAY_OF_WEEK_UPTD', 'AIRLINE', 'FLIGHT_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
       'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
       'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
       'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
       'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
       'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
       'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]


# In[35]:


Raw_df.head()


# In[36]:


Tiime_zn_mppng = pd.read_csv(r"E:\Data Storage\Flight\US_Cities_Time_Offsets.csv")


# In[37]:


Tiime_zn_mppng.head()


# In[38]:


Raw_df["ORIGIN_AIRPORT"].nunique()


# In[39]:


Raw_df = Raw_df[df['DESTINATION_AIRPORT'].str.match(r"^[A-Z]{3}$", na=False) &
               df['ORIGIN_AIRPORT'].str.match(r"^[A-Z]{3}$", na=False)]


# In[40]:


Raw_df.shape


# In[41]:


dimf3 = dimf1.merge(Tiime_zn_mppng, left_on = ["CITY", "STATE"],
                    right_on = ["City", "State"], how = "left")


# In[42]:


dimf3.columns


# In[43]:


dimf3 = dimf3[['IATA_CODE', 'AIRPORT', 'CITY', 'STATE', 'COUNTRY', 'LATITUDE',
       'LONGITUDE','Time Offset (hrs from ET)']]


# In[44]:


dimf3.head()


# In[45]:


Raw_df = Raw_df.merge(dimf3, left_on = ['ORIGIN_AIRPORT'], right_on = ['IATA_CODE'], how = "left")


# In[46]:


Raw_df.head()


# In[47]:


Raw_df = Raw_df[['DATE','DAY_OF_WEEK_UPTD', 'AIRLINE', 'FLIGHT_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
       'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
       'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
       'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
       'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
       'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
       'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'Time Offset (hrs from ET)']]


# In[48]:


Raw_df.rename(columns = {"Time Offset (hrs from ET)": "TIME_OFFSET_ORIGIN"}, inplace = True)


# In[49]:


Raw_df.head()


# In[50]:


Raw_df = Raw_df.merge(dimf3, left_on = ['DESTINATION_AIRPORT'], right_on = ['IATA_CODE'], how = "left")


# In[51]:


Raw_df = Raw_df[['DATE','DAY_OF_WEEK_UPTD', 'AIRLINE', 'FLIGHT_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
       'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
       'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
       'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
       'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
       'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
       'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'TIME_OFFSET_ORIGIN','Time Offset (hrs from ET)']]


# In[52]:


Raw_df.rename(columns = {"Time Offset (hrs from ET)": "TIME_OFFSET_DESTINY"}, inplace = True)


# In[53]:


Raw_df.head()


# In[54]:


Raw_df["TIMEZONE_DIFF"] = -1*(Raw_df["TIME_OFFSET_ORIGIN"] - Raw_df["TIME_OFFSET_DESTINY"])


# In[55]:


Raw_df.head()


# In[56]:


Raw_df = Raw_df[['DATE','DAY_OF_WEEK_UPTD', 'AIRLINE', 'FLIGHT_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
       'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
       'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
       'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
       'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
       'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
       'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'TIMEZONE_DIFF']]


# In[57]:


Raw_df.head()


# In[58]:


Raw_df['WEEKEND_FLG'] = [1 if t == "Saturday" or t == "Sunday" else
                              0 for t in Raw_df["DAY_OF_WEEK_UPTD"]]


# In[59]:


Raw_df = Raw_df[['DATE','DAY_OF_WEEK_UPTD', 'WEEKEND_FLG','AIRLINE', 'FLIGHT_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
       'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
       'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
       'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
       'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
       'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
       'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'TIMEZONE_DIFF']]


# In[60]:


Raw_df.head()


# In[61]:


def get_season(month): 
    if month in [3,4,5]:
        return "Spring"
    elif month in [6,7,8]:
        return "Summer"
    elif month in [9,10,11]:
        return "Fall/Autumn"
    elif month in [12,1,2]:
        return "Winter"


# In[62]:


Raw_df["SEASON"] = Raw_df['DATE'].dt.month.apply(get_season)


# In[63]:


Raw_df = Raw_df[['DATE','DAY_OF_WEEK_UPTD', 'WEEKEND_FLG','SEASON','AIRLINE', 'FLIGHT_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
       'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
       'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
       'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
       'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
       'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
       'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'TIMEZONE_DIFF']]


# In[64]:


Raw_df.rename(columns = {"AIRLINE": "AIRLINE_CODE"}, inplace = True)


# In[65]:


delay_features = [
    'DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY',
    'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'TAXI_OUT', 'TAXI_IN',
    'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE']


# In[66]:


corr = Raw_df[delay_features].corr()


# In[67]:


sns.heatmap(corr, annot = True, fmt=".2f", cmap = 'coolwarm')
plt.show()


# In[68]:


Raw_df = Raw_df.merge(dimf2 , left_on = ['AIRLINE_CODE'], right_on = ['IATA_CODE'], how = 'left')


# In[69]:


Raw_df.head()


# In[70]:


Raw_df.rename(columns = {'AIRLINE': 'AIRLINE_NAME'}, inplace = True)
#Raw_df.rename(columns = {'AIRLINE_x': 'AIRLINE_CODE'}, inplace = False)


# In[71]:


Raw_df = Raw_df[['DATE','DAY_OF_WEEK_UPTD', 'WEEKEND_FLG','SEASON','AIRLINE_CODE','AIRLINE_NAME', 'FLIGHT_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
       'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
       'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
       'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
       'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
       'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
       'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'TIMEZONE_DIFF']]


# In[72]:


Raw_df.head()


# In[97]:


Pivot_Airline = Raw_df.groupby('AIRLINE_NAME').agg(mean_departure_delay =('DEPARTURE_DELAY','mean'),
                                   flight_count = ('DEPARTURE_DELAY', 'count')
                                  ).sort_values(by = 'flight_count',ascending = False)
Pivot_Airline['mean_departure_delay']=Pivot_Airline['mean_departure_delay'].round(2)
ttl_flt = Pivot_Airline['flight_count'].sum()
Pivot_Airline= Pivot_Airline.reset_index()
Pivot_Airline["% Distribution"] = ((Pivot_Airline['flight_count']/ttl_flt)*100).round(2).astype(str)+'%'
print(Pivot_Airline)


# In[90]:


Pivot_Month = Raw_df.groupby(Raw_df['DATE'].dt.month).agg(mean_departure_delay =('DEPARTURE_DELAY','mean'),
                                   flight_count = ('DEPARTURE_DELAY', 'count')
                                  ).sort_values(by = 'DATE', ascending = True)
Pivot_Month['mean_departure_delay']=Pivot_Month['mean_departure_delay'].round(2)
Pivot_Month = Pivot_Month.reset_index() 
Pivot_Month.rename(columns = {'DATE':'Month'}, inplace = True)
print(Pivot_Month)


# In[99]:


Pivot_Month.columns


# In[100]:


Pivot_Month['flight_count'].sum()


# In[101]:


Raw_df['DEPARTURE_DELAY'].isnull().sum()


# In[102]:


Pivot_Airline.columns


# In[103]:


plt.figure(figsize=(12,6))
plt.bar(Pivot_Airline['AIRLINE_NAME'], Pivot_Airline['mean_departure_delay'], color = 'skyblue')
plt.xticks(rotation = 45, ha = 'right')
plt.xlabel('AIRLINE')
plt.ylabel('Avg. departure delay (in min.)')
plt.title('Avg. Departure Delay by Airline')
plt.tight_layout()
plt.show()


# **Hypothesis Testing :**
# *Every Airlines has identical delay time*

# In[105]:


Tst_df = Raw_df[['AIRLINE_NAME', 'DEPARTURE_DELAY']].dropna()
Tst_df.head()


# In[106]:


Tst_df.shape


# In[107]:


Tst_df['AIRLINE_NAME'].nunique()


# In[111]:


groups = [group["DEPARTURE_DELAY"].values for name, group in Tst_df.groupby("AIRLINE_NAME")]
print(groups)


# In[112]:


f_stat, p_value = stats.f_oneway(*groups)


# In[114]:


print('f-statistics :', f_stat)
print('p-value :', p_value)


# *P-Value is ~ 0 which signifies the statement considered is very wrong and all the airlines have significantly different time delay*

# In[130]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[116]:


Raw_df['DELAY_FLG'] = Raw_df['DEPARTURE_DELAY'].apply(lambda x:1 if x>15 else 0)


# In[117]:


Raw_df.head()


# In[118]:


Raw_df.columns


# In[133]:


features = ['DAY_OF_WEEK_UPTD','WEEKEND_FLG','SEASON', 'AIRLINE_NAME', 'ORIGIN_AIRPORT',
       'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE']


# In[134]:


df_encoded = Raw_df[features + ['DELAY_FLG']].copy()
label_encoders = {}


# In[135]:


for col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le


# In[136]:


X = df_encoded[features]
y = df_encoded['DELAY_FLG']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[137]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

