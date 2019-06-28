"""
Created on Wed Jun 13 17:19:23 2018
@author: msdogan

This module retrieves inflow data from CDEC for defined station IDs.
Detailed info: 
documentation: http://ulmo.readthedocs.io/en/latest/
GitHub repo: https://github.com/ulmo-dev/ulmo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib, time, datetime
# from ulmo import cdec

# # get all available stattion info
# stations = pd.DataFrame(cdec.historical.get_stations()).to_csv('CDEC_stations.csv',index=True)

# # get all available sensor info
# sensors = pd.DataFrame(cdec.historical.get_sensors()).to_csv('CDEC_sensors.csv',index=False)

station_IDs = [ 
				'SHA', # 'Shasta'
                # 'KES', # 'Keswick'
                # 'ORO', # 'Oroville'
                'BUL', # 'Bullards Bar'
                # 'ENG', # 'Englebright'
                'FOL', # 'Folsom'
                # 'NAT', # 'Nimbus'
                'NML', # 'New Melones'
                # 'DNP', # 'Don Pedro'
                # 'EXC', # 'New Exchequer'
                'PNF'  # 'Pine Flat'
              ]

sensor_ID = 76 # inflow
start_date='2009-12-31'
end_date='2018-09-01'
resolution = 'hourly' # Possible values are 'event', 'hourly', 'daily', and 'monthly'
duration_code = 'H'
conversion = 0.028316847 # convert ft3/s to m3/s

# dat = cdec.historical.get_data(['SHA'],resolutions=['monthly'],sensor_ids=[sensor_ID])

# df = pd.DataFrame()
# for station_ID in station_IDs:
# 	print('retrieving station: '+station_ID)
# 	data = cdec.historical.get_data(
# 									station_ids=[station_ID], # check out station ID above
# 									sensor_ids=[sensor_ID], # check out sensor ID above
# 									resolutions=[resolution],
# 									start=start_date, # define start date, default is None
# 									end=end_date # define end date, default is None
# 									)
# 	# organize retrieved dictionary and convert to pandas Data Frame (convert negative values to positive and interpolate missing values)
# 	data = pd.DataFrame(data[station_ID][data[station_ID].keys()[0]]).abs().interpolate()*conversion
# 	data.columns = [station_ID]
# 	df[station_ID] = data[station_ID]

# df['KES'] = 0 # downstream of SHA
# df['NAT'] = 0 # downstream of FOL
# # save data
# df.to_csv('inflow_cms2.csv')


# # ****** Mustafa's Ulmo ******

# ex_url = 'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=SHA&SensorNums=76&dur_code=H&Start=2009-06-06&End=2018-09-06'


# for station_ID in station_IDs:
#     print('retrieving station: '+station_ID)
#     url = 'http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?'+'Stations='+str(station_ID)+'&SensorNums='+str(sensor_ID)+'&dur_code='+str(duration_code)+'&Start='+str(start_date)+'&End='+str(end_date)
#     print('url: '+url)
#     web = urllib.urlopen(url)
#     s = web.read()
#     web.close()
#     ff = open(str(station_ID)+"_CDEC.csv", "w")
#     ff.write(s)
#     ff.close()
#     time.sleep(6)


# daterange = pd.date_range(start=start_date,end=end_date,freq=duration_code)
# print(daterange)


# for station_ID in station_IDs:
#     print('organizing station: '+station_ID)
#     df = pd.read_csv(str(station_ID)+"_CDEC.csv",index_col=0,header=0)
#     df.index = pd.to_datetime(df.index)

# df = pd.read_csv('inflow_cms1.csv',index_col=0,header=0)
# df.index = pd.to_datetime(df.index)

# # look after outliars and correct if necessary
# print(df.describe())

# key = df.keys()[-1]
# print(key)
# print(df.loc[lambda df: df[key] > df[key].quantile(.9999), :])


df = pd.read_csv("CDEC_data_cfs.csv",header=0,index_col=0)
# df = pd.to_numeric(df.values,errors='coerce')
d = pd.DataFrame()
for key in df.keys():
    d[key]=pd.to_numeric(df[key],errors='coerce')
d = d.abs().interpolate()*conversion
d.to_csv('CDEC_data_cms.csv')

# key = d.keys()[1]
# print(key)
# print(d.loc[lambda d: d[key] > d[key].quantile(.9999), :])

fig = plt.figure(figsize=(5,4)); ax = plt.gca()
d.boxplot(ax=ax, sym='*',showmeans=True,showfliers=False)
plt.xticks(np.arange(1,len(d.keys())+1),['Shasta','Folsom','New Melones','Pine Flat'])
plt.title('Hourly Average Reservoir Inflow ($m^3/s$)',loc='left',fontweight='bold')
plt.tight_layout()
plt.savefig('inflow_cms.pdf',transparent=False)
plt.show()
    


