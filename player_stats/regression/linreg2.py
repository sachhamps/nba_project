"""
linreg2.py
~~~~~~~~~~

Linear regression of a players stats over a period of seasons,
this will be used to caluculate a prediction of statlines for players
in the 2018-19 seasons - Currently a temporary dataset is being used. This
dataset uses players stats per season over their careers. The real dataset
will contain the stats of every game of a player over the previous 3 seasons
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.spatial.distance import cdist
from numpy.linalg import inv
from sklearn.kernel_ridge import KernelRidge
from bs4 import BeautifulSoup
from urllib.request import urlopen
from pandas.plotting import scatter_matrix
from IPython.display import display
import seaborn as sns



#Scraping from Basketball reference for player statistics
# #Every game-log from the previous 3 seasons
full_player_data = pd.DataFrame()
for i in range(2016,2019):
	url = 'https://www.basketball-reference.com/players/d/duranke01/gamelog/{}'.format(i)
	html = urlopen(url)
	soup = BeautifulSoup(html, 'html.parser')
	column_headers = [th.getText() for th in 
			soup.findAll('th',{'class':'poptip'})]

	data_rows = soup.findAll('tr', {'id':['pgl_basic.{}'.format(x) for x in range (2000)]})
	player_data =[[td.getText() for td in data_rows[i].findAll('td')]
		for i in range(len(data_rows))]

	column_headers.pop(0)
	# print(column_headers)
	df = pd.DataFrame(player_data, columns=column_headers)
	df = df.drop(['Opp','Tm', '\xa0','\xa0'],axis=1)
	df = df[:].fillna(0)
	df['MP'] = df['MP'].str[:2]
	df['Age'] = df['Age'].str[:2]
	df['Date'] = df['Date'].str[:4]
	full_player_data = full_player_data.append(df, ignore_index=True)


# print(full_player_data)
# test_cols = ['MP','PTS','FG%','GmSc','+/-', 'Date']

# tf = full_player_data[test_cols].apply(pd.to_numeric, errors='coerce')

# sns_plot = sns.pairplot(tf)
# sns_plot.savefig("output.png")


#Advanced game-log from previous 3 seasons
full_player_data_adv = pd.DataFrame()
for i in range(2016,2019):
	url_adv = 'https://www.basketball-reference.com/players/d/hardeja01/gamelog-advanced/{}/'.format(i)
	html_adv = urlopen(url_adv)
	soup_adv = BeautifulSoup(html_adv, 'html.parser')
	column_headers = [th.getText() for th in 
							soup_adv.findAll('th',{'class' :'poptip'})]
	data_rows = soup_adv.findAll('tr', {'id':['pgl_advanced.{}'.format(x) for x in range(772)]})
	player_data = [[td.getText() for td in data_rows[i].findAll('td')] for i in range(len(data_rows))]
	

	column_headers.pop(0)
	df_adv = pd.DataFrame(player_data,columns=column_headers)
	df_adv = df_adv.drop(['Opp','Tm', '\xa0', '\xa0', 'G', 'MP', 'GmSc', 'GS', 'Date', 'Age'],axis=1)
	full_player_data_adv = full_player_data_adv.append(df_adv, ignore_index=True)

final_player_data = pd.concat([full_player_data,full_player_data_adv],axis=1)


test_cols = ['MP','PTS','FG%','GmSc','+/-', 'Date', 'eFG%','TS%','USG%', 'ORtg']
test = final_player_data[test_cols].apply(pd.to_numeric, errors='coerce')
#print(test.to_string())
sns_plot = sns.pairplot(test,hue='PTS')
sns_plot.savefig("output.png")



tdf.replace([['Houston Rockets','Toronto Raptors','Golden State Warriors','Utah Jazz',
 'Philadelphia 76ers','Oklahoma City Thunder','Boston Celtics',
 'San Antonio Spurs','Portland Trail Blazers','Minnesota Timberwolves'
 'Denver Nuggets' 'New Orleans Pelicans','Indiana Pacers',
 'Cleveland Cavaliers','Washington Wizards' 'Miami Heat',
 'Los Angeles Clippers' 'Charlotte Hornets','Detroit Pistons'
 'Milwaukee Bucks','Los Angeles Lakers','Dallas Mavericks',
 'New York Knicks','Brooklyn Nets','Orlando Magic','Atlanta Hawks',
 'Memphis Grizzlies','Chicago Bulls','Sacramento Kings','Phoenix Suns'], ['HOU', 'TOR', 'UTA','PHI', 'OKC', 'BOS', 'SAS', 'POR', 'MIN',
  'DEN', 'NOP', 'IND', 'CLE', 'WAS', 'MIA', 'LAC', 'CHO', 'DET', 'MIL', 'LAL', 'DAL', 'NYK', 'BRK', 'ORL', 'ATL', 'MEM', 'CHI', 'SAC', 'PHO']])


