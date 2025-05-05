# general packages 
import sys
import glob

import pandas as pd
import xarray as xr
import numpy as np

# name of run
RUNNAME = 'FU_project3'

FOLDER = # input folder
OFOLDER =  # output folder
RUNS = ["h{:03d}".format(number) for number in range(10, 170)]

# country selection
COUNTRIES = ['NRD', 'BRI', 'NWE', 'CEN', 'IBA', 'BAL', 'EAS']# countries or regions 

var = 'residual_pvwind'
q_set = 0.97
nr_of_events = 160
season = 'winter'# 'all' #'all', 'winter', 'summer'
dimension_name='region'

def group_as_event(df, daygap=2):
    """
    give all events in a run a number. Consider days within percentile that are less 
    than 2 days apart as the same event 

    parameters
    ----------
    group (pd.group): a grouped panda series

    returns
    -------
    an index for all consequative date with a max of daygap days apart
    """
    dt = df
    day = pd.Timedelta(f'{daygap}d')
    breaks = (dt.diff() > day)
    groups = breaks.cumsum()+1
    return groups


df_seasons = {}
all_events = {}

for season in ['winter', 'summer', 'all']:
    for country in COUNTRIES:
        ds = xr.open_dataset(FOLDER + f'{country}_{RUNNAME}.nc')
        if var == 'residual_pvwind':
            ds[var] = ds['demand'] - ds['pv_util']-ds['wind_offshore']-ds['wind_onshore']
    
        ds = ds[[var]]
        df = ds.isel(region=0).drop(dimension_name).to_dataframe()
        mean_per_doy = df.reset_index().groupby(df.reset_index().time.dt.dayofyear).mean()
        list_for_plot = []

        dft = df.reset_index()
        
        # cut per season
        if season == 'summer':
            dft = dft.loc[dft.time.dt.dayofyear > 152].loc[dft.time.dt.dayofyear <272]   
        if season == 'winter':
            dft0 = dft.loc[dft.time.dt.dayofyear <= 152]
            dft1 = dft.loc[dft.time.dt.dayofyear >= 272]
            dft = pd.concat([dft0, dft1])

        # select all days within defined percentile q
        threshold = dft[var].quantile(q_set)
        dft = dft.loc[dft[var]>threshold]
        # give all events in a run a number. 
        # Consider days within percentile that are less than x days apart as the same event 
        dft['event_of_run'] = dft.groupby('runs')['time'].transform(group_as_event)
        # also include the days below q097
        dft3 = pd.concat([
            dft.groupby(['runs','event_of_run']).time.min(), 
            dft.groupby(['runs','event_of_run']).time.max()], axis=1)
        dft3.columns = ['start', 'end']

        dflist = []
        dfkeys = []
        for r in set(dft3.reset_index().runs):
            fullevent = [df.loc[r].loc[dft3.loc[r].loc[i].start:dft3.loc[r].loc[i].end] for i in dft3.loc[r].index]
            dflist.append(pd.concat(fullevent, keys=range(1,len(fullevent)+1)).reset_index().set_index('time'))
            dfkeys.append(r)
        dft4 = pd.concat(dflist, keys = dfkeys)
        dft4 = dft4.rename({'level_0':'event_of_run'}, axis=1)
        dft4 = dft4.reset_index()
        dft4 = dft4.rename({'level_0':'runs'}, axis=1)

        
        # index the events over all the runs (not per run)
        dft['event_nr'] = dft.groupby(['runs', 'event_of_run']).ngroup()
        # take the mean residual per event
        dft[f'event_mean_{var}'] = dft.groupby('event_nr')[var].transform('mean')
        dft[f'event_total_{var}'] = dft.groupby('event_nr')[var].transform('sum')
        # assign event_nr based on level oof mean residual load
        dft = dft.sort_values([f'event_total_{var}', 'time'], ascending=[False, True])
        dft['event_nr'] = dft.groupby(['runs', 'event_of_run'], sort=False).ngroup()
        dft['event_nr'] = dft['event_nr']+1
        # count the number of days an event lasts
        dft['nr_of_days'] = dft.groupby('event_nr')[[var]].transform('count')
        # make a seperate dataset of the nr of days that the events last
        nr_of_days = dft.groupby('event_nr').count()[[var]]  

        # only select n-events
        dft = dft.loc[dft.event_nr<(nr_of_events)+1]

        # get 
        data_per_event = dft.groupby(['event_nr']).first()[['time',f'event_total_{var}', f'event_mean_{var}', 'nr_of_days']]
        data_per_event = data_per_event.rename({'time': 'first_day'}, axis=1)
        events = dft 

        events['month'] = events.time.dt.month
        events['doy'] = events.time.dt.dayofyear
        events['month'] = events.groupby('event_nr')['month'].transform(lambda x: x.mode().iloc[0])
        events['week'] = events.time.dt.isocalendar().week
        events['week'] = events.groupby('event_nr')['week'].transform(lambda x: x.mode().iloc[0])
        meandoy = mean_per_doy.loc[events.time.dt.dayofyear][var].values
        events['anom'] = events[var] - meandoy
        events['rel_anom'] = (events[var] - meandoy)/meandoy
        all_events[country] = events
        
    all_mean_event = []
    for country in COUNTRIES:
        events = all_events[country].loc[all_events[country].event_nr<(nr_of_events)+1]
        mean_event_data = events.groupby('event_nr').first()
        mean_event_data.loc[:, ['anom', 'rel_anom']] = events.groupby('event_nr')[['anom', 'rel_anom']].mean()
        mean_event_data = mean_event_data.rename({'anom': 'anom_mean', 'rel_anom':'rel_anom_mean'})
        mean_event_data['country']  = country
        all_mean_event.append(mean_event_data)
    all_mean_event = pd.concat(all_mean_event).reset_index()
    looplist = []
    for country in COUNTRIES:
        events = all_events[country].loc[all_events[country].event_nr<(nr_of_events)+1]
        events['country'] = country
        looplist.append(events)
    df_all_events = pd.concat(looplist).reset_index()
    df_all_events.to_csv(f'{OFOLDER}{var}_q{q_set}_{season}_daygap3_{RUNNAME}.csv')