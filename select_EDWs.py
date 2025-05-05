import xarray as xr
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Constants
RUNNAME = 'FU_project3'
FOLDER = # input folder
OFOLDER = # output folder
# RUNS = ["h{:03d}".format(number) for number in range(10, 170)]
VAR0 = 'residual' #'residual'  # residual_pvwind, residual, netto_demand
COUNTRIES = ['NRD', 'BRI', 'NWE', 'CEN', 'IBA', 'BAL', 'EAS']# countries or regions ['allregions']#['allregions']#
NR_OF_EVENTS = 1600
SEASONS = ['DJF', 'MAM', 'JJA', 'SON', 'all']#['summer', 'winter']#['all', 'summer', 'winter'] #
EVENT_LENGTHS = [30]#[7,30, 60]#[1,7, 30, 60]

def open_energy_dataset(country, folder):
    filename = os.path.join(folder, f'{country}_{RUNNAME}.nc')
    if country == 'allregions':
        filename = filename.replace('/per_region/', '/total/')
    # Open the energy dataset into a variable
    return xr.open_dataset(filename)

def select_season_data(ds, season):
    # make subset of data for seasons
    if season == 'JJA':
        # return ds.where(ds.time.dt.month.isin([6, 7, 8, 9]), drop=True)
        return ds.where(ds.time.dt.month.isin([6, 7, 8, ]), drop=True)
    elif season == 'DJF':
        # return ds.where(ds.time.dt.month.isin([1, 2, 3, 4, 5, 10, 11, 12]), drop=True)
         return ds.where(ds.time.dt.month.isin([12, 1, 2]), drop=True)
    elif season == 'MAM':
        return ds.where(ds.time.dt.month.isin([3,4,5]), drop=True)
    elif season == 'SON':
        return ds.where(ds.time.dt.month.isin([9,10,11]), drop=True)
    else:
        return ds

def process_events(ds, var0, event_length, nr_of_events):
    # initialize lists
    events = []
    ts_for_safe = []
    # start with a full dataset
    if var0 == 'residual_pvwind':
        ds[var0] = ds['demand'] - ds['pv_util']-ds['pv_roof']-ds['wind_offshore']-ds['wind_onshore']
    
    t0 = ds[var0]
    # to prevent overlap, we loop over events and remove the highest one before moving to the next
    for i in tqdm(range(nr_of_events)):
        # take the rolling sum of the still available data 
        t0_rol = t0.rolling(time=event_length).sum()
        # get the maximum event
        t0_rol_max = t0_rol.where(t0_rol == t0_rol.max(), drop=True)
        # get the run and timestamps during which the event happens
        t0_max_run = t0_rol_max.runs.values[0]
        t0_maxtimestamp = t0_rol_max.time.values[0]
        t0_mintimestamp = t0_maxtimestamp - np.timedelta64(event_length-1, 'D')
        # make this into a timeslice and save into a list
        t0_maxtimeslice = slice(t0_mintimestamp, t0_maxtimestamp)
        events.append((t0_max_run, t0_maxtimeslice))
        ts_for_safe.append((t0_max_run, t0_mintimestamp, t0_maxtimestamp))
        # select the event from the dataset (not rolling)
        t0_event = ds[var0].sel(time=t0_maxtimeslice, runs=t0_max_run).expand_dims('runs')
        t0_event.name = 'residual_event'
        # merge into event dataset so i can substract 
        ds0 = xr.merge([t0, t0_event])
        # subtract and remove where it is the same
        t1 = t0.where((ds0[var0] - ds0['residual_event']) != 0)
        # this is the new dataset
        t0 = t1.copy()
    return events, ts_for_safe

def main():
    for season in SEASONS:
        for event_length in EVENT_LENGTHS:
            dflist = []
            for country in COUNTRIES:
                print(f'Selecting {NR_OF_EVENTS} {event_length}-day events for {country}')
                ds = open_energy_dataset(country, FOLDER)
                ds_season = select_season_data(ds, season)
                events, ts_for_safe = process_events(ds_season, VAR0, event_length, NR_OF_EVENTS)

                # save all event-dates into a dataframe and in a csv file
                df = pd.DataFrame(ts_for_safe, columns=['runs', 'ts0', 'tsn'])
                df['country'] = country
                dflist.append(df)
            dfa = pd.concat(dflist).reset_index()
            dfa = dfa.rename({'index':'event_nr'}, axis=1)

            dfa['event_nr'] = dfa['event_nr']+1
            if country == 'allregions':
                ofile = f'{OFOLDER}{VAR0}_el{event_length}_{season}_{RUNNAME}_{country}.csv'
            else: 
                ofile = f'{OFOLDER}{VAR0}_el{event_length}_{season}_{RUNNAME}.csv'
            dfa.to_csv(ofile)
            print(ofile)
        

if __name__ == "__main__":
    main()
