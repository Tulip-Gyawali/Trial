# src/features/extract_p_wave_feature.py
import os, random
from typing import Optional, List
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from tqdm import tqdm
from src.utils.helpers import p_wave_features_calc
import numpy as np

class PWaveExtractor:
    def __init__(self, client=None):
        self.client = client or Client("IRIS")

    def extract_from_timewindow(self, starttime: UTCDateTime, endtime: UTCDateTime, net="IU", stations: Optional[List[str]]=None, channel="BHZ"):
        stations = stations or ["ANMO","COR","MAJO","KBL"]
        st = None
        for station in stations:
            try:
                st = self.client.get_waveforms(net, station, "*", channel, starttime, endtime)
                if st and len(st) > 0:
                    break
            except Exception:
                st = None
                continue
        if not st:
            return None
        tr = st[0].copy()
        try:
            tr.detrend('demean')
            tr.filter('bandpass', freqmin=0.5, freqmax=20.0)
        except Exception:
            pass
        dt = tr.stats.delta
        try:
            cft = classic_sta_lta(tr.data, int(max(1, int(1/dt))), int(max(1, int(10/dt))))
            trig = trigger_onset(cft, 2.5, 1.0)
        except Exception:
            trig = []
        if len(trig) == 0:
            return None
        p_index = trig[0][0]
        win = int(max(1, int(2.0 / dt)))
        p_window = tr.data[p_index: p_index + win]
        if len(p_window) < 5:
            return None
        feats = p_wave_features_calc(p_window, dt)
        feats.update({"station": tr.stats.station, "network": tr.stats.network, "starttime": str(starttime), "sampling_rate": tr.stats.sampling_rate})
        return feats

def generate_random_starttime(years=(2022,2023,2024)):
    year = random.choice(list(years))
    month = random.randint(1,12)
    day = random.randint(1,28)
    hour = random.randint(0,23)
    return UTCDateTime(pd.Timestamp(year=year, month=month, day=day, hour=hour).to_pydatetime())

def batch_extract(out_csv="Data/EEW_features_2024-10-21.csv", num_samples=10, stations=None, net="IU"):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    extractor = PWaveExtractor()
    records=[]
    for _ in tqdm(range(num_samples)):
        start = generate_random_starttime()
        end = start + 2*3600
        try:
            feats = extractor.extract_from_timewindow(start, end, net=net, stations=stations)
            if feats:
                records.append(feats)
        except Exception:
            continue
    if not records:
        return pd.DataFrame(records)
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    return df
