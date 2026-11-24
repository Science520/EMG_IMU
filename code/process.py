"""Process the Standing Lab native ADC stream on its sample-index timebase."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import signal, ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
IDS = {'112801':'P1', '113314':'P2', '120813':'P3-abort', '121156':'P3'}
MUSCLES = ['臀中肌','股直肌','股外侧肌','股内侧肌','股二头肌','胫骨前肌','腓肠肌','比目鱼肌']
DIR = ['left','right','forward','backward']


def events(folder):
    starts, ends = [], []
    with open(folder/'events.jsonl') as handle:
        for line in handle:
            e = json.loads(line)
            if e['type']=='phase_start' and e['label'] in DIR:
                starts.append(e)
            if e['type']=='phase_end' and e['reason']=='elapsed':
                ends.append(e)
    keys = ['trial','phase_index']
    s = pd.DataFrame(starts).merge(pd.DataFrame(ends)[keys+['host_s']], on=keys,
                                    suffixes=('_start','_end'), validate='one_to_one')
    csv = pd.read_csv(folder/'trials.csv')
    s = s.merge(csv[['trial','phase_id','issued_host_s','label']], on=['trial','phase_id','label'],validate='one_to_one')
    assert np.allclose(s.issued_host_s,s.host_s_start,rtol=0,atol=1e-6)
    return s[['trial','label','host_s_start','host_s_end']]


def native(folder,anchor):
    files=sorted(folder.glob('native/emg/chunks/*.npz'))
    with np.load(files[-1]) as z:
        length=int(z['index'].max())+1
    adc=np.full((16,length),np.nan,dtype='float32')
    clocks=[]
    for p in files:
        with np.load(p) as z:
            idx=z['index'].astype(int)
            tags=z['tags']
            assert np.all(z['segment']==1)
            adc[tags,idx]=z['adc']
            clocks.append((idx[-1],z['rx_ns'][-1]*1e-9+anchor))
    clock=np.array(clocks)
    slope,offset=np.polyfit(clock[:,0],clock[:,1],1)
    residual=clock[:,1]-(offset+slope*clock[:,0])
    return adc,offset+slope*np.arange(length),1/slope,np.percentile(abs(residual),95)


def filter_plan(x,fs):
    # Aggregate 2-second spectra across the recording, omitting incomplete windows.
    blocks=x[:len(x)//(2*round(fs))*(2*round(fs))].reshape(-1,2*round(fs))
    blocks=blocks[np.isfinite(blocks).all(axis=1)]
    if not len(blocks):
        return [], dict(low_fraction=np.nan,high_fraction=np.nan,line_db=np.nan,line_fraction=np.nan,highpass=0,lowpass=0,notch=False)
    f,p=signal.periodogram(blocks,fs,window='hann',axis=1)
    p=np.median(p,axis=0)
    total=max(p.sum(),1e-30)
    upper=min(450,0.45*fs)
    low=p[(f>0)&(f<20)].sum()/total
    high=p[f>upper].sum()/total
    line=abs(f-50)<=1
    flank=(abs(f-50)>=3)&(abs(f-50)<=8)
    db=10*np.log10(max(p[line].mean(),1e-30)/max(p[flank].mean(),1e-30))
    filters=[]
    for use,hz,kind in [(low>.05,20,'highpass'),(high>.05,upper,'lowpass')]:
        if use:
            filters.extend(signal.butter(4,hz,btype=kind,fs=fs,output='sos'))
    fraction=p[line].sum()/total
    notch=db>=8 and fraction>=.01
    if notch:
        filters.extend(signal.tf2sos(*signal.iirnotch(50,30,fs)))
    return filters,dict(low_fraction=low,high_fraction=high,line_db=db,line_fraction=fraction,
                        highpass=20 if low>.05 else 0,lowpass=upper if high>.05 else 0,notch=notch)


def longest_flat(a):
    edges=np.r_[0,np.where(np.diff(a)!=0)[0]+1,len(a)]
    return np.diff(edges).max()


def session(folder):
    sid=IDS[folder.name.split('-')[1]]
    meta=json.loads((folder/'calibration/emg_sdk_metadata.json').read_text())
    stream=json.loads((folder/'streams.json').read_text())['emg/ch_01']
    clock=stream['sdk_clock_anchor']
    adc,t,fs,jitter=native(folder,clock['platform_perf_counter_s']-clock['sdk_monotonic_s'])
    assert abs(fs/meta['sample_rate']-1)<.02 and meta['calibration_unit']=='mV'
    stages=events(folder)
    print(sid,adc.shape,round(fs,3),'Hz',len(stages),'shift events',flush=True)
    rows,qc,waves=[],[],[]
    for ch,cal in enumerate(sorted(meta['channels'],key=lambda v:v['tag'])):
        counts=adc[ch]
        x=(counts.astype(float)-cal['baseline'])*cal['gain']
        filters,plan=filter_plan(x,fs)
        info=dict(subject=sid,channel=ch+1,muscle=('左' if ch<8 else '右')+MUSCLES[ch%8])
        qc.append(dict(**info,**plan,samples=len(x),missing=int(np.isnan(x).sum()),
                       rail_samples=int(np.sum((counts==0)|(counts==2**24-1))),
                       longest_flat_s=longest_flat(counts)/fs,fs_hz=fs,clock_residual_p95_s=jitter))
        lp=signal.butter(2,5,fs=fs,output='sos')
        for e in stages.itertuples():
            start,end=e.host_s_start,e.host_s_end
            a,b,c,d=np.searchsorted(t,[start-.5,start,end,end+.5])
            seg=x[a:d]
            row=dict(**info,trial=e.trial,direction=e.label,start_host_s=start,end_host_s=end)
            reasons=[]
            if sid.endswith('abort'): reasons.append('aborted_session')
            if a==0 or d==len(t) or not np.isfinite(seg).all(): reasons.append('coverage')
            if len(seg):
                if longest_flat(seg)/fs>=.05: reasons.append('flat_50ms')
                if np.any((counts[a:d]==0)|(counts[a:d]==2**24-1)): reasons.append('ADC_rail')
                diff=np.diff(seg)
                scale=stats.median_abs_deviation(diff,scale='normal')
                row['spike_fraction']=np.mean(abs(diff-np.median(diff))>12*max(scale,1e-12))
                if row['spike_fraction']>.01: reasons.append('spike_burden')
            row['exclude']='|'.join(reasons)
            if not reasons:
                centered=seg-np.median(seg)
                y=signal.sosfiltfilt(np.array(filters),centered) if filters else centered
                env=np.maximum(0,signal.sosfiltfilt(lp,abs(y)))
                def rms(v):
                    return np.sqrt(np.maximum(0,ndimage.uniform_filter1d(v*v,round(.075*fs))))
                cut=slice(b-a,c-a)
                r,raw_r,env,y=rms(y)[cut],rms(centered)[cut],env[cut],y[cut]
                rel=t[b:c]-start
                row.update(mean_mV=env.mean(),peak_mV=env.max(),peak_s=rel[env.argmax()],
                           iemg_mV_s=np.sum(abs(y))/fs,rms_mV=np.sqrt(np.mean(y*y)),
                           rms75_mean_mV=r.mean(),raw_rms_env_r=np.corrcoef(raw_r,env)[0,1],
                           filtered_rms_env_r=np.corrcoef(r,env)[0,1])
                waves.append([sid,ch+1,e.trial,e.label,*np.interp(np.linspace(0,end-start,101),rel,env)])
                if sid=='P1' and ch==0 and e.trial==1 and e.label=='left':
                    np.savez(OUT/'example.npz',time=rel,raw=seg[cut],filtered=y,envelope=env,rms=r)
            rows.append(row)
    return rows,qc,waves


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    features,quality,profiles=[],[],[]
    for session_dir in sorted((ROOT/'EMG').glob('20*')):
        rows,qc,waves=session(session_dir)
        features.extend(rows)
        quality.extend(qc)
        profiles.extend(waves)
    pd.DataFrame(features).to_csv(OUT/'features.csv',index=False)
    pd.DataFrame(quality).to_csv(OUT/'quality.csv',index=False)
    pd.DataFrame(profiles,columns=['subject','channel','trial','direction',*range(101)]).to_csv(OUT/'profiles.csv.gz',index=False)
    print('QC channels, all trial-channels, usable profiles:',len(quality),len(features),len(profiles))
