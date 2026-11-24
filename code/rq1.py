"""Summarize equal-weight repeated shifts within participants and render RQ1 figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from process import OUT, DIR

SUB=['P1','P2','P3']
NAMES=['GMed','RF','VL','VM','BF','TA','Gastroc','Sol']
LABELS=[f'{i+1:02d}  {"L" if i<8 else "R"} {NAMES[i%8]}' for i in range(16)]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def save(fig,name):
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=300)
    plt.close(fig)
    print('Saved',name,flush=True)


def summarize(f,w):
    records=[]
    for (s,c,d),g in f.groupby(['subject','channel','direction']):
        curves=w.loc[(w.subject==s)&(w.channel==c)&(w.direction==d)].iloc[:,4:].to_numpy()
        n=len(g)
        assert n==len(curves)
        cor=np.corrcoef(curves) if n>=2 else None
        records.append(dict(subject=s,channel=c,muscle=g.muscle.iloc[0],direction=d,n=n,
                            median_mV=g.mean_mV.median(),cv=g.mean_mV.std()/g.mean_mV.mean(),
                            shape_r=np.median(cor[np.triu_indices(n,1)]) if n>=2 else np.nan))
    metrics=pd.DataFrame(records)
    ranks=[]
    for (s,c),g in metrics.groupby(['subject','channel']):
        if len(g)!=4 or g.n.min()<3: continue
        best,second=g.sort_values('median_mV',ascending=False).iloc[:2].itertuples(index=False)
        trials=f[(f.subject==s)&(f.channel==c)].pivot(index='trial',columns='direction',values='mean_mV').dropna()
        lead=best.median_mV/second.median_mV
        wins=(trials[best.direction]>trials.drop(columns=best.direction).max(axis=1)).sum()
        ranks.append(dict(subject=s,channel=c,muscle=best.muscle,preferred=best.direction,
                          lead_ratio=lead,wins=wins,complete_rounds=len(trials),cv=best.cv,shape_r=best.shape_r,
                          candidate=lead>=1.25 and wins>=3 and best.cv<=.30 and best.shape_r>=.50))
    ranks=pd.DataFrame(ranks)
    metrics.to_csv(OUT/'repeatability.csv',index=False)
    ranks.to_csv(OUT/'selectivity.csv',index=False)
    return metrics,ranks


def plots(w,m):
    ex=np.load(OUT/'example.npz')
    fig,axs=plt.subplots(3,1,sharex=True,figsize=(7,5.3),layout='constrained')
    for ax,key,label in zip(axs,['raw','filtered','envelope'],['A   Calibrated raw','B   Processed EMG','C   Activation']):
        if key=='envelope':
            ax.plot(ex['time'],ex['rms'],color='0.6',lw=1,label='75 ms RMS')
        ax.plot(ex['time'],ex[key],color='#116979',lw=.65 if key!='envelope' else 1.3,label='5 Hz envelope')
        ax.set_ylabel('mV'); ax.set_title(label,loc='left',fontsize=10)
    axs[-1].legend(loc='upper right'); axs[-1].set_xlabel('Time from left-shift cue (s)')
    fig.suptitle('P1  ·  Left gluteus medius  ·  First left shift')
    save(fig,'fig1_processing')
    fig,axs=plt.subplots(4,4,figsize=(10,7.6),sharex=True,layout='constrained')
    for c,ax in enumerate(axs.flat,1):
        for d,color in zip(DIR,['#0072b2','#d55e00','#009e73','#a65790']):
            g=w[(w.subject=='P1')&(w.channel==c)&(w.direction==d)]
            if len(g): ax.plot(range(101),g.iloc[:,4:].median(),label=d.capitalize(),color=color,lw=1.2)
        ax.set_title(LABELS[c-1],loc='left',fontsize=10); ax.set_ylim(bottom=0)
    axs[0,0].legend(fontsize=7,ncol=2)
    fig.supxlabel('Shift phase (%)');fig.supylabel('Envelope (mV; channel-specific scales)')
    fig.suptitle('P1  ·  Median activation across valid repetitions')
    save(fig,'fig2_envelopes')
    mm=m.copy()
    mm['relative']=np.log2(mm.median_mV/mm.groupby(['subject','channel']).median_mV.transform('mean'))
    mm.loc[mm.groupby(['subject','channel']).direction.transform('count')<4,'relative']=np.nan
    fig,axs=plt.subplots(1,3,sharey=True,figsize=(9,6.6),layout='constrained')
    limit=abs(mm.relative).max()
    for ax,s in zip(axs,SUB):
        mat=mm[mm.subject==s].pivot(index='channel',columns='direction',values='relative').reindex(index=range(1,17),columns=DIR)
        im=ax.imshow(mat,cmap=plt.get_cmap('RdBu_r').with_extremes(bad='#b8b8b8'),vmin=-limit,vmax=limit,aspect='auto')
        ax.set_xticks(range(4),['Left','Right','Fwd','Back']); ax.set_title(s)
    axs[0].set_yticks(range(16),LABELS)
    fig.colorbar(im,ax=axs,label='log2(direction median / four-direction mean)',shrink=.74)
    fig.suptitle('Directional modulation within participant and channel')
    fig.supxlabel('Grey cells indicate insufficient data for a four-direction comparison',fontsize=9)
    save(fig,'fig3_directions')
    fig,axs=plt.subplots(1,2,sharey=True,figsize=(11,7),layout='constrained')
    for ax,key,title,cmap in zip(axs,['cv','shape_r'],['A   Amplitude CV','B   Median pairwise waveform correlation'],['viridis','RdBu_r']):
        matrix=mm.assign(**{key:mm[key].where(mm.n>=3)}).pivot(index='channel',columns=['subject','direction'],values=key).reindex(index=range(1,17),columns=pd.MultiIndex.from_product([SUB,DIR]))
        im=ax.imshow(matrix,aspect='auto',cmap=plt.get_cmap(cmap).with_extremes(bad='#b8b8b8'),vmin=0 if key=='cv' else -1,vmax=mm.cv.max() if key=='cv' else 1)
        ax.set_xticks(range(12),['L','R','F','B']*3); ax.set_title(title,fontsize=11,pad=22)
        for x,s in zip([1.5,5.5,9.5],SUB): ax.text(x,-.9,s,ha='center',fontsize=9)
        for x in [3.5,7.5]: ax.axvline(x,color='white',lw=2)
        fig.colorbar(im,ax=ax,shrink=.65)
    axs[0].set_yticks(range(16),LABELS)
    fig.supxlabel('Cue direction within each participant  ·  Grey cells have fewer than 3 repetitions')
    save(fig,'fig4_repeats')


if __name__=='__main__':
    f=pd.read_csv(OUT/'features.csv')
    f=f[f.exclude.isna()]
    w=pd.read_csv(OUT/'profiles.csv.gz')
    metrics,ranks=summarize(f,w)
    print(ranks[ranks.candidate].to_string(index=False),flush=True)
    plots(w,metrics)
