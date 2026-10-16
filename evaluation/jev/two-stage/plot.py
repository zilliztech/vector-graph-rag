"""Render the measured quality-latency comparison."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).parent
data=json.loads((D/'chart-data.json').read_text())
pts=data['points']
old={'naive_points':data['naive_points']}
colors={'jev_single':'#168568','jev_two':'#168568','jev_two_114':'#087049','gpt-6-luna':'#a16826','hippo2':'#777f8b','flash':'#2674bd'}
labels={'jev_single':('Jev 1.13 · relations only','o',(10,-12)),'jev_two':('Jev 1.13 · two-stage','D',(-85,18)),'jev_two_114':('Jev 1.14 · two-stage','*',(10,10)),'gpt-6-luna':('GPT-6 Luna','o',(10,-12)),'hippo2':('HippoRAG 2','s',(10,-12)),'flash':('DeepSeek V4.1 Flash','o',(10,10))}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','svg.hashsalt':'vgrag-current'})
fig,ax=plt.subplots(figsize=(15,8.5));fig.subplots_adjust(left=.09,right=.97,top=.82,bottom=.19)
ax.set(xlim=(-.45,16),ylim=(55.5,90),xticks=[0,2,4,6,8,10,12,14,16],yticks=[60,65,70,75,80,85,90],xlabel='Additional model-call latency (s)',ylabel='Average Recall@5 (%)')
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#e7ebef');ax.set_axisbelow(True)
allpts=[(k,v['latency'],v['quality']) for k,v in pts.items()]+[(x['model'],0,x['mean_recall5']) for x in old['naive_points']]
front=[p for p in allpts if not any(x<=p[1] and y>=p[2] and (x<p[1] or y>p[2]) for _,x,y in allpts)]
front.sort(key=lambda p:p[1]);ax.plot([p[1] for p in front],[p[2] for p in front],'--',color='#168568',lw=1.3,alpha=.7,label='Observed Pareto frontier');ax.legend(loc='lower right',frameon=False)
for k,p in pts.items():
 name,marker,offset=labels[k];x,y=p['latency'],p['quality'];c=colors[k]
 ax.scatter(x,y,s=190 if marker=='*' else 80,marker=marker,color=c,edgecolor='white',linewidth=.9,zorder=4)
 ax.annotate(f'{name}\n{y:.2f}% · {"≈" if k=="hippo2" else ""}{x:.2f} s',(x,y),xytext=offset,textcoords='offset points',color=c,fontsize=11,va='bottom' if offset[1]>0 else 'top',ha='left',linespacing=1.4)
for p in old['naive_points']:
 y=p['mean_recall5'];ax.scatter(0,y,s=55,marker='s',facecolors='white',edgecolors='#777f8b',zorder=4)
 ax.annotate(f'Naive RAG · {p["model"]}  ({y:.2f}%)',(0,y),xytext=(12,-4),textcoords='offset points',fontsize=11,color='#777f8b')
fig.text(.09,.94,'Retrieval quality–latency trade-off',fontsize=22,weight='bold',color='#253143')
fig.text(.09,.885,'MuSiQue + 2Wiki · Equal-weight Recall@5 · 1,000 queries per dataset',fontsize=12,color='#657181')
fig.text(.09,.09,'Jev, Luna and Flash: measured API means. HippoRAG 2: reference estimate (≈); separate evaluation setup.',fontsize=10,color='#657181')
fig.text(.09,.055,'0 s excludes embedding, retrieval and answer generation. API means include network and provider queueing; runs differ in time/concurrency.',fontsize=10,color='#657181')
for ext in ['png','svg']:fig.savefig(D.parents[2]/f'docs/assets/evaluation/quality-latency.{ext}',dpi=200,facecolor='white',metadata={'Date':None} if ext=='svg' else None)

svg = D.parents[2] / 'docs/assets/evaluation/quality-latency.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
