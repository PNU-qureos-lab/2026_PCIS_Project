"""Figure6·7의 그래프, 통계 CSV, 발표용 통계표 PNG를 생성합니다.

사용법: python plot_figure6_7_tables.py estimates.csv output_root
기존 7단계의 입력 검사와 그림 스타일을 사용합니다. 추정은 다시 실행하지 않습니다.
전체 사례를 각 성분에서 평가하므로 실제 계수 0과 범위 밖 추정 F도 포함합니다.
"""
from pathlib import Path
import argparse
import json
import os
import tempfile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from step07_figures import _load_estimates, _limits, _scatter, _style_axis, _format_stats, COMPONENTS

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'iteration_pipeline_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table_image(path,title,subtitle,headers,rows,widths,notes):
    """숫자를 새로 계산하지 않고 검증된 통계를 16:9 표로 렌더링합니다."""
    width,height=3840,2160
    image=Image.new('RGB',(width,height),'#F2F6F8')
    navy,teal,ink,muted='#17354B','#087E8B','#1A3447','#5A6F7D'
    regular='C:/Windows/Fonts/malgun.ttf';bold='C:/Windows/Fonts/malgunbd.ttf'
    def text(x,y,value,size=52,color=ink,weight=False,anchor='lm'):
        draw.text((x,y),str(value),font=ImageFont.truetype(bold if weight else regular,size),fill=color,anchor=anchor)
    draw=ImageDraw.Draw(image)
    draw.rectangle((160,138,176,338),fill=teal)
    text(212,145,'ERROR ANALYSIS',32,teal,True)
    text(204,250,title,94,navy,True)
    text(210,360,subtitle,42,muted)
    draw.rounded_rectangle((3050,154,3680,355),radius=26,fill=navy)
    text(3095,209,'평가 사례',36,'#C8DDE7')
    text(3633,289,'4,860',82,'#FFFFFF',True,'rm')
    left,right,top,header_h,row_h=160,3680,560,145,210
    bottom=top+header_h+row_h*len(rows)
    shadow=Image.new('RGBA',(width,height),(0,0,0,0))
    ImageDraw.Draw(shadow).rounded_rectangle((left,top+18,right,bottom+18),radius=28,fill=(12,44,63,25))
    image=Image.alpha_composite(image.convert('RGBA'),shadow.filter(ImageFilter.GaussianBlur(22))).convert('RGB')
    draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((left,top,right,bottom),radius=28,fill='white')
    draw.rounded_rectangle((left,top,right,top+header_h),radius=28,fill=navy)
    draw.rectangle((left,top+65,right,top+header_h),fill=navy)
    edges=[left]
    for span in widths: edges.append(edges[-1]+span)
    assert edges[-1]==right
    for j,label in enumerate(headers):
        text(edges[j]+44 if j==0 else (edges[j]+edges[j+1])/2,top+header_h/2,label,46,'white',True,'lm' if j==0 else 'mm')
    for i,row in enumerate(rows):
        y=top+header_h+i*row_h
        fill='#EAF5F6' if i%2==0 else 'white'
        draw.rounded_rectangle((left,y,right,y+row_h),radius=28 if i==len(rows)-1 else 0,fill=fill)
        if i==len(rows)-1: draw.rectangle((left,y,right,y+35),fill=fill)
        if i: draw.line((left+36,y,right-36,y),fill='#D9E6EB',width=2)
        for j,value in enumerate(row):
            text(edges[j]+44 if j==0 else edges[j+1]-52,y+row_h/2,value,
                 52 if j==0 else 58,navy if j==0 else teal if j==2 else ink,
                 j in (0,2),'lm' if j==0 else 'rm')
    for i,note in enumerate(notes):
        text(175,1710+89*i,note,40,navy if i==0 else muted,i==0)
    draw.line((160,2023,3680,2023),fill='#D3E0E6',width=2)
    text(175,2071,'9차 HydroLight 최종 대표해 평가',32,muted)
    text(3665,2071,'전체 자료 기준',32,muted,False,'rm')
    image.save(path,dpi=(300,300))

def error_stats(error):
    error=np.asarray(error,float)
    return dict(n=len(error),rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(abs(error))),
                bias=float(np.mean(error)),sd=float(np.std(error,ddof=1)),median=float(np.median(error)),
                minimum=float(error.min()),maximum=float(error.max()),
                p025=float(np.quantile(error,.025)),p975=float(np.quantile(error,.975)))

def make_figure6_overall(estimates_csv,output_root):
    """전체 4,860개 사례의 세 성분 오차를 예시 스타일의 한 축에 모읍니다.

    세 계수 오차는 사례마다 합이 0이므로 한 값으로 더하거나 통합 r을 계산하지 않습니다.
    성분별로 색을 구분하고 각각 Pearson r, 양측 p, 최소제곱 회귀선을 표시합니다.
    """
    from scipy.stats import linregress
    from matplotlib.lines import Line2D
    data,audit=_load_estimates(Path(estimates_csv))
    assert len(audit)==4860 and len(data['z_hat'])==4860
    assert int(data['f_outside'].sum())==283
    x=data['z_hat']-data['true_z']
    errors={key:data[f'f_{key}']-data[f'true_f_{key}'] for key,_,_ in COMPONENTS}
    assert np.allclose(sum(errors.values()),0,atol=1e-12,rtol=0)
    folder=Path(output_root)/'Figure6'
    folder.mkdir(parents=True,exist_ok=True)
    palette={'sand':'#D39B27','brown':'#A158CB','green':'#269F83'}
    style={'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
           'mathtext.fontset':'stix','font.size':16,'axes.labelsize':21,
           'xtick.labelsize':15,'ytick.labelsize':15,'axes.linewidth':1.2,
           'pdf.fonttype':42,'ps.fonttype':42}
    rows=[]
    with plt.rc_context(style):
        fig,ax=plt.subplots(figsize=(11,8.5))
        fig.subplots_adjust(left=.115,right=.975,bottom=.15,top=.885)
        fig.suptitle('Figure 6. Depth error vs. fractional coverage error',fontsize=22,fontweight='bold',y=.975)
        xlimits=_limits(x,include_zero=True)
        ylimits=_limits(*errors.values(),include_zero=True)
        handles=[]
        for key,label,_ in COMPONENTS:
            y=errors[key]
            fit=linregress(x,y)
            # p가 수치 정밀도 한계로 0이 반환되어도 'p=0'으로 표현하지 않습니다.
            p_display='< 0.001' if fit.pvalue<.001 else f'= {fit.pvalue:.3f}'
            p_underflow=bool(fit.pvalue==0)
            rows.append(dict(component=key,n_cases=len(x),pearson_r=float(fit.rvalue),
                             p_two_sided=None if p_underflow else float(fit.pvalue),
                             p_display=p_display,p_numerical_underflow=p_underflow,
                             slope_per_m=float(fit.slope),intercept=float(fit.intercept),
                             slope_stderr=float(fit.stderr),f_outside_included=283))
            ax.scatter(x,y,s=23,c=palette[key],edgecolors='none',alpha=.48,zorder=3,rasterized=False)
            line_x=np.asarray(xlimits)
            ax.plot(line_x,fit.intercept+fit.slope*line_x,color=palette[key],ls='--',lw=1.8,zorder=4)
            handles.append(Line2D([],[],ls='--',lw=1.8,marker='o',markersize=6,
                                  color=palette[key],label=f'{label}:  r = {fit.rvalue:+.3f},  p {p_display}'))
        ax.axvline(0,color='#454545',ls=(0,(7,4)),lw=1.1,zorder=2)
        ax.axhline(0,color='#454545',ls=(0,(7,4)),lw=1.1,zorder=2)
        ax.set(xlim=xlimits,ylim=ylimits)
        ax.set_xlabel('Depth error, $e_z$ (m)',labelpad=8)
        ax.set_ylabel('Coverage error, $e_F$',labelpad=8)
        ax.grid(True,color='#D6DADF',lw=.8,alpha=.9)
        ax.set_axisbelow(True)
        ax.tick_params(direction='out',length=5,width=1.1)
        ax.legend(handles=handles,loc='upper left',fontsize=12.5,frameon=True,
                  framealpha=.96,facecolor='white',edgecolor='#333333',borderpad=.75,labelspacing=.8)
        fig.text(.5,.045,'All 4,860 cases; three components per case (14,580 plotted values).',
                 ha='center',fontsize=11.5)
        fig.text(.5,.018,'Errors = estimated - true. Out-of-range F included. Dashed colored lines: linear fits.',
                 ha='center',fontsize=10.5)
        for extension in ('png','pdf'):
            fig.savefig(folder/f'figure6_all_data.{extension}',dpi=300,facecolor='white')
        plt.close(fig)
    stats=pd.DataFrame(rows)
    old_stats=folder/'figure6_statistics.csv'
    if old_stats.exists():
        original=pd.read_csv(old_stats,float_precision='round_trip').set_index('component')
        for row in rows:
            assert abs(row['pearson_r']-original.loc[row['component'],'pearson_r'])<1e-12
    stats.to_csv(folder/'figure6_all_data_statistics.csv',index=False,encoding='utf-8-sig')
    metadata=dict(cases=4860,component_cases_each=4860,plotted_values=14580,
                  f_outside_included=283,zero_truth_coefficients_included=True,
                  error='estimated-true',pooled_component_correlation=False,
                  regression='OLS with intercept, separately by component',
                  p_method='Two-sided Pearson correlation test for 4860 cases per component',
                  input_modified=False,jitter=False,axis_clipping=False)
    (folder/'figure6_all_data_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'전체그래프_설명.md').write_text('''# Figure6 전체 산점도

전체 4,860개 사례를 한 그래프에 표시했습니다. 사례마다 모래·갈조·녹조 오차를 각각 그려 총 14,580개 점입니다.
가로축은 추정 수심-실제 수심(m), 세로축은 추정 계수-실제 계수(무단위)입니다.
음수 또는 1 초과 F인 283개 사례와 실제 계수 0인 성분도 포함했습니다. 점을 옮기거나 큰 오차를 잘라내지 않았습니다.

색으로 성분을 구분하며, 성분별 점선은 절편을 포함한 최소제곱 회귀선입니다.
r은 Pearson 상관계수이고 p는 성분별 4,860개 사례에 대한 양측 Pearson 검정값입니다.
p는 일반적인 사례 독립 가정을 사용하는 값이며, 수질 조건 공유에 대한 군집 보정값은 아닙니다.

세 계수의 합이 1이므로 같은 사례의 세 계수 오차를 더하면 0입니다.
세 성분을 구분하지 않고 통합 상관계수를 만들면 상쇄되므로 r과 회귀선을 성분별로 표시했습니다.
첨부 예시의 r=0.63 또는 점 분포를 사용하지 않고 실제 자료에서 계산했습니다.
''',encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False))

def make_figure6_7(estimates_csv,output_root):
    estimates_csv,output_root=Path(estimates_csv),Path(output_root)
    data,audit=_load_estimates(estimates_csv)
    n=len(data['z_hat'])
    assert n==4860 and len(audit)==4860
    outside=data['f_outside']
    assert outside.sum()==283
    depth_error=data['z_hat']-data['true_z']
    coef={key:data[f'f_{key}']-data[f'true_f_{key}'] for key,_,_ in COMPONENTS}
    assert np.allclose(sum(coef.values()),0,atol=1e-12,rtol=0)
    f6,f7=output_root/'Figure6',output_root/'Figure7'
    f6.mkdir(parents=True,exist_ok=True);f7.mkdir(parents=True,exist_ok=True)
    correlations=[];distributions=[]
    for key,label,_ in COMPONENTS:
        err=coef[key]
        xc,yc=depth_error-depth_error.mean(),err-err.mean()
        pearson=float(np.dot(xc,yc)/np.sqrt(np.dot(xc,xc)*np.dot(yc,yc)))
        correlations.append(dict(component=key,label=label,pearson_r=pearson,**error_stats(err)))
    variables=[('depth','Depth',depth_error,'m','#0072B2')]+[(key,label,coef[key],'unitless',color) for key,label,color in COMPONENTS]
    for key,label,err,unit,_ in variables:
        distributions.append(dict(variable=key,label=label,unit=unit,**error_stats(err)))
    pd.DataFrame(correlations).to_csv(f6/'figure6_statistics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(distributions).to_csv(f7/'figure7_statistics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(dict(case_id=data['case_id'],true_depth=data['true_z'],depth_error_m=depth_error,
                      sand_coefficient_error=coef['sand'],brown_coefficient_error=coef['brown'],
                      green_coefficient_error=coef['green'],f_outside=outside)).to_csv(f6/'figure6_points.csv',index=False,encoding='utf-8-sig')
    bins_rows=[]
    note=f'All {n:,} cases; out-of-range F included: {int(outside.sum()):,}'
    style={'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
           'legend.fontsize':8,'figure.titlesize':14,'pdf.fonttype':42,'ps.fonttype':42}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
        for ax,(key,label,color),stats in zip(axes,COMPONENTS,correlations):
            err=coef[key]
            _scatter(ax,depth_error,err,outside,color)
            ax.axhline(0,color='#555555',ls='--',lw=.8)
            ax.axvline(0,color='#555555',ls='--',lw=.8)
            ax.set_xlim(_limits(depth_error,include_zero=True));ax.set_ylim(_limits(err,include_zero=True))
            ax.set_title(f"{label}\nPearson r = {stats['pearson_r']:+.3f}; n = {n:,}")
            ax.set_xlabel('Depth error: estimated - true (m)')
            ax.set_ylabel('Coefficient error: estimated - true (unitless)')
            _style_axis(ax)
        axes[0].legend(loc='best')
        fig.suptitle('Figure 6. Signed depth and coefficient errors\n'+note,fontsize=12)
        for ext in ('png','pdf'): fig.savefig(f6/f'figure6_error_relationships.{ext}',dpi=300,facecolor='white',bbox_inches='tight')
        plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11.5,8.8),layout='constrained')
        for ax,(key,label,err,unit,color) in zip(axes.flat,variables):
            bins=np.linspace(float(err.min()),float(err.max()),min(70,max(15,int(np.ceil(np.sqrt(n)))))+1)
            if err.max()==err.min(): bins=np.linspace(*_limits(err,include_zero=True),16)
            counts,edges=np.histogram(err,bins=bins)
            assert counts.sum()==4860
            bins_rows.extend(dict(variable=key,bin_left=float(a),bin_right=float(b),count=int(c),
                                  rightmost_bin=i==len(counts)-1) for i,(a,b,c) in enumerate(zip(edges[:-1],edges[1:],counts)))
            ax.hist(err,bins=bins,color=color,alpha=.78,edgecolor='white',linewidth=.35)
            ax.axvline(0,color='#333333',ls='--',lw=1)
            ax.set_xlim(_limits(err,include_zero=True))
            ax.set_title(label+'\n'+_format_stats(err,'m' if key=='depth' else '').replace('\n','  |  '),fontsize=10.5)
            ax.set_xlabel('Estimated - true'+(' (m)' if key=='depth' else ' (unitless)'))
            ax.set_ylabel('Number of cases');_style_axis(ax)
        fig.suptitle('Figure 7. Signed error distributions\n'+note,fontsize=12)
        for ext in ('png','pdf'): fig.savefig(f7/f'figure7_error_distributions.{ext}',dpi=300,facecolor='white',bbox_inches='tight')
        plt.close(fig)
    pd.DataFrame(bins_rows).to_csv(f7/'figure7_histogram_bins.csv',index=False,encoding='utf-8-sig')
    f6rows=[[r['label'],f"{n:,}",f"{r['pearson_r']:+.3f}",f"{r['rmse']:.3f}",f"{r['mae']:.3f}",f"{r['bias']:+.3f}"] for r in correlations]
    table_image(f6/'figure6_statistics_table.png','Figure 6. 수심·바닥 오차의 관계',note.replace('All 4,860 cases; out-of-range F included: 283','전체 4,860개  ·  범위 밖 F 283개 포함'),
                ['바닥 계수','n','Pearson r','F 오차 RMSE','F 오차 MAE','F 오차 Bias'],f6rows,[1050,310,540,540,540,540],
                ['상관계수는 수심 오차와 각 바닥 계수 오차가 함께 움직이는 정도를 나타냅니다.',
                 '오차 = 추정 - 실제. 양수는 과대 추정, 음수는 과소 추정입니다.',
                 '각 성분에서 실제 F=0인 사례까지 4,860개 전부 포함했습니다. F 오차의 단위는 없습니다.'])
    f7rows=[['수심 (m)' if r['variable']=='depth' else r['label']+' F',f"{n:,}",f"{r['rmse']:.3f}",f"{r['mae']:.3f}",f"{r['bias']:+.3f}",f"{r['sd']:.3f}",f"{r['median']:+.3f}"] for r in distributions]
    table_image(f7/'figure7_statistics_table.png','Figure 7. 수심·바닥 오차의 분포','전체 4,860개  ·  범위 밖 F 283개 포함',
                ['평가 변수','n','RMSE','MAE','Bias','표준편차','중앙값'],f7rows,[1150,320,410,410,410,410,410],
                ['평균 오차는 Bias이며, 표준편차는 오차가 평균 주변에 얼마나 퍼져 있는지 나타냅니다.',
                 '표준편차는 표본 기준(n-1)입니다. 수심 오차는 m, 바닥 계수 오차는 무단위입니다.',
                 '각 성분에서 실제 F=0인 사례도 포함합니다. 히스토그램은 전체 오차 범위를 표시합니다.'])
    metadata=dict(input_estimates=str(estimates_csv.resolve()),cases=4860,f_outside_included=283,
                  include_true_zero_coefficients=True,coefficient_cases_each=4860,error='estimated-true',
                  correlation='Pearson correlation between depth error and component error',sd_ddof=1,
                  inversion_recomputed=False,relative_error=False,figures=[6,7])
    for folder in (f6,f7):
        (folder/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        (folder/'설명.md').write_text('''# Figure6·7 통계 기준

전체 4,860개 최종 대표해를 사용합니다. F<0 또는 F>1인 283개도 포함합니다.
각 바닥 계수는 실제 F=0인 경우까지 모두 포함하므로, 실제 해당 성분이 있는 3,078개만 평가한 Figure1의 성분별 표와 기준이 다릅니다.

Figure6: 가로축은 수심 오차(m), 세로축은 해당 바닥 계수 오차(무단위). Pearson r은 두 오차의 상관계수이며 성능 R²가 아닙니다.
양수 r은 두 오차가 같은 방향으로, 음수 r은 반대 방향으로 움직이는 경향입니다. 상관관계 자체가 원인을 증명하지는 않습니다.
Figure6 표의 RMSE·MAE·Bias는 해당 바닥 계수 오차에 대한 값입니다.

Figure7: 수심·세 바닥 계수 오차의 빈도 분포. 오차는 추정-실제이며 상대오차가 아닙니다.
RMSE=sqrt(mean(error²)), MAE=mean(abs(error)), Bias=mean(error), 표본 표준편차는 ddof=1, 중앙값은 median(error)입니다.
히스토그램 구간은 왼쪽 포함·오른쪽 제외이고 마지막 구간만 오른쪽 끝도 포함합니다. 변수마다 빈도 합은 4,860입니다.
''',encoding='utf-8')
    make_figure6_overall(estimates_csv,output_root)
    print(json.dumps(dict(figure6=correlations,figure7=distributions),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('estimates_csv',type=Path)
    parser.add_argument('output_root',type=Path)
    parser.add_argument('--figure6-overall-only',action='store_true',help='Create only the combined Figure6 scatter')
    args=parser.parse_args()
    if args.figure6_overall_only:
        make_figure6_overall(args.estimates_csv,args.output_root)
    else:
        make_figure6_7(args.estimates_csv,args.output_root)
