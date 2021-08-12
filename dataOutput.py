import pandas as pd
from WindPy import w
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
w.start() # 默认命令超时时间为120秒，如需设置超时时间可以加入waitTime参数，例如waitTime=60,即设置命令超时时间为60秒
w.isconnected() # 判断WindPy是否已经登录成功

from basicData import ind1,today

data = pd.read_excel(f'./intermediateData/data_{today}.xlsx')

def recommend_df_month():
    # 按月
    temp_report = data[ind1 + ['报告日期']].set_index('报告日期').resample('M').sum()
    temp_report_mean = temp_report.rolling(5).mean()
    recommend = []
    for year in range(2013, dt.datetime.today().year + 1):
        for month in range(1, 13):
            tempdict = {}
            if month != 12:
                startdate = dt.datetime(year, month, 1)
                enddate = dt.datetime(year, month + 1, 1)
            else:
                startdate = dt.datetime(year, month, 1)
                enddate = dt.datetime(year + 1, 1, 1)
            tempdict = dict(zip(data[(data['报告日期'] >= startdate) & (data['报告日期'] < enddate)][ind1].sum().index,
                                data[(data['报告日期'] >= startdate) & (data['报告日期'] < enddate)][ind1].sum()))
            tempdict['研报总数'] = data[(data['报告日期'] >= startdate) & (data['报告日期'] < enddate)][ind1].count().iloc[0]
            tempdict['startdate'] = startdate
            tempdict['enddate'] = enddate
            recommend.append(tempdict)
    recommend_df = pd.DataFrame(recommend)
    recommend_df[ind1] = recommend_df[ind1].astype('int')
    recommend_df = recommend_df.drop(index=recommend_df[recommend_df['研报总数'] == 0].index)
    recommend_df = recommend_df[['startdate', 'enddate', '研报总数'] + ind1]
    recommend_df = recommend_df.reset_index(drop=True)

    # recommend_df.set_index('startdate').to_excel('./outputData/申万行业推荐次数(月).xlsx')
    return recommend_df

def recommend_df_week():
    # 按周
    today = dt.date.today()
    today = str(today).replace('-', '')

    recommend = []
    temp_report = data[ind1 + ['报告日期']].set_index('报告日期').resample('w').sum()
    temp_report_mean = temp_report.rolling(5).mean()
    for startdate, enddate in zip(temp_report.index, temp_report.index.shift(1)):
        tempdict = dict(zip(data[(data['报告日期'] >= startdate) & (data['报告日期'] < enddate)][ind1].sum().index,
                            data[(data['报告日期'] >= startdate) & (data['报告日期'] < enddate)][ind1].sum()))
        tempdict['研报总数'] = data[(data['报告日期'] >= startdate) & (data['报告日期'] < enddate)][ind1].count().iloc[0]
        tempdict['startdate'] = startdate
        tempdict['enddate'] = enddate
        recommend.append(tempdict)

    recommend_df = pd.DataFrame(recommend)
    recommend_df[ind1] = recommend_df[ind1].astype('int')
    recommend_df = recommend_df.drop(index=recommend_df[recommend_df['研报总数'] == 0].index)
    recommend_df = recommend_df[['startdate', 'enddate', '研报总数'] + ind1]
    recommend_df = recommend_df.reset_index(drop=True)

    recommend_df.set_index('startdate').to_excel(f'./outputData/申万行业推荐次数_{today}.xlsx')
    return recommend_df

# 函数，输入月份(1,3,6)，输出最近（1，3，6个月）的涨幅
def compute_pct_chg(month=1):
    today = data['报告日期'].iloc[-1]
    itemdate = str(w.tdaysoffset(-month, today, "Period=M", usedf=True)[1].values[0][0])[:10]
    pdata = w.wss(
        "801010.SI,801020.SI,801030.SI,801040.SI,801050.SI,801080.SI,801110.SI,801120.SI,801130.SI,801140.SI,801150.SI,801160.SI,801170.SI,801180.SI,801200.SI,801210.SI,801230.SI,801710.SI,801720.SI,801730.SI,801740.SI,801750.SI,801760.SI,801770.SI,801780.SI,801790.SI,801880.SI,801890.SI",
        "sec_name,close_per,chg_per,pct_chg_per,turn_free_per", f"startDate={itemdate};endDate={str(today)[:10]}",
        usedf=True)[1]

    pdata['SEC_NAME'] = pdata['SEC_NAME'].apply(lambda x: x[:-4])
    pdata = pdata.reset_index(drop=True).rename({"SEC_NAME": '板块名称', 'CLOSE_PER': '板块最新价',
                                               'TURN_FREE_PER': '板块换手率', 'CHG_PER': '板块涨跌额', 'PCT_CHG_PER': '板块涨跌幅'},
                                              axis=1)
    return pdata

# 函数，输入月份(1,3,6)，输出最近（1，3，6个月）的各行业的推荐比例
def compute_ratio_by_month(month=1):
    recommend_month = recommend_df_month()
    globals()['last' + str(month)] = (recommend_month.iloc[-month:, 2:].sum() / recommend_month.iloc[-month:, 2:].sum()[
        '研报总数']) 
    return eval('last' + str(month))


def output_final():
    recommend_month = recommend_df_month()
    recommend_week = recommend_df_week()

    ratio_by_month6 = compute_ratio_by_month(6)
    ratio_by_month3 = compute_ratio_by_month(3)
    ratio_by_month1 = compute_ratio_by_month(1)

    today = recommend_month['enddate'].iloc[-1]
    templist = []
    for i in ind1:
        tempdict = {}
        tempdict['行业'] = i
        tempdict['最近6个月推荐比例'] = ratio_by_month6[i].round(2)
        tempdict['最近3个月推荐比例'] = ratio_by_month3[i].round(2)
        tempdict['最近1个月推荐比例'] = ratio_by_month1[i].round(2)
        templist.append(tempdict)
    out = pd.DataFrame(templist)

    df_bk_combine = pd.DataFrame()
    for i in [1, 3, 6]:
        df_bk = compute_pct_chg(month=i)
        df_bk['板块涨跌幅'] = df_bk['板块涨跌幅'].round(2)
        df_bk["期限"] = i
        df_bk_combine = pd.concat((df_bk_combine, df_bk), axis=0, join='outer', ignore_index=False)
        df_bk_pivot = pd.pivot_table(df_bk_combine, index='板块名称', columns='期限', values='板块涨跌幅')

    df_bk_pivot = df_bk_pivot.reset_index()
    for i in [1, 3, 6]:
        df_bk_pivot.sort_values(by=i, ascending=False, inplace=True)
        df_bk_pivot.reset_index(drop=True, inplace=True)
        df_bk_pivot.reset_index(inplace=True)
        df_bk_pivot['index'] = df_bk_pivot['index'] + 1
        df_bk_pivot.rename({'index': f'最近{i}个月涨跌幅排序'}, axis=1, inplace=True)

    out_final = pd.merge(out, df_bk_pivot, left_on=['行业'], right_on=['板块名称'], how='left')
    out_final = out_final.rename({1: '最近1个月涨跌幅', 3: '最近3个月涨跌幅', 6: '最近6个月涨跌幅'}, axis=1).drop('板块名称', axis=1)

    for row in out_final.index:
        for i in recommend_week[-14:].index:
            startdate = recommend_week.loc[i, 'startdate']
            enddate = recommend_week.loc[i, 'enddate']
            tempdata = data[(data['报告日期'] > startdate) & (data['报告日期'] <= enddate)]
            temp_ind_data = tempdata[tempdata[out_final.loc[row, '行业']] == 1]
            if temp_ind_data.shape[0] != 0:
                out_final.loc[row, f'enddate:{str(enddate)[:10]}'] = '; '.join(
                    temp_ind_data.apply(lambda x: str(x['报告日期'])[:10] + ' ' + x['机构'], axis=1).tolist())
            else:
                pass
    today = dt.date.today()
    today = str(today).replace('-', '')
    out_final.to_excel(f'./outputData/行业底层数据观察_{today}.xlsx')

if __name__ == '__main__':
    output_final()