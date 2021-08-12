import os
import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from basicData import ind, ind1, ind2, ind3, trans_dict,trans_dict_df, sw1, sw2, sw3,\
    inddata, benchmark, TRADING_DAY, ONE_DAY, today

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题


# #######################需要修改的部分#########################

source_data_path = './Data/行业配置池-20210301160511.xls'


# ###########################################################

class datapre():
    def __init__(self):
        if os.path.exists(f'./intermediateData/data_{today}.xlsx'):
            self.data = pd.read_excel(f'./intermediateData/data_{today}.xlsx')
        else:
            self.rawdata = pd.read_excel(source_data_path)
            self.basic_process()
            self.ind_txt_process()
            self.fill_up_industry()
            self.data.to_excel(f'./intermediateData/data_{today}.xlsx')

        self.rawinddata = inddata
        self.rawbenchmark = benchmark
        self.get_market_data()
        self.count_by_month = self.data.set_index('报告日期').resample('m').count()
        self.count_by_week = self.data.set_index('报告日期').resample('w').count()
        self.count_by_day = self.data.groupby('day').count().reset_index()

    def basic_process(self):
        data = self.rawdata.groupby(['报告日期', '机构']).apply(lambda x: ','.join(x['行业']))
        data = data.reset_index().rename({0: '行业'}, axis=1)
        data['报告日期'] = data['报告日期'].astype('datetime64[D]')
        data['day'] = data['报告日期'].apply(lambda x: x.day)

        # 清除无意义词汇
        for i in ['板块', '产业链', '主题', '行业']:
            data = data.applymap(lambda x: x.replace(i, '') if type(x) == str else x)

        # 1. 机构信息处理
        data['机构'] = data['机构'].rename({'广发恒生': '广发证券',
                                        '申银万国': '申万宏源',
                                        '宏源证券': '申万宏源',
                                        })
        # 2. 选取头部券商 40
        top_security = data['机构'].value_counts()[:40].index
        data = data[data['机构'].isin(top_security)]
        data = data.reset_index(drop=True)
        self.data = data
        return data

    def visual_by_day(self):
        # 按日研报数量
        fig = plt.figure(figsize=(12, 4), dpi=100)
        ax1 = fig.add_subplot(1, 1, 1)
        sns.barplot(data=self.count_by_day, x='day', y='行业', color='#448ee4', ax=ax1)
        return fig

    def visual_by_month(self):
        # 按月份研报数量
        fig = plt.figure(figsize=(12, 4), dpi=100)
        ax1 = fig.add_subplot(1, 1, 1)

        sns.barplot(data=self.count_by_month.reset_index(), x='报告日期', y='行业', color='#448ee4', ax=ax1)
        xticklabel = ['2014-1', '2015-1', '2016-1', '2017-1', '2018-1', '2019-1', '2020-1', '2021-1']
        ax1.set_xticks([4, 16, 28, 40, 52, 64, 76, 88])
        ax1.set_xticklabels(xticklabel)
        return fig

    def visual_by_institute(self):
        # 按机构 研报数量
        self.count_by_ins = self.data['机构'].value_counts().sort_values(ascending=False)[:50]
        fig = plt.figure(figsize=(12, 6), dpi=100)
        ax1 = fig.add_subplot(1, 1, 1)
        sns.barplot(x=self.count_by_ins.index, y=self.count_by_ins.values, color='#448ee4', ax=ax1)
        ax1.set_xticklabels(self.count_by_ins.index, rotation=90)
        return fig

    def fill_up_industry(self):
        for i in self.data.index:
            for j in ind2:
                if self.data.loc[i, j] == 1:
                    temp = ind[ind['二级行业名称'] == j]['一级行业名称'].iloc[0]
                    self.data.loc[i, temp] = 1
            for k in ind3:
                if self.data.loc[i, k] == 1:
                    temp = ind[ind['三级行业名称'] == k]['一级行业名称'].iloc[0]
                    self.data.loc[i, temp] = 1
        return self.data

    def get_market_data(self):
        # 行业涨跌行情数据处理：（包含申万一级，二级，三级），可修改参数：频率 （仅交易日）
        inddata = self.rawinddata.reset_index().rename({'index': 'date'}, axis=1)
        inddata = inddata.rename(dict(zip(list(sw1['SEC_NAME'].index), list(sw1['SEC_NAME'].values))), axis=1)
        inddata = inddata.rename(dict(zip(list(sw2['SEC_NAME'].index), list(sw2['SEC_NAME'].values))), axis=1)
        inddata = inddata.rename(dict(zip(list(sw3['SEC_NAME'].index), list(sw3['SEC_NAME'].values))), axis=1)

        inddata = inddata.T.reset_index().drop_duplicates(subset=['index']).set_index('index').T.reset_index(drop=True)
        inddata[inddata.columns[1:]] = inddata[inddata.columns[1:]].astype('float64')
        inddata['date'] = inddata['date'].astype('datetime64[D]')
        inddata = inddata.fillna(method='ffill')
        inddata = inddata.set_index('date')
        self.inddata = inddata

        # 参考基准涨跌行业，可修改参数：指数代码，频率 (仅交易日)
        benchmark = self.rawbenchmark.reset_index().rename({'index': 'date'}, axis=1)
        benchmark['month'] = benchmark['date'].apply(lambda x: x.month)
        benchmark['year'] = benchmark['date'].apply(lambda x: x.year)
        benchmark["bench_pct_chg"] = benchmark["PCT_CHG"]

        benchmark['date'] = benchmark['date'].astype('datetime64[D]')
        benchmark = benchmark.fillna(method='ffill')
        benchmark = benchmark.set_index('date')  # .resample('M').sum()
        self.benchmark = benchmark

        # # ------------------------------------------------------------------------------------
        # benchmark= w.wsd(','.join(sw1.index), "pct_chg", "2013-09-01", "2021-02-28", "Days=Alldays;PriceAdj=F",usedf=True)[1]
        # benchmark = benchmark.mean(axis=1)
        # benchmark.index.name = 'date'
        # benchmark = pd.DataFrame(benchmark,columns=['bench_pct_chg'])
        # benchmark = benchmark.fillna(method='ffill')

    def ind_txt_process(self):
        data = self.data
        data['行业拆分'] = data['行业'].apply(clean_search)
        data['行业拆分'] = data['行业拆分'].apply(trans)
        data['行业包含'] = data['行业拆分'].apply(lambda x: fuzzy_match(x, ind1, ind2, ind3)[0])

        data[ind1] = 0
        data[ind2] = 0
        data[ind3] = 0

        data['inter_list'] = data['行业拆分'].apply(lambda x: fuzzy_match(x, ind1, ind2, ind3)[0])
        data['diff_list'] = data['行业拆分'].apply(
            lambda x: list(set(x).difference(set(fuzzy_match(x, ind1, ind2, ind3)[1]))))

        for count, i in enumerate(data['inter_list']):
            for each in i:
                data.loc[count, each] = 1

        self.data = data


class datastrategy():
    def __init__(self, item, dataobj, period='W'):
        self.data = dataobj
        self.item = item
        self.temp_report = dataobj.data.set_index('报告日期').resample('W').sum()  # 'M'和count_by_month ；'W'和count_by_week

    def buy_signal(self, ):
        # relative signal 是根据研报日期发出的，可能包含非交易日
        # 'M'和count_by_month ；'W'和count_by_week
        item = self.item
        temp_report = self.temp_report[[self.item]]
        relative_count = temp_report[self.item] / self.data.count_by_week['行业']

        # 相对数量信号
        mean = relative_count.rolling(5).mean()
        relative_signal = relative_count[(mean >= mean.rolling(100).max()) & (mean > 0.25)].index
        # relative_signal = relative_count[(mean>mean.rolling(5).mean()*1.25)&(relative_count>0.25)].index
        # relative_signal = relative_count[(relative_count>relative_count.rolling(10).mean()*1.5)&(relative_count>0.25)].index

        # 绝对数量信号
        mean = temp_report.rolling(5).mean()
        abs_signal = mean[(mean[item] >= mean[item].rolling(250).max()) & (mean[item] >= 5)].index

        # 价格趋势信号
        temp_pct = ((self.data.inddata[item] - self.data.benchmark["bench_pct_chg"]) / 100 + 1).cumprod()
        inter_signal1 = list(set(map_next_trading_day(relative_signal)).intersection(
            set(temp_pct.loc[temp_pct.pct_change(30) > 0].index)))
        inter_signal2 = list(
            set(map_next_trading_day(abs_signal)).intersection(set(temp_pct.loc[temp_pct.pct_change(90) > 0].index)))
        inter_signal = list(set(inter_signal1).union(set(inter_signal2)))
        return (inter_signal)  # 可能包含非交易日,需要转换

    def sell_signal(self, ):
        # 'M'和count_by_month 搭配；'W'和count_by_week搭配
        temp_report = self.temp_report[[self.item]]
        relative_count = temp_report[self.item] / self.data.count_by_week['行业']

        relative_signal = relative_count[
            (relative_count < relative_count.rolling(5).mean() * 0.25) & (relative_count < 0.1)].index
        return map_next_trading_day(relative_signal)

    def strategy(self, ):
        # temp_access 针对超额收益/累计收益 日度数据（交易日）index是日期
        temp_access = pd.DataFrame((self.data.inddata[self.item] - self.data.benchmark["bench_pct_chg"]) / 100).rename(
            {0: 'excess_return'}, axis=1)

        temp_access['signal'] = 0
        temp_access.loc[self.buy_signal(), 'signal'] = 1
        temp_access.loc[self.sell_signal(), 'signal'] = -1
        temp_access = temp_access.reset_index()  # 此处把index重置

        temp_access['position'] = 0
        temp_access['strategy'] = 1
        for i in temp_access.index[1:]:
            if temp_access.loc[i, 'signal'] == 1:
                temp_access.loc[i, 'position'] = 1
            if temp_access.loc[i, 'signal'] == -1:
                temp_access.loc[i, 'position'] = -1
            if temp_access.loc[i, 'signal'] == 0:
                temp_access.loc[i, 'position'] = temp_access.loc[i - 1, 'position']

        for i in temp_access.index[1:]:
            if temp_access.loc[i, 'position'] == 1:
                temp_access.loc[i, 'strategy'] = temp_access.loc[i - 1, 'strategy'] * (
                            1 + temp_access.loc[i, 'excess_return'])
            if temp_access.loc[i, 'position'] == -1:
                temp_access.loc[i, 'strategy'] = temp_access.loc[i - 1, 'strategy'] * (
                            1 - temp_access.loc[i, 'excess_return'])
            if temp_access.loc[i, 'position'] == 0:
                temp_access.loc[i, 'strategy'] = temp_access.loc[i - 1, 'strategy']
        self.cumulative = temp_access

        return self.cumulative

    def itemplot(self, ):
        item = self.item
        dataobj = self.data
        temp_report = self.temp_report
        # 1.-------------------------------------------------------------------------------------------------
        fig = plt.figure(figsize=(12, 8), dpi=100)
        ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3, colspan=1)

        relative_count = self.temp_report[item] / dataobj.count_by_week['行业']
        temp_pct = ((dataobj.inddata[item] - dataobj.benchmark["bench_pct_chg"]) / 100 + 1).cumprod()  # 日频，只有累计超额收益

        # 1.1-------------------------------------------------------------------------------------------------

        if temp_pct[-1] - temp_pct[0] < 0:
            kind = 1
        else:
            kind = 0

        if kind == 1:
            # 1.1.1 最大涨幅
            # sns.lineplot(temp.index,maxup(temp_pct),color='#edc8ff',alpha = 0.3,ax=ax_,label='动态最大涨幅（右）')
            x_between = temp_pct.index
            ax1.fill_between(x=x_between, y1=maxup(temp_pct), y2=0, alpha=0.5,
                             facecolor='#edc8ff', label='动态最大涨幅（左）')  # label='动态最大涨幅（左）'
            ax1.set_ylim(-0.05, 1)
        #         ax1.legend(loc=2)
        else:
            # 1.1.2 最大回撤
            # sns.lineplot(temp.index,-maxdown(temp_pct),color='#edc8ff',alpha = 0.3,ax=ax_,label='动态最大回撤（右）')
            x_between = temp_pct.index
            ax1.fill_between(x=x_between, y1=-maxdown(temp_pct), y2=0, alpha=0.5,
                             facecolor='#edc8ff', label='动态最大回撤（左）')  # label='动态最大回撤（左）'
            ax1.set_ylim(-1, 0.05)
        #         ax1.legend(loc=2)

        # 1.2-------------------------------------------------------------------------------------------------
        ax_ = ax1.twinx()
        sns.lineplot(temp_pct.index, temp_pct, color='#448ee4', alpha=0.5, ax=ax_, )  # label='行业累计超额收益（右）'
        sns.lineplot(temp_pct.index, temp_pct.rolling(30).mean(), color='#448ee4', ax=ax_,
                     label='行业累计超额收益（右）')  # label='行业累计超额收益（右）'
        # ax_.scatter(self.buy_signal(), temp_pct.loc[self.buy_signal()], marker='*', s=150, c='y')
        #     ax_.scatter(sell_signal(item),temp_pct.loc[sell_signal(item)],marker='*',s=150,c ='k')
        ax_.set_xlabel('')
        ax_.set_xticks([])
        ax_.set_title(item)
        #     ax_.legend(loc=1)

        # 2. --------------------------------------------------------------------------------------------------
        ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=1, colspan=1, sharex=ax1)
        sns.lineplot(temp_report.index, relative_count, color='#ff5b00', alpha=0.5, ax=ax2, )
        sns.lineplot(temp_report.index, relative_count.rolling(5).mean(), color='#ff5b00', ax=ax2, label='研报推荐相对数量（左）')
        ax2.set_ylabel('')

        # 3. --------------------------------------------------------------------------------------------------
        ax3 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, colspan=1, sharex=ax1)
        sns.lineplot(temp_report.index, temp_report[item], color='#ff5b00', alpha=0.5, ax=ax3, )
        sns.lineplot(temp_report.index, temp_report[item].rolling(5).mean(), color='#ff5b00', ax=ax3,
                     label='研报推荐绝对数量（左）')
        #     sns.lineplot(temp_pct.index,strategy(item)['strategy'],color='#ff5b00',ax=ax3,label='策略收益')
        ax3.set_ylabel('')
        ax3.set_xlabel('报告日期')
        plt.subplots_adjust(wspace=0, hspace=0)


# ##################################其它辅助的函数##################################
def clean_search(x):
    q = []

    # 判断有无括号，有括号的需要特殊提取
    if len(re.findall('([\(\（].*?[\)\）])', x)) > 0:
        r = re.findall('([\(\（].*?[\)\）])', x)

        # 针对每个括号中的信息单独处理
        # 1. 从原文中去除括号
        # 2. 找出分隔符进行切分
        # 3. 组合成list
        for each in r:
            x = x.replace(each, '')
            if len(re.findall('[,，、]', each)) > 0:
                q = q + each[1:-1].split(re.findall('[,，、]', each)[0])
            else:
                q.append(each[1:-1])

        # 单独处理去除括号后的原文
        cut = x.split(',')

    # 没有括号的单独处理
    else:
        cut = x.split(',')

    if len(q) > 0:
        return cut + q
    else:
        return cut


def trans(lis):
    temp = []
    for i in lis:
        if i in trans_dict_df['原有名称'].tolist():
            temp = temp + trans_dict_df.loc[trans_dict[i], '现有名称']
        else:
            temp.append(i)
    return temp


def fuzzy_match(lis, ind1, ind2, ind3):
    original = []
    standard = []
    ind = ind1 + ind2 + ind3
    ind = list(set(ind))
    for i in lis:
        for j in ind:
            if j in i:
                standard.append(j)
                original.append(i)
    return standard, original


def next_trading_day(certain_day):
    next_day = certain_day + ONE_DAY
    while next_day not in TRADING_DAY['date']:
        next_day += ONE_DAY
    return next_day


def map_next_trading_day(c):
    return list(map(next_trading_day, c))


def maxup(data):
    data = pd.DataFrame(data)
    roll_min = data.expanding().min()
    return np.max(data / roll_min - 1, axis=1)


def maxdown(data):
    data = pd.DataFrame(data)
    roll_max = data.expanding().max()
    return np.max(1 - data / roll_max, axis=1)  # 回撤值是一个正数，0到1之间


if __name__ == '__main__':
    pdata = datapre()
    for item in sw1['SEC_NAME']:
        q = datastrategy(item, pdata)
        q.itemplot()
        plt.savefig(f'./pic/{item}.png')
