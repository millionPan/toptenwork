# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:09:56 2024

@author: pan
"""

# conda activate streamlit_test
# cd /d d:\topten\topten
# streamlit run topten_local.py --server.port 8501

# print(np.__version__)
import os
# os.chdir('d:/topten/topten')

#获取当天交易额前n名
import akshare as ak
import pandas as pd #2.2.2

import datetime

import streamlit as st
import xgboost as xgb #
import numpy as np #1.26.4



#全局配置
st.set_page_config(
    page_title="work",    #页面标题
    page_icon=":rainbow:",        #icon:emoji":rainbow:"
    layout="wide",                #页面布局
    initial_sidebar_state="auto"  #侧边栏
)

# st.write(xgb.__version__)
date_choose=st.date_input(label="choose",value=datetime.date.today(),label_visibility="collapsed")
enddate=date_choose.strftime("%Y%m%d")
#当天运行
# enddate=datetime.date.today().strftime("%Y%m%d")
# enddate="20240628"


@st.cache_data
def industry(enddate):
    r_stock_info=pd.read_excel("./stock_info.xlsx")
    r_stock_info['symbol']=r_stock_info['ts_code'].apply(lambda x :x[:6])
    return r_stock_info
r_stock_info=industry(enddate)

@st.cache_data
def gettcal(enddate):
    tcal = (ak.tool_trade_date_hist_sina()
            .loc[lambda x :x['trade_date']>=datetime.date(2024,1,1)])#0.1s
    
    # type(tcal.iloc[0].iloc[0])
    tcal['date_str']=tcal['trade_date'].apply(lambda x:x.strftime("%Y-%m-%d"))
    tcal['datestr']=tcal['trade_date'].apply(lambda x:x.strftime("%Y%m%d"))
    return tcal
tcal=gettcal(enddate)
#today=datetime.date(2024,5,7)



path='/'.join(['.','dailymarket','daily'])


# 预测某一天
#找到最接近enddate的交易日
newest_tradeday=tcal.loc[tcal['datestr']<=enddate]['datestr'].iloc[-1]
#读取最近交易日的当天行情信息 
todaystr=datetime.date.today().strftime("%Y%m%d")

if newest_tradeday < todaystr:
    try:
        try:
            ashare=pd.read_excel(path+'/'+newest_tradeday+".xlsx").drop(['Unnamed: 0', '序号'],axis=1)
        except:
            ashare=pd.read_excel(path+'/'+newest_tradeday+".xlsx")
        #ashare.columns
        ashare['代码']=ashare['代码'].apply(lambda x :str(x).zfill(6))
    except:
        st.write('没有该日交易数据可以用来预测噢！')
        raise SystemExit
    
else:    
    #当天运行
    ashare=ak.stock_zh_a_spot_em().drop(['序号'],axis=1)
    ashare['date']=datetime.datetime.strptime(datetime.date.today().strftime("%Y%m%d"),'%Y%m%d')
    # ashare.to_excel('d:/akss1.xlsx',index=False)
#
  
today_data_ac=(ashare.sort_values(by=['成交额'],ascending=False).iloc[:20]
     .loc[lambda x :~x['名称'].str.contains('ST')]
     .loc[lambda x :(x['代码'].str.startswith('60'))|(x['代码'].str.startswith('00'))]
     .loc[lambda x :(x['最新价']>5)&(x['最新价']<50)]
    ).iloc[:10]

havelist=[]#[]
havedata=ashare[ashare['代码'].isin(havelist)]
if len(havelist)==0:
    today_data=today_data_ac.copy()
else:
    today_data=pd.concat([today_data_ac,havedata],axis=0).drop_duplicates(keep='first').sort_values(by=['成交额'],ascending=False)



@st.cache_data
def getmerge(enddate):
    merge_data=pd.read_excel("./dailymarket/dailymarketmerge.xlsx")
    return merge_data
merge_data=getmerge(enddate)




replcl = {'index':'序号', 'code':'代码', 'name':'名称','tclose': '最新价_x','pct_chg': '涨跌幅_x',
          'change':'涨跌额','vol':'成交量','amount': '成交额',
                 'pct_range':'振幅','thigh': '最高_x','tlow': '最低_x','topen': '今开_x','lclose': '昨收', 
                 'volrate':'量比','turnover': '换手率','PE': '市盈率-动态','PB': '市净率', 'totalmv':'总市值',
                 'market_equity':'流通市值','up_rate' :'涨速', '5min_rf':'5分钟涨跌',
                 '60days_pct_chg': '60日涨跌幅','year_pct_chg': '年初至今涨跌幅',
                 'next_close': '最新价_y','next_pct_chg': '涨跌幅_y',
                 'next_high':'最高_y','next_low': '最低_y', 'next_open':'今开_y',}

def swap_key_value(old_dict):
   new_dict = {key:value for value,key in old_dict.items()}
   return new_dict
#中译英
cte=swap_key_value(replcl)

def translatecolname(df,fromto):
    edited_col=df.columns.to_list()
    if fromto=='cte':
        repl=cte
    else:
        repl=replcl
    tran_col=[repl[i] if i in repl else i for i in edited_col]
    return tran_col

#目标列
profit_pct=1.5
merge_data['ycol']=merge_data.apply(lambda x :1 if x['high_rate']>profit_pct else 0,axis=1) 

#
def predictwhole(merge_data,today_data,startdate,model_enddate,trainr):
    # print('444-----------')
    # startdate='20240101'
    # model_enddate=tcal.loc[tcal['datestr']<newest_tradeday]['datestr'].iloc[-2]
    #'20240601'
    # trainr=0.8
    
    origin_data=(merge_data.loc[lambda x :x['date']>=datetime.datetime.strptime(startdate,'%Y%m%d')]
    .loc[lambda x :x['date']<=datetime.datetime.strptime(model_enddate,'%Y%m%d')])
    
    today_data.rename(columns={'最新价':'最新价_x', '涨跌幅':'涨跌幅_x','最高':'最高_x', '最低':'最低_x', '今开':'今开_x'},inplace=True)
    historydata=pd.concat([origin_data,today_data],axis=0,ignore_index=True)
    
    historydata['代码']=historydata['代码'].apply(lambda x :str(x).zfill(6))
    historydata.columns=translatecolname(historydata,fromto='cte')
    
    historydata["updiff"]=historydata.apply(lambda x :x['thigh']-max(x['topen'],x['tclose']),axis=1)
    historydata["downdiff"]=historydata.apply(lambda x :min(x['topen'],x['tclose'])-x['tlow'],axis=1)
    
    
    historydata["oc"]=(historydata['topen']-historydata['lclose'])/historydata['lclose']#今开-昨收
    historydata["cc"]=(historydata['tclose']-historydata['lclose'])/historydata['lclose']#今收-昨收
    historydata["hc"]=(historydata['thigh']-historydata['lclose'])/historydata['lclose']#今高-昨高
    historydata["lc"]=(historydata['tlow']-historydata['lclose'])/historydata['lclose']#今低-昨高
    historydata["eb"]=(historydata['PE']/historydata['PB'])
    historydata["co"]=(historydata['tclose']-historydata['topen'])/historydata['topen']#今开-昨收
    
    #划分训练模型的数据（训练集测试集）-包含模型截止日期和专门用于预测的数据
    #type(historydata['日期'][0])
    #非etf的日期本来就是日期格式，不用转换
    #historydata['日期']=historydata['日期'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d').date())
    scaled_data=historydata.loc[lambda x :x['date']<=datetime.datetime.strptime(model_enddate,'%Y%m%d')]
    npridict_data=historydata.loc[lambda x :x['date']>datetime.datetime.strptime(model_enddate,'%Y%m%d')]
    #是否按顺序
    scaled_data=scaled_data.sample(frac=1,random_state=16).reset_index(drop=True)
    
    ycol='ycol'
    
    #pd增大测试集
    n_train=int(scaled_data.shape[0]*trainr)
    train_XGB= scaled_data.iloc[:n_train].dropna()
    test_XGB = scaled_data.iloc[n_train:].dropna()
    #pd预测最新一天
    # train_XGB= scaled_data[(pd.notna(scaled_data[ycol]))&(pd.notna(scaled_data['oo']))]
    npredict_XGB = npridict_data[pd.isna(npridict_data[ycol])]
    
    #historydata.columns
    #'up_rate', '5min_rf','60days_pct_chg', 'pct_range', 'year_pct_chg', 'updiff', 'downdiff', 'oc', 'cc', 'hh', 'lh'
    xlist_four=(['volrate','turnover', 'oc', 'cc', 'hc', 'lc', 'pct_range',
                 'updiff', 'downdiff','PB','PE'
                 
                  ])#变量列
    xlist=[xlist_four]
    #444--
    params_four = {
        'booster':'gbtree',
        'objective':'binary:logistic',  # binary:logistic此处为回归预测，这里如果改成multi:softmax 则可以进行多分类
        'gamma':0.1,
        'max_depth':4,
        'lambda':4,
        'subsample':trainr,
        'colsample_bytree':trainr,
        'min_child_weight':3,
        'verbosity':1,
        'eta':0.1,
        'seed':1000,
        
    }
    
    paramslist=[params_four]
    
       
    for model in range(len(paramslist)): 
        # print(model)
        #model=0
        train_XGB_X, train_XGB_Y = train_XGB[xlist[model]],train_XGB.loc[:,ycol]
        test_XGB_X, test_XGB_Y = test_XGB[xlist[model]],test_XGB.loc[:,ycol]
        
        npredict_XGB_X, npredict_XGB_Y = npredict_XGB[xlist[model]],npredict_XGB.loc[:,ycol]
        
        
        #生成数据集格式
        xgb_train = xgb.DMatrix(train_XGB_X,label = train_XGB_Y)
        # xgb_test = xgb.DMatrix(test_XGB_X,label = test_XGB_Y)
        xgb_test = xgb.DMatrix(test_XGB_X)
        
        npredict_test = xgb.DMatrix(npredict_XGB_X) 
        
        num_rounds =120
        #watchlist = [(xgb_test,'eval'),(xgb_train,'train')]
        # watchlist = [(xgb_train,'train')]
        # 
        # print(xgb.__version__)
    
    
        #xgboost模型训练
    
        # print(params)
        #params=paramslist[0]
        model_xgb = xgb.train(paramslist[model],xgb_train,num_rounds)
        
        #%matplotlib qt5
        # xgb.plot_importance(model_xgb)
        
        #对测试集进行预测
        y_pred_xgb = model_xgb.predict(xgb_test)
        
        #将测试集的预测Y转换成数据框，加上时间index
        testy=pd.DataFrame(y_pred_xgb,index=test_XGB_Y.index,columns=['y_pred_xgb_t']).astype(float)
        
        #对明日涨跌进行预测
        npredict_XGB_Y = model_xgb.predict(npredict_test)
        predicty=pd.DataFrame()
        predicty['code']=npredict_XGB['code']
        predicty['name']=npredict_XGB['name']
        predicty['n_pred']= npredict_XGB_Y.astype(float)
        predicty['price']=round(npredict_XGB['tclose'],2)
        predicty['top_high']=round(npredict_XGB['tclose']*1.015,2)#tomorrow_predict_high
        
        # 将测试集的预测Y转换成数据框，加上时间index
        testy=pd.DataFrame(y_pred_xgb,index=test_XGB_Y.index,columns=['y_pred_xgb_t']).astype(float)
        
        
        # 对训练集进行预测
        y_pred_xgb_t = model_xgb.predict(xgb_train)
        
        # 将训练集的预测Y转成数据框
        y_pred_xgb_d=pd.DataFrame(y_pred_xgb_t,index=train_XGB_Y.index,columns=['y_pred_xgb_t']).astype(float)
        
        # 将训练集真实Y和预测Y合并数据框并求准确率
        train_acc=pd.concat([train_XGB_Y,y_pred_xgb_d],axis=1)
        train_acc.loc[train_acc['y_pred_xgb_t']>0.5,'predict']=1
        train_acc.loc[train_acc['y_pred_xgb_t']<=0.5,'predict']=0
        
        #后30个准确率
        TP=len(train_acc.iloc[:].loc[(train_acc['ycol']==1)&(train_acc['predict']==1)])
        TN=len(train_acc.iloc[:].loc[(train_acc['ycol']==0)&(train_acc['predict']==0)])
        FP=len(train_acc.iloc[:].loc[(train_acc['ycol']==0)&(train_acc['predict']==1)])
        FN=len(train_acc.iloc[:].loc[(train_acc['ycol']==1)&(train_acc['predict']==0)])
        st.write("Accuracy: "+str(round((TP+TN)/(TP+FP+FN+TN), 4)))
        st.write("WRONG: "+str(round((FP)/(TP+FP+FN+TN), 4)))
        st.write("tpr: "+str(round((TP)/(TP+FP), 4)))
        #将测试集真实Y和预测Y合并数据框并求准确率
        test_acc=pd.concat([test_XGB_Y,testy],axis=1)
        test_acc.loc[test_acc['y_pred_xgb_t']>0.5,'predict']=1
        test_acc.loc[test_acc['y_pred_xgb_t']<=0.5,'predict']=0
        # test_acc['date']=test_XGB['date']
        # 准确率
        TP=len(test_acc.loc[(test_acc['ycol']==1)&(test_acc['predict']==1)])
        TN=len(test_acc.loc[(test_acc['ycol']==0)&(test_acc['predict']==0)])
        FP=len(test_acc.loc[(test_acc['ycol']==0)&(test_acc['predict']==1)])
        FN=len(test_acc.loc[(test_acc['ycol']==1)&(test_acc['predict']==0)])
        acc=round((TP+TN)/(TP+FP+FN+TN), 4)#accuracy
        # fpr=round((FP)/(TP+FP+FN+TN), 4)#false positive rate
        if TP==0:
            tpr=0
        else:
            tpr=round((TP)/(TP+FP), 4)#在所有预测为正类别,实际为正类别比例。
        if TN==0:
            tnr=0
        else:
            tnr=round((TN)/(TN+FN), 4)#在所有预测为负类别,实际为负类别比例。
        
        st.write("acc: "+str(round(acc, 4)))
        st.write("tpr: "+str(round(tpr, 4)))
        st.write("tnr: "+str(round(tnr, 4)))
        
        xgbdata=predicty.copy()
        
        xgbdata['operate']=xgbdata.apply(lambda x :1 if x['n_pred']>0.5 else 0,axis=1) 
        xgbdata['n_pred']=xgbdata['n_pred'].apply(lambda i :pd.to_numeric(i))
        
        # print(xgbdata.sort_values(by=['n_pred'],ascending=False))
    return xgbdata
    

if st.button('跑模型啦！', type="primary"): 
    trainr=0.8
    startdate='20240101'
    model_enddate_last=tcal.loc[tcal['datestr']<newest_tradeday]['datestr'].iloc[-2]
    model_enddate_best='20240531'
    #0531  Accuracy: 0.975  WRONG: 0.025 tpr: 0.9538
    if model_enddate_best>=newest_tradeday:
        st.write("error")
        raise SystemExit
    else: 
        col1, col2 = st.columns(2)

        with col1:
           predictdata_last=predictwhole(merge_data=merge_data,today_data=today_data,startdate=startdate,model_enddate=model_enddate_last,trainr=trainr)
        with col2:
           predictdata_best=predictwhole(merge_data=merge_data,today_data=today_data,startdate=startdate,model_enddate=model_enddate_best,trainr=trainr)

        predictdata=pd.merge(predictdata_last,predictdata_best,how='left',on=['code','name'])
        
        predictdata['operate']=predictdata.apply(lambda x :1 if ((x['n_pred_x']>0.5)&(x['n_pred_y']>0.5)) else 0,axis=1) 
        
        predictdata['n_pred']=predictdata.apply(lambda x:np.mean(x[['n_pred_x','n_pred_y']]),axis=1)
        predictdata=(predictdata.sort_values(by=['n_pred'],ascending=False)
        .merge(r_stock_info[['symbol','industry']],how='left',left_on='code',right_on='symbol')
        .drop('symbol', axis=1))
        #区分
        predictdata.loc[predictdata['code'].isin(havelist),'type']='have'
        comment_element=set(havelist)&set(today_data_ac['代码'].tolist())
        predictdata.loc[predictdata['code'].isin(today_data_ac['代码'].tolist()),'type']='ac'
        predictdata.loc[predictdata['code'].isin(comment_element),'type']='have&ac'
        #
        bbb=predictdata.columns.to_list()
        invaildcol=['code','name','industry','n_pred','operate','type','n_pred_x','n_pred_y']
        new_col=[x for x in bbb if x not in invaildcol]
        bbb =invaildcol + new_col
        predictdata=predictdata.reindex(columns=bbb).sort_values(by=['n_pred'],ascending=False)
        # print("two_model")
        # print(predictdata)
        
        
        # predictdata=(predictwhole(merge_data=merge_data,today_data=today_data,startdate=startdate,model_enddate=model_enddate_last,trainr=trainr)       
        # .sort_values(by=['n_pred'],ascending=False)
        # .merge(r_stock_info[['symbol','industry']],how='left',left_on='code',right_on='symbol')
        # .drop('symbol', axis=1))
        # predictdata.loc[predictdata['code'].isin(havelist),'type']='have'
        # comment_element=set(havelist)&set(today_data_ac['代码'].tolist())
        # predictdata.loc[predictdata['code'].isin(today_data_ac['代码'].tolist()),'type']='ac'
        # predictdata.loc[predictdata['code'].isin(comment_element),'type']='have&ac'
        # print("newest_modelenddate:")
        # print(predictdata)
        
        # predictdata=(predictwhole(merge_data=merge_data,today_data=today_data,startdate=startdate,model_enddate=model_enddate_best,trainr=trainr)       
        # .sort_values(by=['n_pred'],ascending=False)
        # .merge(r_stock_info[['symbol','industry']],how='left',left_on='code',right_on='symbol')
        # .drop('symbol', axis=1))
        # predictdata.loc[predictdata['code'].isin(havelist),'type']='have'
        # comment_element=set(havelist)&set(today_data_ac['代码'].tolist())
        # predictdata.loc[predictdata['code'].isin(today_data_ac['代码'].tolist()),'type']='ac'
        # predictdata.loc[predictdata['code'].isin(comment_element),'type']='have&ac'
        # print("best_modelenddate")
        # print(predictdata)
    
    
        
    ##############################################
    ##验证   

    
    last_predict=predictdata.copy()
    
    #1/获取需要预测日期的实际数据
    symbollist=last_predict['code'].tolist()
    #pradate='20240619'   
    pradate=tcal.loc[tcal['datestr']>newest_tradeday]['datestr'].iloc[0]
    
    if (pradate<=datetime.datetime.today().date().strftime("%Y%m%d")):#下一交易日比今天大
        realtimedata=pd.DataFrame()
        for symbol in symbollist:
            if (str(symbol)[0]=="5")|(str(symbol)[0]=="1"):
                historydata = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=pradate, end_date=pradate, adjust="")
            else:
                historydata = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=pradate, end_date=pradate, adjust="")
            historydata.rename(columns={"收盘":"price"},inplace=True)
            historydata['code']=symbol
            realtimedata=pd.concat([realtimedata,historydata],axis=0,ignore_index=True)
        realtimedata['pre_close']=realtimedata['price']-realtimedata['涨跌额']
        realtimedata['high_rate']=(realtimedata['最高']-realtimedata['pre_close'])/realtimedata['pre_close']*100
        
        realtimedata=realtimedata[['code','high_rate']]
        
        
        # =============================================================================
        # 
        # =============================================================================
        #昨日预测匹配当天实时数据，对比
        compared_data=last_predict.merge(realtimedata,how='left',on='code')
        compared_data['pra_ud']=compared_data.apply(lambda x :1 if x['high_rate']>profit_pct else 0,axis=1)  #pratical_updown  
        
        #分模型查看预测效果
        compared_data['jude_wo']=compared_data.apply(lambda x :1 if x['pra_ud']==x['operate'] else 0,axis=1) 
        
        st.write("accpre_wo:"+str(compared_data['jude_wo'].sum()/compared_data.shape[0]))#accuracy_predict_four
        
        #调整列顺序'acc','tpr','tnr'
        bbb=compared_data.columns.to_list()
        invaildcol=['code','name','industry','jude_wo','pra_ud','operate','type','n_pred']
        new_col=[x for x in bbb if x not in invaildcol]
        bbb =invaildcol + new_col
        compared_data=compared_data.reindex(columns=bbb).sort_values(by=['n_pred'],ascending=False)
    else:
        st.write("还没第二个交易日的数据可以验证喔!")
    
    yellow_css = 'background-color: yellow'
    sfun = lambda x: [yellow_css]*len(x) if x.n_pred_y >= 0.8 else ['']*len(x)
    
    if 'compared_data' in globals():
        st.dataframe(compared_data,hide_index=True)
        
    else:
        st.dataframe(predictdata.style.apply(sfun,axis=1),hide_index=True,
                     column_order=("code", "name",'n_pred','operate','n_pred_y','n_pred_x','price_x','top_high_x',
                                   ),
                     column_config={
                            "n_pred": st.column_config.NumberColumn(format="%.2f",),
                            "price_x": st.column_config.NumberColumn(format="%d",),
                            "top_high_x": st.column_config.NumberColumn(format="%.2f",),},)
        

     



