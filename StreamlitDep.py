import streamlit as st
import time
import pandas as pd
import pandas_datareader as dr
import numpy as np
from yahoofinancials import YahooFinancials
from datetime import date
# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import date

#Import Model Packages 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import cluster, covariance, manifold

#Package for optimization of mean variance optimization
import cvxopt as opt
from cvxopt import blas, solvers


algo = "MVP"
st.set_page_config(layout="wide")
col1, mid, col2 = st.columns([15,5,50])
with col1:
    st.image('Logo_header@2x.jpg',width=300)
with col2:
    st.title('Portfolio Optimization using Reinforcement Learning')
    st.subheader("By Group 4")

n50stocks = ["ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "HCLTECH.NS",
            "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "IOC.NS",
            "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
            "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SUNPHARMA.NS", "TATACONSUM.NS", 
            "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS", "HDFC.NS", 
            "DRREDDY.NS", "DIVISLAB.NS", "BRITANNIA.NS", "COALINDIA.NS", "EICHERMOT.NS", "SAIL.NS", "PNB.NS", 
            "AUROPHARMA.NS", "GAIL.NS"]
stocks = ["TATAMOTORS.NS", "ASHOKLEY.NS", "EICHERMOT.NS", "MARUTI.NS", "TVSMOTOR.NS", "MOTHERSON.NS", "BHARATFORG.NS", "BAJAJ-AUTO.NS", "BALKRISIND.NS", "M&M.NS",
          "AUBANK.NS", "INDUSINDBK.NS", "PNB.NS", "SBIN.NS", "FEDERALBNK.NS", "HDFCBANK.NS", "KOTAKBANK.NS", "IDFCFIRSTB.NS", "ICICIBANK.NS", "BANKBARODA.NS",
          "BAJFINANCE.NS", "ADANIENT.NS", "TATAMOTORS.NS", "INDUSINDBK.NS", "AXISBANK.NS", "DIVISLAB.NS", "ICICIBANK.NS", "HDFC.NS", "KOTAKBANK.NS", "ITC.NS",
          "VBL.NS", "DABUR.NS", "UBL.NS", "HINDUNILVR.NS", "BRITANNIA.NS", "EMAMILTD.NS", "COLPAL.NS", "NESTLEIND.NS", "GODREJCP.NS", "MCDOWELL-N.NS",
          "MAXHEALTH.NS", "AUROPHARMA.NS", "IPCALAB.NS", "BIOCON.NS", "SYNGENE.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", "METROPOLIS.NS", "TORNTPHARM.NS", "ALKEM.NS",
          "HCLTECH.NS", "COFORGE.NS", "MPHASIS.NS", "LTIM.NS", "TECHM.NS", "LTTS.NS", "TCS.NS", "WIPRO.NS", "PERSISTENT.NS", "INFY.NS",
          "NDTV.NS", "DISHTV.NS", "HATHWAY.NS", "PVR.NS", "NAVNETEDUL.NS", "TV18BRDCST.NS", "SUNTV.NS", "NAZARA.NS", "NETWORK18.NS", "ZEEL.NS",
          "GODREJPROP.NS", "DLF.NS", "OBEROIRLTY.NS", "SOBHA.NS", "MAHLIFE.NS", "LODHA.NS", "PRESTIGE.NS", "IBREALEST.NS", "BRIGADE.NS", "PHOENIXLTD.NS"]
sel = None
choice = st.selectbox('Choose an option',
                      ['Nifty50', 'Choose Sector','Inter-sector'],0)

if choice == 'Nifty50':
    if st.button("Show stocks in Nifty50"):
        st.write("These are stocks in Nifty50:")
        for i in n50stocks:
            st.text(i)
    sel = st.multiselect("Select stocks from the list(Click on 'Show stocks' above for the list):",n50stocks,n50stocks[0])
    amt = st.text_input('Enter amount for investment:',placeholder ="Enter amount")
    if st.button("Submit"):
        st.write("These are the recommended investments for the selected stocks:")
        #for i in sel:
            #st.text(i)
        tickers = np.array(sel)
        yahoo_financials = YahooFinancials(np.array(sel))
        data = yahoo_financials.get_historical_price_data(start_date='2020-04-01', 
                                                    end_date='2023-03-31', 
                                                    time_interval='daily')
        prices_df = pd.DataFrame({a: {x['formatted_date']: x['close'] for x in data[a]['prices']} for a in tickers})
        #st.table(prices_df.head())
        with st.spinner("Getting results..."):
            dataset = prices_df.copy()

            #dataset = pd.read_csv("autoret.csv")
            missing_fractions = dataset.isnull().mean().sort_values(ascending=False)

            missing_fractions.head(10)

            drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

            dataset.drop(labels=drop_list, axis=1, inplace=True)

            dataset=dataset.fillna(method='ffill')

            X = dataset.copy()
            row= len(X)
            train_len = int(row*.8)
            #train_len = int(row*.8846)

            X_train = dataset.head(train_len)

            X_test = dataset.tail(row-train_len)

            returns =  X_train.pct_change().dropna() #pd.read_csv("autoret.csv") #X_train.pct_change().dropna()
            returns_test =  X_test.pct_change().dropna() #pd.read_csv("autoret_test.csv") #X_test.pct_change().dropna()
            def correlDist(corr):
            # A distance matrix based on correlation, where 0<=d[i,j]<=1
            # This is a proper distance metric
                dist = ((1 - corr) / 2.)**.5  # distance matrix
                return dist
            #Calulate linkage
            dist = correlDist(returns.corr())
            link = linkage(dist, 'ward')
            #link[0]

            def getQuasiDiag(link):
            # Sort clustered items by distance
                link = link.astype(int)
                sortIx = pd.Series([link[-1, 0], link[-1, 1]])
                numItems = link[-1, 3]  # number of original items
                while sortIx.max() >= numItems:
                    sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
                    df0 = sortIx[sortIx >= numItems]  # find clusters
                    i = df0.index
                    j = df0.values - numItems
                    sortIx[i] = link[j, 0]  # item 1
                    df0 = pd.Series(link[j, 1], index=i + 1)
                    sortIx = sortIx.append(df0)  # item 2
                    sortIx = sortIx.sort_index()  # re-sort
                    sortIx.index = range(sortIx.shape[0])  # re-index
                return sortIx.tolist()

            def getClusterVar(cov,cItems):
                # Compute variance per cluster
                cov_=cov.loc[cItems,cItems] # matrix slice
                w_=getIVP(cov_).reshape(-1,1)
                cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
                return cVar

            def getRecBipart(cov, sortIx):
            # Compute HRP alloc
                w = pd.Series(1, index=sortIx)
                cItems = [sortIx]  # initialize all items in one cluster
                while len(cItems) > 0:
                    cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
                    for i in range(0, len(cItems), 2):  # parse in pairs
                        cItems0 = cItems[i]  # cluster 1
                        cItems1 = cItems[i + 1]  # cluster 2
                        cVar0 = getClusterVar(cov, cItems0)
                        cVar1 = getClusterVar(cov, cItems1)
                        alpha = 1 - cVar0 / (cVar0 + cVar1)
                        w[cItems0] *= alpha  # weight 1
                        w[cItems1] *= 1 - alpha  # weight 2
                return w

            def getMVP(cov):
                cov = cov.T.values
                n = len(cov)
                N = 100
                mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

                # Convert to cvxopt matrices
                S = opt.matrix(cov)
                #pbar = opt.matrix(np.mean(returns, axis=1))
                pbar = opt.matrix(np.ones(cov.shape[0]))

                # Create constraint matrices
                G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
                h = opt.matrix(0.0, (n, 1))
                A = opt.matrix(1.0, (1, n))
                b = opt.matrix(1.0)


                # Calculate efficient frontier weights using quadratic programming
                solvers.options['show_progress'] = False
                portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                    for mu in mus]
                ## CALCULATE RISKS AND RETURNS FOR FRONTIER    
                returns = [blas.dot(pbar, x) for x in portfolios]
                risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
                ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
                m1 = np.polyfit(returns, risks, 2)
                x1 = np.sqrt(m1[2] / m1[0])
                # CALCULATE THE OPTIMAL PORTFOLIO    
                wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
                return list(wt)

            def getIVP(cov, **kargs):
                # Compute the inverse-variance portfolio
                ivp = 1. / np.diag(cov)
                ivp /= ivp.sum()
                return ivp

            def getHRP(cov, corr):
                # Construct a hierarchical portfolio
                dist = correlDist(corr)
                link = sch.linkage(dist, 'single')
                #plt.figure(figsize=(20, 10))
                #dn = sch.dendrogram(link, labels=cov.index.values)
                #plt.show()
                sortIx = getQuasiDiag(link)
                sortIx = corr.index[sortIx].tolist()
                hrp = getRecBipart(cov, sortIx)
                return hrp.sort_index()

            def get_req_portfolios(returns):
                cov, corr = returns.cov(), returns.corr()
                if algo == 'HRP':
                    hrp = round(getHRP(cov, corr),2)
                    portfolios = pd.DataFrame([hrp], index=['Amounts']).T
                else:
                    mvp = getMVP(cov)
                    mvp = pd.Series(mvp, index=cov.index)
                    portfolios = pd.DataFrame([round(mvp,2)], index=['Amounts']).T
                #portfolios = pd.DataFrame([ivp, hrp], index=['IVP', 'HRP']).T
                return portfolios

            portfolios = get_req_portfolios(returns)
            portfolios.iloc[:,0] = portfolios.iloc[:,0]*int(amt)
            portfolios.reset_index(level=0, inplace=True)
            portfolios.rename(columns = {'index':'Stocks'}, inplace = True)
        st.table(portfolios.style.hide())        
elif choice == 'Inter-sector':
    if st.button("Show all available stocks"):
        st.write("These stocks are available for selection:")
        for i in stocks:
            st.text(i)
    sel = st.multiselect("Choose relevant stocks from the list(Click on 'Show stocks' above for the list):",stocks,stocks[0])
    amt = st.text_input('Enter amount for investment:',placeholder ="Enter amount")
    if st.button("Submit"):
        st.write("These are the recommended investments for the selected stocks:")
        for i in sel:
            st.text(i)
        tickers = np.array(sel)
        yahoo_financials = YahooFinancials(np.array(sel))
        data = yahoo_financials.get_historical_price_data(start_date='2020-04-01', 
                                                    end_date='2023-03-31', 
                                                    time_interval='daily')
        prices_df = pd.DataFrame({a: {x['formatted_date']: x['close'] for x in data[a]['prices']} for a in tickers})
        #st.table(prices_df.head())
        with st.spinner("Getting results..."):
            dataset = prices_df.copy()

            #dataset = pd.read_csv("autoret.csv")
            missing_fractions = dataset.isnull().mean().sort_values(ascending=False)

            missing_fractions.head(10)

            drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

            dataset.drop(labels=drop_list, axis=1, inplace=True)

            dataset=dataset.fillna(method='ffill')

            X = dataset.copy()
            row= len(X)
            train_len = int(row*.8)
            #train_len = int(row*.8846)

            X_train = dataset.head(train_len)

            X_test = dataset.tail(row-train_len)

            returns =  X_train.pct_change().dropna() #pd.read_csv("autoret.csv") #X_train.pct_change().dropna()
            returns_test =  X_test.pct_change().dropna() #pd.read_csv("autoret_test.csv") #X_test.pct_change().dropna()
            def correlDist(corr):
            # A distance matrix based on correlation, where 0<=d[i,j]<=1
            # This is a proper distance metric
                dist = ((1 - corr) / 2.)**.5  # distance matrix
                return dist
            #Calulate linkage
            dist = correlDist(returns.corr())
            link = linkage(dist, 'ward')
            #link[0]

            def getQuasiDiag(link):
            # Sort clustered items by distance
                link = link.astype(int)
                sortIx = pd.Series([link[-1, 0], link[-1, 1]])
                numItems = link[-1, 3]  # number of original items
                while sortIx.max() >= numItems:
                    sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
                    df0 = sortIx[sortIx >= numItems]  # find clusters
                    i = df0.index
                    j = df0.values - numItems
                    sortIx[i] = link[j, 0]  # item 1
                    df0 = pd.Series(link[j, 1], index=i + 1)
                    sortIx = sortIx.append(df0)  # item 2
                    sortIx = sortIx.sort_index()  # re-sort
                    sortIx.index = range(sortIx.shape[0])  # re-index
                return sortIx.tolist()

            def getClusterVar(cov,cItems):
                # Compute variance per cluster
                cov_=cov.loc[cItems,cItems] # matrix slice
                w_=getIVP(cov_).reshape(-1,1)
                cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
                return cVar

            def getRecBipart(cov, sortIx):
            # Compute HRP alloc
                w = pd.Series(1, index=sortIx)
                cItems = [sortIx]  # initialize all items in one cluster
                while len(cItems) > 0:
                    cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
                    for i in range(0, len(cItems), 2):  # parse in pairs
                        cItems0 = cItems[i]  # cluster 1
                        cItems1 = cItems[i + 1]  # cluster 2
                        cVar0 = getClusterVar(cov, cItems0)
                        cVar1 = getClusterVar(cov, cItems1)
                        alpha = 1 - cVar0 / (cVar0 + cVar1)
                        w[cItems0] *= alpha  # weight 1
                        w[cItems1] *= 1 - alpha  # weight 2
                return w

            def getMVP(cov):
                cov = cov.T.values
                n = len(cov)
                N = 100
                mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

                # Convert to cvxopt matrices
                S = opt.matrix(cov)
                #pbar = opt.matrix(np.mean(returns, axis=1))
                pbar = opt.matrix(np.ones(cov.shape[0]))

                # Create constraint matrices
                G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
                h = opt.matrix(0.0, (n, 1))
                A = opt.matrix(1.0, (1, n))
                b = opt.matrix(1.0)


                # Calculate efficient frontier weights using quadratic programming
                solvers.options['show_progress'] = False
                portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                    for mu in mus]
                ## CALCULATE RISKS AND RETURNS FOR FRONTIER    
                returns = [blas.dot(pbar, x) for x in portfolios]
                risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
                ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
                m1 = np.polyfit(returns, risks, 2)
                x1 = np.sqrt(m1[2] / m1[0])
                # CALCULATE THE OPTIMAL PORTFOLIO    
                wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
                return list(wt)

            def getIVP(cov, **kargs):
                # Compute the inverse-variance portfolio
                ivp = 1. / np.diag(cov)
                ivp /= ivp.sum()
                return ivp

            def getHRP(cov, corr):
                # Construct a hierarchical portfolio
                dist = correlDist(corr)
                link = sch.linkage(dist, 'single')
                #plt.figure(figsize=(20, 10))
                #dn = sch.dendrogram(link, labels=cov.index.values)
                #plt.show()
                sortIx = getQuasiDiag(link)
                sortIx = corr.index[sortIx].tolist()
                hrp = getRecBipart(cov, sortIx)
                return hrp.sort_index()

            def get_req_portfolios(returns):
                cov, corr = returns.cov(), returns.corr()
                if algo == 'HRP':
                    hrp = round(getHRP(cov, corr),2)
                    portfolios = pd.DataFrame([hrp], index=['Amounts']).T
                else:
                    mvp = getMVP(cov)
                    mvp = pd.Series(mvp, index=cov.index)
                    portfolios = pd.DataFrame([round(mvp,2)], index=['Amounts']).T
                #portfolios = pd.DataFrame([ivp, hrp], index=['IVP', 'HRP']).T
                return portfolios

            portfolios = get_req_portfolios(returns)
            portfolios.index.names = ['Stocks']
            portfolios.iloc[:,0] = portfolios.iloc[:,0]*int(amt)
        st.table(portfolios.iloc[:,0])
else:
    sector = st.selectbox("Select sector:",["AUTO","BANKING","FINANCIAL",
                                            "FMCG","HEALTHCARE","IT",
                                            "MEDIA","REALTY"],0)
    if sector == "AUTO":
        if st.button("Show stocks in AUTO sector"):
            st.write("These 10 stocks are in AUTO sector:")
            for i in stocks[:10]:
                st.text(i)
        ch = st.selectbox("Select the above 10 stocks or select custom stocks?(Click on Show stocks option above)",["Yes","Select Custom"],0)
        if ch == "Yes":
            sel = stocks[:10]
        else:
            sel = st.multiselect("Select stocks in AUTO sector:",stocks[:10],stocks[0])
    elif sector == "BANKING":
        if st.button("Show stocks in BANKING sector"):
            st.write("These 10 stocks are in BANKING sector:")
            for i in stocks[10:20]:
                st.text(i)
        ch = st.selectbox("Select the above 10 stocks or select custom stocks?(Click on Show stocks option above)",["Yes","Select Custom"],0)
        if ch == "Yes":
            sel = stocks[10:20]
        else:
            sel = st.multiselect("Select stocks in BANKING sector:",stocks[10:20],stocks[10])
    elif sector == "FINANCIAL":
        if st.button("Show stocks in FINANCIAL sector"):
            st.write("These 10 stocks are in FINANCIAL sector:")
            for i in stocks[20:30]:
                st.text(i)
        ch = st.selectbox("Select the above 10 stocks or select custom stocks?(Click on Show stocks option above)",["Yes","Select Custom"],0)
        if ch == "Yes":
            sel = stocks[20:30]
        else:
            sel = st.multiselect("Select stocks in FINANCIAL sector:",stocks[20:30],stocks[20])
    elif sector == "FMCG":
        if st.button("Show stocks in FMCG sector"):
            st.write("These 10 stocks are in FMCG sector:")
            for i in stocks[30:40]:
                st.text(i)
        ch = st.selectbox("Select the above 10 stocks or select custom stocks?(Click on Show stocks option above)",["Yes","Select Custom"],0)
        if ch == "Yes":
            sel = stocks[30:40]
        else:
            sel = st.multiselect("Select stocks in FMCG sector:",stocks[30:40],stocks[30])
    elif sector == "HEALTHCARE":
        if st.button("Show stocks in HEALTHCARE sector"):
            st.write("These 10 stocks are in HEALTHCARE sector:")
            for i in stocks[40:50]:
                st.text(i)
        ch = st.selectbox("Select the above 10 stocks or select custom stocks?(Click on Show stocks option above)",["Yes","Select Custom"],0)
        if ch == "Yes":
            sel = stocks[40:50]
        else:
            sel = st.multiselect("Select stocks in HEALTHCARE sector:",stocks[40:50],stocks[40])
    elif sector == "IT":
        if st.button("Show stocks in IT sector"):
            st.write("These 10 stocks are in IT sector:")
            for i in stocks[50:60]:
                st.text(i)
        ch = st.selectbox("Select the above 10 stocks or select custom stocks?(Click on Show stocks option above)",["Yes","Select Custom"],0)
        if ch == "Yes":
            sel = stocks[50:60]
        else:
            sel = st.multiselect("Select stocks in IT sector:",stocks[50:60],stocks[50])
    elif sector == "MEDIA":
        if st.button("Show stocks in MEDIA sector"):
            st.write("These 10 stocks are in MEDIA sector:")
            for i in stocks[60:70]:
                st.text(i)
        ch = st.selectbox("Select the above 10 stocks or select custom stocks?(Click on Show stocks option above)",["Yes","Select Custom"],0)
        if ch == "Yes":
            sel = stocks[60:70]
        else:
            sel = st.multiselect("Select stocks in MEDIA sector:",stocks[60:70],stocks[60])
    else:
        if st.button("Show stocks in REALTY sector"):
            st.write("These 10 stocks are in REALTY sector:")
            for i in stocks[70:80]:
                st.text(i)
        ch = st.selectbox("Select the above 10 stocks or select custom stocks?(Click on Show stocks option above)",["Yes","Select Custom"],0)
        if ch == "Yes":
            sel = stocks[70:80]
        else:
            sel = st.multiselect("Select stocks in REALTY sector:",stocks[70:80],stocks[70])
    amt = st.text_input('Enter amount for investment:',placeholder ="Enter amount")
    if st.button("Submit"):
        st.write("These are the recommended investments for the selected stocks:")
        for i in sel:
            st.text(i)
        tickers = np.array(sel)
        yahoo_financials = YahooFinancials(np.array(sel))
        data = yahoo_financials.get_historical_price_data(start_date='2020-04-01', 
                                                   end_date='2023-03-31', 
                                                   time_interval='daily')
        prices_df = pd.DataFrame({a: {x['formatted_date']: x['close'] for x in data[a]['prices']} for a in tickers})
        #st.table(prices_df.head())
        with st.spinner("Getting results..."):
            dataset = prices_df.copy()

            #dataset = pd.read_csv("autoret.csv")
            missing_fractions = dataset.isnull().mean().sort_values(ascending=False)

            missing_fractions.head(10)

            drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

            dataset.drop(labels=drop_list, axis=1, inplace=True)

            dataset=dataset.fillna(method='ffill')

            X = dataset.copy()
            row= len(X)
            train_len = int(row*.8)
            #train_len = int(row*.8846)

            X_train = dataset.head(train_len)

            X_test = dataset.tail(row-train_len)

            returns =  X_train.pct_change().dropna() #pd.read_csv("autoret.csv") #X_train.pct_change().dropna()
            returns_test =  X_test.pct_change().dropna() #pd.read_csv("autoret_test.csv") #X_test.pct_change().dropna()
            def correlDist(corr):
            # A distance matrix based on correlation, where 0<=d[i,j]<=1
            # This is a proper distance metric
                dist = ((1 - corr) / 2.)**.5  # distance matrix
                return dist
            #Calulate linkage
            dist = correlDist(returns.corr())
            link = linkage(dist, 'ward')
            #link[0]

            def getQuasiDiag(link):
            # Sort clustered items by distance
                link = link.astype(int)
                sortIx = pd.Series([link[-1, 0], link[-1, 1]])
                numItems = link[-1, 3]  # number of original items
                while sortIx.max() >= numItems:
                    sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
                    df0 = sortIx[sortIx >= numItems]  # find clusters
                    i = df0.index
                    j = df0.values - numItems
                    sortIx[i] = link[j, 0]  # item 1
                    df0 = pd.Series(link[j, 1], index=i + 1)
                    sortIx = sortIx.append(df0)  # item 2
                    sortIx = sortIx.sort_index()  # re-sort
                    sortIx.index = range(sortIx.shape[0])  # re-index
                return sortIx.tolist()

            def getClusterVar(cov,cItems):
                # Compute variance per cluster
                cov_=cov.loc[cItems,cItems] # matrix slice
                w_=getIVP(cov_).reshape(-1,1)
                cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
                return cVar

            def getRecBipart(cov, sortIx):
            # Compute HRP alloc
                w = pd.Series(1, index=sortIx)
                cItems = [sortIx]  # initialize all items in one cluster
                while len(cItems) > 0:
                    cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
                    for i in range(0, len(cItems), 2):  # parse in pairs
                        cItems0 = cItems[i]  # cluster 1
                        cItems1 = cItems[i + 1]  # cluster 2
                        cVar0 = getClusterVar(cov, cItems0)
                        cVar1 = getClusterVar(cov, cItems1)
                        alpha = 1 - cVar0 / (cVar0 + cVar1)
                        w[cItems0] *= alpha  # weight 1
                        w[cItems1] *= 1 - alpha  # weight 2
                return w

            def getMVP(cov):
                cov = cov.T.values
                n = len(cov)
                N = 100
                mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

                # Convert to cvxopt matrices
                S = opt.matrix(cov)
                #pbar = opt.matrix(np.mean(returns, axis=1))
                pbar = opt.matrix(np.ones(cov.shape[0]))

                # Create constraint matrices
                G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
                h = opt.matrix(0.0, (n, 1))
                A = opt.matrix(1.0, (1, n))
                b = opt.matrix(1.0)


                # Calculate efficient frontier weights using quadratic programming
                solvers.options['show_progress'] = False
                portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                    for mu in mus]
                ## CALCULATE RISKS AND RETURNS FOR FRONTIER    
                returns = [blas.dot(pbar, x) for x in portfolios]
                risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
                ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
                m1 = np.polyfit(returns, risks, 2)
                x1 = np.sqrt(m1[2] / m1[0])
                # CALCULATE THE OPTIMAL PORTFOLIO    
                wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
                return list(wt)

            def getIVP(cov, **kargs):
                # Compute the inverse-variance portfolio
                ivp = 1. / np.diag(cov)
                ivp /= ivp.sum()
                return ivp

            def getHRP(cov, corr):
                # Construct a hierarchical portfolio
                dist = correlDist(corr)
                link = sch.linkage(dist, 'single')
                #plt.figure(figsize=(20, 10))
                #dn = sch.dendrogram(link, labels=cov.index.values)
                #plt.show()
                sortIx = getQuasiDiag(link)
                sortIx = corr.index[sortIx].tolist()
                hrp = getRecBipart(cov, sortIx)
                return hrp.sort_index()

            def get_req_portfolios(returns):
                cov, corr = returns.cov(), returns.corr()
                if algo == 'HRP':
                    hrp = round(getHRP(cov, corr),2)
                    portfolios = pd.DataFrame([hrp], index=['Amounts']).T
                else:
                    mvp = getMVP(cov)
                    mvp = pd.Series(mvp, index=cov.index)
                    portfolios = pd.DataFrame([round(mvp,2)], index=['Amounts']).T
                #portfolios = pd.DataFrame([ivp, hrp], index=['IVP', 'HRP']).T
                return portfolios

            portfolios = get_req_portfolios(returns)
            portfolios.index.names = ['Stocks']
            portfolios.iloc[:,0] = portfolios.iloc[:,0]*int(amt)
        st.table(portfolios.iloc[:,0])


