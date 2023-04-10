import streamlit as st
import pandas as pd
import pandas_datareader as dr
from yahoofinancials import YahooFinancials
import numpy as np
import pandas as pd
from datetime import date


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
    if st.button("Submit"):
        st.write("These are the final selected stocks:")
        for i in sel:
            st.text(i)
        tickers = np.array(sel)
        yahoo_financials = YahooFinancials(np.array(sel))
        data = yahoo_financials.get_historical_price_data(start_date='2020-04-01', 
                                                    end_date='2023-03-31', 
                                                    time_interval='daily')
        prices_df = pd.DataFrame({a: {x['formatted_date']: x['close'] for x in data[a]['prices']} for a in tickers})
        st.table(prices_df.head())
elif choice == 'Inter-sector':
    if st.button("Show all available stocks"):
        st.write("These stocks are available for selection:")
        for i in stocks:
            st.text(i)
    sel = st.multiselect("Choose relevant stocks from the list(Click on 'Show stocks' above for the list):",stocks,stocks[0])
    if st.button("Submit"):
        st.write("These are the final selected stocks:")
        for i in sel:
            st.text(i)
        tickers = np.array(sel)
        yahoo_financials = YahooFinancials(np.array(sel))
        data = yahoo_financials.get_historical_price_data(start_date='2020-04-01', 
                                                    end_date='2023-03-31', 
                                                    time_interval='daily')
        prices_df = pd.DataFrame({a: {x['formatted_date']: x['close'] for x in data[a]['prices']} for a in tickers})
        st.table(prices_df.head())
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

    if st.button("Submit"):
        st.write("These are the final selected stocks:")
        for i in sel:
            st.text(i)
        tickers = np.array(sel)
        yahoo_financials = YahooFinancials(np.array(sel))
        data = yahoo_financials.get_historical_price_data(start_date='2020-04-01', 
                                                   # end_date='2023-03-31', 
                                                   # time_interval='daily')
        prices_df = pd.DataFrame({a: {x['formatted_date']: x['close'] for x in data[a]['prices']} for a in tickers})
        st.table(prices_df.head())
