import pandas as pd 
import requests 
import re
import bs4
from datetime import date
import datetime

def stock_metric (): # hàm tạo lấy dữ liệu chỉ số tài chính từ trang web 
    print('-----------------------------')
    print('Get url')
    print('-----------------------------')
    url = "https://s.cafef.vn/screener.aspx#data" 
    page = requests.get(url)
    print('-----------------------------')
    print('Set type Regex')
    print('-----------------------------')
    symbol = r'"Symbol":"(.*?)"\s*'
    centerName = r'"CenterName":"(.*?)"\s*'
    PE = r'"PE":([^"]+)'
    Beta = r'"Beta":([^"]+)'
    Price = r'"Price":([^"]+)'
    print('-----------------------------')
    print('Processing Regex in page.text')
    print('-----------------------------')
    symbol_data = re.findall(symbol,page.text)
    centerName_data = re.findall(centerName,page.text)
    PE_data = re.findall(PE,page.text)
    Beta_data = re.findall(Beta,page.text)
    Price_data = re.findall(Price,page.text)
    print('-----------------------------')
    print('Set dictionary and read dataframe')
    print('-----------------------------')
    stock = {'Ticker':symbol_data,'Stock_Ex':centerName_data,'Price':Price_data,'PE':PE_data,'Beta':Beta_data}
    df = pd.DataFrame.from_dict(stock)
    print('-----------------------------')
    print('Processing data')
    print('-----------------------------')
    # loại bỏ đi các dấu phẩy tồn tại trong các cột
    df['PE']=df['PE'].apply(lambda x: x.split(',')[0]) 
    df['Beta']=df['Beta'].apply(lambda x: x.split(',')[0])
    df['Price']=df['Price'].apply(lambda x: x.split(',')[0])
    return df

def extract_text (name_ex,day): # hàm đọc text của trang web
    name = name_ex
    url = 'https://s.cafef.vn/tracuulichsu2/1/' + name + '/' + day + '.chn' # cài đặt link trang web crawl
    page = requests.get(url)
    return page.text

def extract_data (text): # hàm lấy dữ liệu từ file text sau khi đọc 
    rel = r'" class="symbol" rel="([^"]+)"'
    price_change = r'(-?\d+\.\d+) \((-?\d+\.\d+) %\)'
    ticker = re.findall(rel,text) # ticker 
    price_change_data = re.findall(price_change,text)
    values = [float(i[1]) for i in price_change_data] 
    stock_frame = {'Ticker':ticker, '% Price':values}
    df = pd.DataFrame.from_dict(stock_frame)
    return df

def extract_data_set (values): # hàm trả về dữ liệu dựa vào ngày mình nhập vào 
    today = date.today()
    previous = today - datetime.timedelta(days=values)
    previous = previous.strftime("%d/%m/%Y")
    hose = extract_text('hose',previous)
    hnx = extract_text('hnx',previous)
    upcom = extract_text('upcom',previous)
    data_hose = extract_data(hose)
    data_hnx = extract_data(hnx)
    data_upcom = extract_data(upcom)
    data_price = pd.concat([data_hose,data_hnx,data_upcom],ignore_index=True)
    return data_price

# chú ý dữ liệu của trang web khi có những ngày trang web không hiển thị dữ liệu nên cần kiểm tra khi nhập ngày mình muốn lấy data

data_price_7d = extract_data_set(7)
data_price_1m = extract_data_set(30)
data_price_3m = extract_data_set(90)
data_price_1y = extract_data_set(365)

# đổi tên cột 

data_price_7d.rename(columns={'% Price':'% Price 7 days'},inplace=True) 
data_price_1m.rename(columns={'% Price':'% Price 1 month'},inplace=True)
data_price_3m.rename(columns={'% Price':'% Price 3 months'},inplace=True)
data_price_1y.rename(columns={'% Price':'% Price 1 year'},inplace=True)

# sau khi có dữ liệu giá, nối với bảng dữ liệu chỉ số tài chính

data_stock = stock_metric().merge(data_price_7d,on='Ticker'
                                        ,how='inner').merge(data_price_1m,on='Ticker'
                                                            ,how='inner').merge(data_price_3m,on='Ticker'
                                                                                ,how='inner').merge(data_price_1y,on='Ticker',how='inner')

# sắp xếp lại thứ tự các cột

data_stock = data_stock[['Ticker','Stock_Ex','Price','% Price 7 days','% Price 1 month'
                                     ,'% Price 3 months','% Price 1 year','PE','Beta']]

# xuất file csv với chế độ ghi đè

data_stock.to_csv('data_stock.csv',index=False,mode='w')





