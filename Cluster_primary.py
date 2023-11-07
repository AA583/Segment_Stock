import streamlit as st
import pandas as pd

st.set_page_config(page_title="Segment Stock", # đặt tên trang web
                    page_icon=':bar_chart:', # chọn icon
                    layout="wide") # lựa chọn layout toàn màn hình

st.header('Segment Stock') # đặt tên tiêu đề

df=pd.read_csv('https://raw.githubusercontent.com/AA583/Segment_Stock/main/data_stock_primary.csv') # đọc dữ liệu từ file csv

pd.options.display.float_format = '{:.3f}'.format # định dạng hiển thị số  

#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------

st.header('I. Dataset') # tiêu đề

st.write('Raw Data') # show data

st.dataframe(df) # xem dữ liệu 

col1,col2 = st.columns(2)

with col1 :
    st.write('Kích thước dữ liệu ban đầu')

    st.dataframe(df.shape)

    st.write('Dữ liệu có 1555 dòng và 9 thuộc tính')

df=df[df['Price']>0] # loại bỏ cổ phiếu có giá < 0

with col2:
    st.write('Kích thước dữ liệu sau khi loại bỏ giá trị ngoại lai')

    st.dataframe(df.shape)  

    st.write('Dữ liệu có 1554 dòng và 9 thuộc tính')

#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------

st.header('II. Visalizations')

import seaborn as sns
import matplotlib.pyplot as plt

st.subheader('Sự phân bổ dữ liệu ban đầu')

col1, col2 = st.columns(2) #tạo 2 cột trong streamlit

with col1:
    fig = plt.figure(figsize=(12,6))

    sns.boxplot(df['% Price 7 days'], color='skyblue')

    # Display the plot as an image in Streamlit
    st.pyplot(fig)

with col2:
    fig = plt.figure(figsize=(12,6))
    
    sns.boxplot(df['% Price 1 month'], color='skyblue')
   
    # Display the plot as an image in Streamlit
    st.pyplot(fig)

col1, col2 = st.columns(2) 

with col1:
    fig = plt.figure(figsize=(12,6))
 
    sns.boxplot(df['% Price 3 months'], color='skyblue')

    # Display the plot as an image in Streamlit
    st.pyplot(fig)

with col2:
    fig = plt.figure(figsize=(12,6))
  
    sns.boxplot(df['% Price 1 year'], color='skyblue')
  
    # Display the plot as an image in Streamlit
    st.pyplot(fig)

col1, col2 = st.columns(2) 

with col1:
    fig = plt.figure(figsize=(12,6))

    sns.boxplot(df['PE'], color='skyblue')
    
    # Display the plot as an image in Streamlit
    st.pyplot(fig)

with col2:
    fig = plt.figure(figsize=(12,6))

    sns.boxplot(df['Beta'], color='skyblue')
   
    # Display the plot as an image in Streamlit
    st.pyplot(fig)

st.subheader('Sự phân bổ dữ liệu sau khi chuẩn hóa bằng IQR')

def normalized_IQR(dataframe,column_name): # tạo hàm chuẩn hóa dữ liệu 
    q1=dataframe[column_name].quantile(0.25)
    q3=dataframe[column_name].quantile(0.75)
    
    IQR= q3-q1

    outlier_up = q3 + 3*IQR 
    outlier_down = q1 - 3*IQR
    dataframe = dataframe[dataframe[column_name] > outlier_down]
    result = dataframe[dataframe[column_name] < outlier_up]
    return result


df = normalized_IQR(df,'PE') # chuẩn hóa PE đầu tiên vì có dữ liệu ngoại lai lớn 
df = normalized_IQR(df,'% Price 1 year') # chuẩn hóa % Price 1 year vì có khoảng giá trị ngoại lai rất lớn
df = normalized_IQR(df,'Beta')
df = normalized_IQR(df,'% Price 3 months')
df = normalized_IQR(df,'% Price 1 month')
df = normalized_IQR(df,'% Price 7 days')

st.dataframe(df.describe()) # mô tả dữ liệu sau khi chuẩn hóa

# trực quan hóa lại dữ liệu sau khi chuẩn hóa

col1, col2 = st.columns(2) 

with col1:
    fig = plt.figure(figsize=(12,6))
 
    sns.boxplot(df['% Price 7 days'], color='skyblue')

    # Display the plot as an image in Streamlit
    st.pyplot(fig)

with col2:
    fig = plt.figure(figsize=(12,6))

    sns.boxplot(df['% Price 1 month'], color='skyblue')

    # Display the plot as an image in Streamlit
    st.pyplot(fig)

col1, col2 = st.columns(2) 

with col1:
    fig = plt.figure(figsize=(12,6))

    sns.boxplot(df['% Price 3 months'], color='skyblue')

    # Display the plot as an image in Streamlit
    st.pyplot(fig)

with col2:
    fig = plt.figure(figsize=(12,6))

    sns.boxplot(df['% Price 1 year'], color='skyblue')

    # Display the plot as an image in Streamlit
    st.pyplot(fig)

col1, col2 = st.columns(2) 

with col1:
    fig = plt.figure(figsize=(12,6))

    sns.boxplot(df['PE'], color='skyblue')

    # Display the plot as an image in Streamlit
    st.pyplot(fig)

with col2:
    fig = plt.figure(figsize=(12,6))

    sns.boxplot(df['Beta'], color='skyblue')

    # Display the plot as an image in Streamlit
    st.pyplot(fig)

#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------

st.header('III. Models')

st.write('Sử dụng Pycaret automation machine learning với mô hình K-MEANS để phân khúc cổ phiếu')

st.write('Lựa chọn dữ liệu phân cụm')

data = df[['% Price 7 days','% Price 1 month','% Price 3 months','% Price 1 year','PE','Beta']]

st.dataframe(data.describe()) # mô tả dữ liệu 

data = data.astype('Float64') #thay đổi kiểu dữ liệu float -> double phù hợp với yêu cầu predictions k-means

st.subheader('Explain Metrics') 

st.write('- **% Price 7 days**: phần trăm giá thay đổi trong 7 ngày')
st.write('- **% Price 1 month**: phần trăm giá thay đổi trong 1 tháng')
st.write('- **% Price 3 months**: phần trăm giá thay đổi trong 3 tháng')
st.write('- **% Price 1 year**: phần trăm giá thay đổi trong 1 năm')
st.write('- **PE**: hệ số giá trên lợi nhuận một cổ phiếu, là tỷ số tài chính dùng để đánh giá mối liên hệ giữa thị giá hiện tại của một cổ phiếu và tỉ suất lợi nhuận trên cổ phần')
st.write('- **Beta**: là hệ số đo lường mức độ rủi ro của một cổ phiếu cụ thể hay một danh mục đầu tư với mức độ rủi ro chung của thị trường chứng khoán')

from pycaret.clustering import * # import thư viện pycaret sử dụng mô hình phân cụm

s = setup(data, session_id=123) 
#normalize = True - lựa chọn chuẩn hóa dữ liệu, session_id = 123 kiểm soát tính ngẫu nhiên của thử nghiệm và có thể tái tạo lại kết quả thử nghiệm mỗi lần chạy

st.write('Khởi tạo mô hình k-means và chạy mô hình')

kmeans = create_model('kmeans') # tạo mô hình phân cụm k-means

result = assign_model(kmeans) # chạy thử mô hình

predictions = predict_model(kmeans, data = data) # chạy trên dữ liệu chính

saveed_model = save_model(kmeans, 'kmeans_pipeline') # lưu lại mô hình phân cụm kmeans_pipeline

loaded_model = load_model('kmeans_pipeline') 

st.write('Số lượng dữ liệu mỗi cluster')

cluster_0 = predictions[predictions['Cluster'] == 'Cluster 0'] # trích xuất và lưu dữ liệu cluster 1

cluster_1 = predictions[predictions['Cluster'] == 'Cluster 1'] # trích xuất và lưu dữ liệu cluster 2

cluster_2 = predictions[predictions['Cluster'] == 'Cluster 2'] # trích xuất và lưu dữ liệu cluster 3

cluster_3 = predictions[predictions['Cluster'] == 'Cluster 3'] # trích xuất và lưu dữ liệu cluster 4

# mô tả dữ liệu của từng cluster

col1, col2 = st.columns(2)

with col1:
    st.write('Cluster 1')
    st.dataframe(cluster_0.describe()) 

with col2:
    st.write('Cluster 2')
    st.dataframe(cluster_1.describe())

col1, col2 = st.columns(2)
with col1:
    st.write('Cluster 3')
    st.dataframe(cluster_2.describe())

with col2:
    st.write('Cluster 4')
    st.dataframe(cluster_3.describe())

#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------

st.write('- Coi sự tăng giảm giá trong thời gian ngắn (7 ngày - 1 tháng) và thời gian dài (3 tháng - 1 năm)')
st.write('\n')
st.write('\n')
st.write('- Thông thường trung bình **PE** khoảng **20 - 25** và dưới khoảng đó sẽ được coi là tỉ lệ tốt, trong khi trên mức đó sẽ được coi là tỉ lệ **PE** kém')
st.write('- **PE** < 0 cho thấy công ty đang lỗ nhưng không có nghĩa là phá sản. Tùy thuộc vào nhiều trường hợp khác nhau do công ty thay đổi chính sách kế toán hay chính sách khấu hao,...')
st.write('- **PE** cao -> cổ phiếu đắt tiền và giá có thể giảm trong tương lai') 
st.write('- **PE** thấp -> cổ phiếu rẻ và giá có thể tăng trong tương lai')
st.write('\n')
st.write('\n')
st.write('- Cổ phiếu có nhiều biến động hơn so với thị trường khi **beta** > 1 và có biến động thấp khi **beta** < 1')
st.write('- **Beta** < 0 biểu thị mối quan hệ nghịch đảo với thị trường, nhưng thường khó xảy ra')

#PHÂN TÍCH TỪNG CLUSTER CÙNG VỚI TRỰC QUAN HÓA 

st.subheader('Cluster 1') 

col1,col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(12,6))

    sns.kdeplot(cluster_0['% Price 7 days'], fill=True, color='blue', label='% Price 7 days')

    sns.kdeplot(cluster_0['% Price 1 month'], fill=True, color='green', label='% Price 1 month')

    plt.xlabel('Price Change (%)')

    plt.ylabel('Density')

    plt.title('Distribution of Price Changes in short time')

    plt.legend()

    st.pyplot(fig)

    st.write('- Trong 7 ngày số cổ phiếu giảm giá chiếm tỉ trọng lớn')
    st.write('- Trong 1 tháng giá cổ phiếu có tiến triển tăng khoảng từ 0 - 0.2%')
    st.write(':arrow_right: Nhìn chung trong khoảng **thời gian ngắn** các cổ phiếu có xu hướng tăng giảm rất nhỏ')

with col2:
    fig, ax = plt.subplots(figsize=(12,6))

    sns.kdeplot(cluster_0['% Price 3 months'], fill=True, color='blue', label='% Price 3 months')

    sns.kdeplot(cluster_0['% Price 1 year'], fill=True, color='green', label='% Price 1 year')

    plt.xlabel('Price Change (%)')

    plt.ylabel('Density')

    plt.title('Distribution of Price Changes in long time')

    plt.legend()

    st.pyplot(fig)

    st.write('- Trong 3 tháng sự thay đổi giá không đáng kể khi chủ yếu tập trung trong khoảng -3,5% tới gần 5%')
    st.write('- Tỉ trọng số cổ phiếu tụt giá từ 3 đến 7% tăng rất cao trong 1 năm')
    st.write(':arrow_right: Trong 1 năm, thời gian đầu biến động không thực sự lớn; nhưng sau 1 năm giá cổ phiếu có xu hướng giảm')


col1,col2 = st.columns(2)

with col1:
    fig = plt.figure(figsize=(12,6))

    sns.histplot(cluster_0['PE'], fill=True, color='#02A388')

    plt.xlabel('Values')

    plt.ylabel('Amount')

    plt.title('Distribution of PE')

    st.pyplot(fig)

    st.write('- Lợi nhuận trên cổ phiếu rất tốt khi trên 60% số cổ phiếu có có chỉ số 0 < PE < 25')

with col2:
    fig = plt.figure(figsize=(12,6))

    sns.histplot(cluster_0['Beta'], fill=True, color='#FE423F')

    plt.xlabel('Values')

    plt.ylabel('Amount')

    plt.title('Distribution of Beta')

    st.pyplot(fig)

    st.write('- Mức biến động so với thị trường là khá tốt khi chỉ số beta chủ yếu nằm trong khoảng 0 đến 1')

st.write(':point_right: Cluster 1 có mức giá biến động tăng 0 - 0.2% trong 1 tháng và giảm 3 - 7% trong 1 năm cùng với chỉ số lợi nhuận trên cổ phiếu 0 < PE < 25 (>60%); chỉ số biến động so với thị trường 0 < Beta < 1 (>90%)')
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
st.subheader('Cluster 2')

col1,col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(12,6))

    sns.kdeplot(cluster_1['% Price 7 days'], fill=True, color='blue', label='% Price 7 days')

    sns.kdeplot(cluster_1['% Price 1 month'], fill=True, color='green', label='% Price 1 month')

    plt.xlabel('Price Change (%)')

    plt.ylabel('Density')

    plt.title('Distribution of Price Changes in short time')

    plt.legend()

    st.pyplot(fig)

    st.write('- Nhìn chung số cổ phiếu mất giá trong 7 ngày chiếm tỉ trọng chính')
    st.write('- Trong 1 tháng, đa phần cổ phiếu đều tăng giá khoảng 0.2 - 0.6%')
    st.write(':arrow_right: Trong thời gian ngắn, nhóm cổ phiếu có biến động có thể chấp nhận được')


with col2:
    fig, ax = plt.subplots(figsize=(12,6))

    sns.kdeplot(cluster_1['% Price 3 months'], fill=True, color='blue', label='% Price 3 months')

    sns.kdeplot(cluster_1['% Price 1 year'], fill=True, color='green', label='% Price 1 year')

    plt.xlabel('Price Change (%)')

    plt.ylabel('Density')

    plt.title('Distribution of Price Changes in long time')

    plt.legend()

    st.pyplot(fig)

    st.write('- Cổ phiếu vẫn tiếp tục tăng giá trong 3 tháng')
    st.write('- Trong 1 năm giá cổ phiếu giảm mạnh từ 4 đến 7%')
    st.write(':arrow_right: Trong thời gian dài, cổ phiếu có xu hướng giảm mạnh')


col1,col2 = st.columns(2)

with col1:
    fig = plt.figure(figsize=(12,6))

    sns.histplot(cluster_1['PE'], fill=True, color='#02A388')

    plt.xlabel('Values')

    plt.ylabel('Amount')

    plt.title('Distribution of PE')

    st.pyplot(fig)

    st.write('- Chỉ số lợi nhuận trên cổ phiếu rất cao chứng tỏ giá cổ phiếu của nhóm cổ phiếu này rất cao')

with col2:
    fig = plt.figure(figsize=(12,6))

    sns.histplot(cluster_1['Beta'], fill=True, color='#FE423F')

    plt.xlabel('Values')

    plt.ylabel('Amount')

    plt.title('Distribution of Beta')

    st.pyplot(fig)

    st.write('- Không nhiều cổ phiếu có mức độ ổn định so với thị trường')

st.write(':point_right: Cluster 2 có mức biến động giá tăng 0.2 - 0.6% trong 1 tháng và giảm 4 - 7% trong 1 năm cùng với chỉ số lợi nhuận trên cổ phiếu > 30 và chỉ số biến động so với thị trường < 0 (>50%)')
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------- 
 
st.subheader('Cluster 3')

col1,col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(12,6))

    sns.kdeplot(cluster_2['% Price 7 days'], fill=True, color='blue', label='% Price 7 days')

    sns.kdeplot(cluster_2['% Price 1 month'], fill=True, color='green', label='% Price 1 month')

    plt.xlabel('Price Change (%)')

    plt.ylabel('Density')

    plt.title('Distribution of Price Changes in short time')

    plt.legend()

    st.pyplot(fig)

    st.write('- Đa phần giá cổ phiếu có xu hướng giảm trong 7 ngày')
    st.write('- Trong 1 tháng giá cổ phiếu tăng 1 đến 2%')
    st.write(':arrow_right: Biến động trong thời gian đầu là không đáng kể')

with col2:
    fig, ax = plt.subplots(figsize=(12,6))

    sns.kdeplot(cluster_2['% Price 3 months'], fill=True, color='blue', label='% Price 3 months')

    sns.kdeplot(cluster_2['% Price 1 year'], fill=True, color='green', label='% Price 1 year')

    plt.xlabel('Price Change (%)')

    plt.ylabel('Density')

    plt.title('Distribution of Price Changes in long time')

    plt.legend()

    st.pyplot(fig)

    st.write('- Trong 3 tháng giá trị cổ phiếu có xu hướng tăng nhưng không đáng kể')
    st.write('- Trong 1 năm giá trị cổ phiếu giảm chủ yếu 2.5 đến 5%')
    st.write(':arrow_right: Mức giá biến động sau thời gian dài như vậy khá tích cực')

col1,col2 = st.columns(2)

with col1:
    fig = plt.figure(figsize=(12,6))

    sns.histplot(cluster_2['PE'], fill=True, color='#02A388')

    plt.xlabel('Values')

    plt.ylabel('Amount')

    plt.title('Distribution of PE')

    st.pyplot(fig)

    st.write('- Chỉ số lợi nhuận trên cổ phiếu âm chứng tỏ công ty đang thua lỗ')

with col2:
    fig = plt.figure(figsize=(12,6))

    sns.histplot(cluster_2['Beta'], fill=True, color='#FE423F')

    plt.xlabel('Values')

    plt.ylabel('Amount')

    plt.title('Distribution of Beta')

    st.pyplot(fig)

    st.write('- Mức độ ổn định của các cổ phiếu so với thị trường đang rất thấp khi chỉ tồn tại một vài cổ phiếu có chỉ số beta > 0')

st.write(':point_right: Cluster 3 có mức biến động giá tăng 1 - 2% trong 1 tháng và giảm 2.5 - 5% trong 1 năm cùng với lợi nhuận trên cổ phiếu < 0 và chỉ số biến động so với thị trường > 0 (xấp xỉ 90%)')
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------

st.subheader('Cluster 4')

col1,col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(12,6))

    sns.kdeplot(cluster_3['% Price 7 days'], fill=True, color='blue', label='% Price 7 days')

    sns.kdeplot(cluster_3['% Price 1 month'], fill=True, color='green', label='% Price 1 month')

    plt.xlabel('Price Change (%)')

    plt.ylabel('Density')

    plt.title('Distribution of Price Changes in short time')

    plt.legend()

    st.pyplot(fig)

    st.write('- Trong 7 ngày số lượng cổ phiếu cũng như giá cổ phiếu giảm đáng kể')
    st.write('- Giá cổ phiếu cùng với số lượng bắt đầu tăng trong 1 tháng')
    st.write(':arrow_right: Nhìn chung trong thời gian ngắn cổ phiếu có sự biến động tăng khá tốt')

with col2:
    fig, ax = plt.subplots(figsize=(12,6))

    sns.kdeplot(cluster_3['% Price 3 months'], fill=True, color='blue', label='% Price 3 months')

    sns.kdeplot(cluster_3['% Price 1 year'], fill=True, color='green', label='% Price 1 year')

    plt.xlabel('Price Change (%)')

    plt.ylabel('Density')

    plt.title('Distribution of Price Changes in long time')

    plt.legend()

    st.pyplot(fig)

    st.write('- Giá trị cổ phiếu trong 3 tháng vẫn khá ổn định, không có sự biến động lớn')
    st.write('- Trong 1 năm, giá của các cố phiểu giảm trong khoảng 4 đến 7%')
    st.write(':arrow_right: Sau thời gian dài mức giá biến động giảm không lớn nhưng số lượng cổ phiếu tụt giá khá lớn')


col1,col2 = st.columns(2)

with col1:
    fig = plt.figure(figsize=(12,6))

    sns.histplot(cluster_3['PE'], fill=True, color='#02A388')

    plt.xlabel('Values')

    plt.ylabel('Amount')

    plt.title('Distribution of PE')

    st.pyplot(fig)

    st.write('- Lợi nhuận trên cổ phiếu rất tốt khi hơn 90% cổ phiểu có chỉ số dưới và trong khoảng 20 đến 25. Chứng tỏ có mang lại lợi nhuận cho nhà đầu tư')

with col2:
    fig = plt.figure(figsize=(12,6))

    sns.histplot(cluster_3['Beta'], fill=True, color='#FE423F')

    plt.xlabel('Values')

    plt.ylabel('Amount')

    plt.title('Distribution of Beta')

    st.pyplot(fig)

    st.write('- Mức độ biến động ổn định so với thị trường khi số lượng cổ phiếu có giá trị lớn hơn 0 và nhỏ hơn 1 rất cao')

st.write(':point_right: Cluster 4 có mức biến động giá tăng trong 1 tháng và giảm 4 - 6% trong 1 năm cùng với chỉ số lợi nhuận trên cổ phiểu 0 < PE < 30 và chỉ số biến động 0 < Beta < 1')

st.write('\n')
st.write('\n')

st.write(":point_right: **Kết luận:** Nhóm cổ phiếu được chia thành 4 nhóm")
st.write(":heavy_plus_sign: Nhóm cổ phiếu có mức biến động giá tăng khoảng < 1% trong 1 tháng và giảm 4 - 7% trong 1 năm cùng với chỉ số lợi nhuận trên cổ phiếu 0 < PE < 25 và chỉ số biến động so với thị trường 0 < Beta < 1")
st.write(":heavy_plus_sign: Nhóm cổ phiếu có mức biến động giá tăng khoảng < 1% trong 1 tháng và giảm 4 - 7% trong 1 năm cùng với chỉ số lợi nhuận trên cổ phiếu PE > 30 và chỉ số biến động so với thị trường Beta < 0")
st.write(":heavy_plus_sign: Nhóm cổ phiếu có mức biến động giá tăng khoảng > 1% trong 1 tháng và giảm 2 - 7% trong 1 năm cùng với chỉ số lợi nhuận trên cổ phiếu PE < 0 và chỉ số biến động so với thị trường Beta > 0")
st.write(":heavy_plus_sign: Nhóm cổ phiếu có mức biến động giá tăng khoảng > 1% trong 1 tháng và giảm 5 - 7% trong 1 năm cùng với chỉ số lợi nhuận trên cổ phiểu 0 < PE < 30 và chỉ số biến động so với thị trường 0 < Beta < 1")