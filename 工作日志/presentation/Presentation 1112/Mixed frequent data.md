---
center: false
height: 1170
---
## 混频数据处理的传统方案
- 在大多数应用中，在混合样本频率存在的情况下，常见的解决方案是对数据进行**预过滤**，
使变量以相同的频率采样。

- 在此过程中，模型中包含的许多潜在有用的信息可能被破坏和错误规范。

---
**传统的混频数据解决方案：**
1. 总体法
- 在低频周期中求高频数据的股票平均值和现金流的和（代表高频数据）
- 假设之前高频期的信息反映在最新的现金流中，就在低频周期中，选最新的高频数据里现金流的值代表整个低频期。
--
2. 插值法
- 对低频的变量进行插值。

- 一种常见的方法是:首先插入缺失数据，然后使用新的增强级数(the new augmented series)估计模型参数，可能要考虑到测量由分解引起的误差。从模型的状态空间表示开始，这两个步骤可以方便地在卡尔曼滤波器设置(Kalman filter set-up)中联合运行。
--
 3.等权重加权 
- 将样本区间内的高频数据点进行简单的等权重加权，忽略每天股票回报率对预测贡献信息量的不同

--
无论是总体法还是插值法，都会对预测产生一定程度的误差

---
## MIDAS模型
**MIDAS模型**常用于宏观经济学的预测应用,是将混频数据整合到统一的回归模型后，用于长期预测的有效工具。

---

- Ghysels, Rubia和Valkanov(2009)比较了三种产生波动率多周期提前预测的不同方法:
- 迭代、直接和MIDAS的比较。使用美国股票市场投资组合的回报数据和规模、账面市值比和行业投资组合的横截面，就平均预测精度而言，使用MSFE进行了样本外分析。其中迭代预测适用于较短的范围，而MIDAS预测则适用于较长的范围。

--
- Clements和Galvao(2009)利用MIDAS方法以简约的方式组合多个领先指标，评估了领先指标对长达一年的产出增长的预测能力。结果证实了MIDAS是改善预测的有用工具。
- 此外，他们还表明，使用实时年份数据提高了预测性能，当目标是预测最终数据而不是首次发布的数据时，指标的预测能力更强，尽管首次发布的数据通常可以更准确地预测。

---
- MIDAS模型可用**频率不匹配的数据**（如每日股票，季度财报）进行预测，主要用于GDP和CPI等宏观经济学的预测。
- 随着日内数据的广泛使用，在**微观金融变量研究**上也发挥出越来越大的作用。

--
- Ghysels, Santa-Clara and Valkanov 2005, Ghysels, et al. 2007)使用日度收益率的平方对月度方差进行了预测,发现股票市场的风险和收益存在显著的正相关
- (Ghysels, Santa-Clara and Valkanov 2006)使用了不同频率的数据对日度波动率进行了预测,发现5分钟频率的数据并不能提高波动率的预测精度,目度已实现收益率对未来波动率的预测效果最佳; (Alper, Fendoglu and Saltoglu 2008)以4个发达国家和10个新兴市场国家为样本对周度的波动率进行预测,发现在四个新兴市场国家中,用日度数据的MIDAS模型预测效果显著地好于周度的GARCH(1,1)模型

---

## MIDAS模型代码及解读：

代码来自GitHub:[ Python 中的混合数据采样 （MIDAS） 建模 ](https://github.com/Yoseph-Zuskin/midaspy)

---
### 1. 安装库midaspy
`pip install --user git+git://github.com/Yoseph-Zuskin/midaspy.git@master`

--
安装时可能会遇到的问题：
1. 确保以下设置已关闭： **关闭设置中网络--代理--手动设置代理**，会影响库的安装
2. **conda命令配置**：找到`.condarc`文件，修改里面的内容如下：

```
show_channel_urls: true

channels:
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

ssl_verify: false

```
--
3. **安装git命令**：需要执行`conda install git`
4. `fatal: unable to connect to github`问题，执行以下命令：
   `git config --global url."https://github.com".insteadOf git://github.com`
   如果还是报错，再执行
   `git config --global --unset http.proxy`
   `git config --global --unset https.proxy`


---
### 2. 加载midaspy与Pandas库：
```
import numpy as np
import pandas as pd
from itertools import product
from midaspy.iolib import *
from midaspy.model import MIDASRegressor
```

示例中使用以下函数从 zip 文件中提取 CSV 文件并从加拿大统计局加载数据：
```
from zipfile import ZipFile
from requests import get
from io import BytesIO
```


```
def extract_zip_from_url(link):
    """
    从URL中提取ZIP
    此函数从在线找到的ZIP文件中提取字节数据
    
    参数
    link (str): ZIP文件的URL链接
    返回
    输出(dict):压缩文件的字节数据字典
    """
 ```

```   
    u=get(link)
    f=BytesIO() 
    f.write(u.content)
    input_zip=ZipFile(f)
    output={i:input_zip.read(i) for i in input_zip.namelist()}
    return output
```

---
```
def load_stats_can_data(pid,low_memory=False):
    """
    加拿大统计数据下载
    自定义函数直接从在线zip文件加载数据到pandas的dataframe
    
    参数
    pid (str):the Statistics Canada online data asset的 产品id
    Low_memory (bool):记住设置读取的为csv文件 memory setting for the pandas.read_csv function
    
    返回
    df (pandas.DataFrame):来自 the data asset的所有变量的dataframe

    """
```

```
    # 首先使用extract_zip_from_url从加拿大统计局网站提取zip文件
    # first extract the zip file from the Statistics Canada website using extract_zip_from_url
    online_zip=extract_zip_from_url('https://www150.statcan.gc.ca/n1/tbl/csv/{}-eng.zip'.format(pid))
    # 然后下载数据资产的结果CSV文件，并将日期设置为datetime64[ns]数据类型
    # then load the resulting csv file for the data asset and set the date to datetime64[ns] data type
    df=pd.read_csv(BytesIO(online_zip['{}.csv'.format(pid)]),parse_dates=[0],low_memory=low_memory)
```

---
```
    # 筛选数据资产以排除与领域相关的数据
    df=df[df.iloc[:,1]=='Canada'] # filter the data asset to exclude data related to the territories
    # 定义变量类型列表，如第4列所示
    variables=list(set(df.iloc[:,3])) # define list of variable types as they appear in the 4th column
    # 将每一系列变量储存在这一字典中
    variable_series={} # store each variable series in this temporary dictionary instance
 ```
 ---
 ```
	# 遍历data asset中所有变量
    for v in variables: # iterable over all the variables in this data asset
        series=df[df.iloc[:,3]==v].set_index(df.columns[0]).VALUE # 筛选这一串变量的dataframe filter dataframe for the variable
        series.rename(v,inplace=True) # 对这一串变量重命名 rename the series to what Statistics Canada refers to it as
        variable_series[v]=series # store the resulting series in the temporary dictionary instance
    df=pd.DataFrame() # create dataframe to store all combined series
    for v in variables: # iterate over each variable and merge it as this works when pd.concat may fail
        df=pd.merge(right=df,left=variable_series[v],right_index=True,left_index=True,how='outer')
    return df        
```
---


### 第一部分：加载行数据
- 示例中从雅虎财经（Yahoo Finance)获取加拿大消费者物价指数的数据
```
'''
第一部分：加载行数据
'''
# 加载Canadian Consumer Price Index数据
# Load the Canadian Consumer Price Index data frm Statistics Canada
cpi_df=load_stats_can_data('10100106')
col='Consumer Price Index (CPI) inversely weighted by volatility and is adjusted to exclude the effect of changes in indirect taxes (CPIW) (year-over-year percent change)'
cpi_df=cpi_df[~cpi_df.index.duplicated(keep='first')][col]
# 从雅虎财经下载S&P/TSX综合指数历史每日数据,从1970年至今
# load the S&P/TSX Composite index historical daily data from Yahoo Finance
tsx=pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/%5EGSPTSE?'+
                'period1=-315619200&period2={}&interval=1d&events=history'.format(
                (pd.Timestamp.today()-pd.Timestamp("1970-01-01"))//pd.Timedelta('1s')),
               parse_dates=['Date'],index_col=0)
```

---
### 第二部分：导出和处理数据

1. 填充空数据
TSX数据只存在于非假日工作日，但整个月的数据仍具有可比性,假设这些天的经济状况没有变化，那么我们可以在连续的每日日期范围内填充缺失的数据来解决这个问题.
**因为tsx中存有完整数据，continuous_datarange中删除了tsx中为0的数据，当天的数据就转换为了空数据，再用ffill()就可以将tsx中每一行空数据用上一行，也就是上一次的数据的填充。**

首先创建一个空的数据框架，包含可用TSX数据的开始日期和结束日期之间的所有日期
```    
'''
第二部分：导出和处理数据
'''
# the TSX data exists only for non-holiday weekdays but to keep all months' data comparable
# let's assume that there is no change in economic conditions on those days, so we can forward fill missing data over a continuous daily date range to address this issue by first creating an empty dataframe of all days in between the start and end dates of the available TSX data
continuous_daterange=pd.date_range(tsx.index[0],tsx.index[-1]).to_frame().drop(0,axis=1)
# then concatenate this continuous date range with the TSX data and do a forward fill

tsx=pd.concat([continuous_daterange,tsx],axis=1).ffill()
del continuous_daterange # delete the temporary continuous_daterange index
```

---
### 第三部分:拟合MIDAS的回归模型

- 示例中筛选了至2019年的数据，进行模型的拟合
`Adj Close`是调整后的收盘价,`pct_change`表示当前元素与先前两个元素百分比
**`endog`表示较低频率的目标数据，在这里是每月一次的CPI ,`exog`表示更高频的目标数据，在这里是每日的股票收盘价**
```
# Select training data up to end of 2019
x1 = tsx['Adj Close'].rename('TSX_7d_pct_change')['1983-11-01':'2019-12-01'].pct_change(7).dropna()
x2 = tsx['Adj Close'].rename('TSX_14d_pct_change')['1983-11-01':'2019-12-01'].pct_change(14).dropna()
x3 = tsx['Adj Close'].rename('TSX_21d_pct_change')['1983-11-01':'2019-12-01'].pct_change(21).dropna()
x4 = tsx['Adj Close'].rename('TSX_28d_pct_change')['1983-11-01':'2019-12-01'].pct_change(28).dropna()
y = cpi_df.rename('CPI_YoY')['1984-02-01':'2019-12-01']
model = MIDASRegressor(endog=y,exog=pd.concat([x1,x2,x3,x4],axis=1).loc['1983-11-30':],
                       xlag=30,ylag=1,poly='beta')
fit = model.fit() 
```
---
### 第四部分：评估模型的拟合
``
- 绘制预测与实际的拟合曲线图形：
```
pd.concat([fit.orig_endog,fit.predict()], axis=1)[-24:].plot()
```
- 拟合分数：
```
fit.score()
```
- 一些指标：
```
fit.significance()
```
示例中结果图，拟合程度看起来相当高：
![[工作日志/picture/Pasted image 20221110235009.png]]

---
#### 第五部分：样本外预测

- 选取2020年到今天的数据，进行预测
```
# Select test data (2020-present)
x1_test = tsx['Adj Close'].rename('TSX_7d_pct_change')['2019-12-01':].pct_change(7).dropna()
x2_test = tsx['Adj Close'].rename('TSX_14d_pct_change')['2019-12-01':].pct_change(14).dropna()
x3_test = tsx['Adj Close'].rename('TSX_21d_pct_change')['2019-12-01':].pct_change(21).dropna()
x4_test = tsx['Adj Close'].rename('TSX_28d_pct_change')['2019-12-01':].pct_change(28).dropna()
y_test = cpi_df.rename('CPI_YoY')['2020-01-01':]
```
----
- 预测及图像绘制
```
# Plot predictions
preds = fit.predict(endog=y_test,exog=pd.concat([x1_test,x2_test,x3_test,x4_test],axis=1))
preds.index = y_test.index[1:]
pd.concat([y_test[1:], preds], axis=1).plot()# 绘制比较图
```

示例中给出的样本外预测比较图：
![[工作日志/picture/Pasted image 20221110235042.png]]