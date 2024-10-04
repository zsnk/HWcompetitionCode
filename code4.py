#%% 导入库
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.cuda.is_available())
device = torch.device("cuda:0")

plt.rcParams['font.family'] = ['STSong']  # Use SimHei font to display Chinese characters
plt.rcParams['font.size'] = 10

#%% 导入数据
# 读取Excel文件
excel_file = 'trainset.xlsx'

# 读取三个表格
df1 = pd.read_excel(excel_file, sheet_name='材料1')
df2 = pd.read_excel(excel_file, sheet_name='材料2')
df3 = pd.read_excel(excel_file, sheet_name='材料3')
df4 = pd.read_excel(excel_file, sheet_name='材料4')

# 添加材料标签
df1['材料类型']='材料1'
df2['材料类型']='材料2'
df3['材料类型']='材料3'
df4['材料类型']='材料4'

# 合并表格
df = pd.concat([df1, df2, df3, df4])

#%% 将df中的非数值型变量转换围为数值型
df['励磁波形'] = df['励磁波形'].map({'正弦波':1,'三角波':2,'梯形波':3})
df['材料类型'] = df['材料类型'].map({'材料1':1,'材料2':2,'材料3':3,'材料4':4})

#%% 把'磁芯损耗，w/m3'作为y变量，其余列作为x变量，训练神经网络模型
y = df['磁芯损耗，w/m3']
x = df.drop(['磁芯损耗，w/m3'], axis=1)
x.columns = x.columns.astype(str)
xx = x.iloc[:,-1025:-1]
x = x.drop(xx.columns,axis=1)
x['磁通密度峰值'] = xx.max(axis=1)
y = y.apply(np.log)

#%% 80%作为训练集，20%作为测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#  normalize the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
y = np.array(y)
y_train = np.array(y_train)

x_train = torch.Tensor(x_train).to(device)
y_train = torch.Tensor(y_train).to(device)
#  define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 512)  # input layer
        self.fc2 = nn.Linear(512, 10)  # hidden layer 
        self.fc3 = nn.Linear(10, 1)   # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # activation function for hidden layer
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
#%% 模型训练
#  train the model
model = Net().to(device) 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(torch.tensor(x_train).float())
    loss = criterion(outputs, torch.tensor(y_train).unsqueeze(1).float())
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch {}: Loss = {}'.format(epoch, loss.item()))

#%%  evaluate the model
x_test = torch.Tensor(x_test).to(device)
y_pred = model(torch.tensor(x_test).float()).cpu().detach().numpy()

y_pred = np.exp(y_pred)
y_test = np.exp(y_test)
plt.scatter(y_test,y_pred)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.show()

#%% r2_score
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
# mse
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))

#%% 预测集结果展示
X = pd.read_excel('附件三（测试集）.xlsx', sheet_name='测试集')

X['励磁波形'] = X['励磁波形'].map({'正弦波':1,'三角波':2,'梯形波':3})
X['材料类型'] = X['磁芯材料'].map({'材料1':1,'材料2':2,'材料3':3,'材料4':4})
X = X.drop(['磁芯材料','序号'], axis=1)

X.columns = X.columns.astype(str)
XX = X.iloc[:,-1024:-1]
X = X.drop(xx.columns,axis=1)
X['磁通密度峰值'] = XX.max(axis=1)

X = sc.transform(X)

# 预测
X = np.array(X)
X = torch.Tensor(X).to(device)
y_pred_test = model(torch.tensor(X).float()).cpu().detach().numpy()
y_pred_test = np.exp(y_pred_test)

#%% 保存预测结果
ydf = pd.DataFrame(y_pred_test)
ydf.to_excel('y_pred_test.xlsx', index=False)

#%% es方程
P_es = df['磁芯损耗，w/m3']
f_es = df['频率，Hz']
logP_es = np.log(P_es)
logf_es = np.log(f_es)
B_m = df.iloc[:,-1025:-1]
B_m = B_m.max(axis=1)
logB_m = np.log(B_m)

X_es = pd.concat([logf_es, logB_m], axis=1)
X_es = sm.add_constant(X_es)
model_es = sm.OLS(logP_es,X_es).fit()
print(model_es.summary())
y_es_pred = model_es.predict(X_es)
y_es_pred = np.exp(y_es_pred)

#%% 修正方程
temp = df['温度，oC']
log_temp = np.log(temp)
X_xzes = pd.concat([X_es, log_temp], axis=1)
model_xzes = sm.OLS(logP_es,X_xzes).fit()
print(model_xzes.summary())
y_xzes_pred = model_xzes.predict(X_xzes)
y_xzes_pred = np.exp(y_xzes_pred)

# %% 各模型预测效果可视化
x_standard = sc.transform(x)
x_standard = torch.Tensor(x_standard).to(device)
y_nn_pred = model(x_standard.float()).cpu().detach().numpy()
y_nn_pred = np.exp(y_nn_pred)
y_nn_pred = pd.DataFrame(y_nn_pred)
y_true = df['磁芯损耗，w/m3']


plt.plot(y_es_pred-y_true,'.',label='ES方程')
plt.plot(y_xzes_pred-y_true,'--',label='修正方程')
plt.plot(y_nn_pred-y_true,'-',label='神经网络')
plt.legend()
plt.show()
# %%
