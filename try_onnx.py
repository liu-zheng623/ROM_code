import torch
import pandas as pd
import numpy as np
import os
import torch.onnx
import onnxruntime
import matplotlib
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
os.chdir('/Users/apple/git/ROM_code/data_test')
np.random.seed(0)
torch.manual_seed(0)
class lr(nn.Module):
    def __init__(self,input_c,hidden_c,output_c):
        super(lr,self).__init__()
        self.layer1=nn.Linear(input_c,hidden_c)
        self.layer2=nn.Linear(hidden_c,output_c)
        self.act=nn.ReLU()

    def forward(self,X):
        output1=self.act(self.layer1(X))
        output2=self.layer2(output1)
        return output2

# In[1]读取数据

data_ftp=pd.read_csv('FTP_new.csv').drop([0])
data_nedt=pd.read_csv('NEDT_new.csv').drop([0])
fea_data=['Tambient [degC]', 'vehicle speed [m/s]', 'compressor speed', 'compressor command']
lab_data=['evapo temperature', 'evapo relative humidity','compressor power']
input_1=data_ftp[fea_data]
input_2=data_nedt[fea_data]
output_1=data_ftp[lab_data]
output_2=data_nedt[lab_data]
# In[]
fea=pd.merge(input_1,input_2,how='outer')
lab=pd.merge(output_1,output_2,how='outer')
#fea=fea.loc[:,['A','B','C','D']]
#lab=lab.loc[:,['E','F','G']]
# In[2]DEA
print(fea.isnull().sum())
print(lab.isnull().sum())
# In[4]划分训练集测试集
rate=0.99
trainX,testX=fea.loc[:int(rate*len(fea)),:],fea.loc[int(rate*len(fea)):,:]
trainY,testY=lab.loc[:int(rate*len(lab)),:],lab.loc[int(rate*len(fea)):,:]
# In[5]转化为torch
# trainx,testx,trainy,testy=torch.from_numpy(trainX.values),torch.from_numpy(testX.values),torch.from_numpy(trainY.values),torch.from_numpy(testY.values)
trainx=torch.from_numpy(trainX.values)
testx=torch.from_numpy(testX.values)
trainy,testy=torch.from_numpy(trainY.values),torch.from_numpy(testY.values)

# In[6]训练与测试
model=lr(4,32,3)
loss_f=torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
epoch=3000
train_loss=[]
test_loss=[]
for _ in range(epoch):
    output=model(trainx.to(torch.float32))
    loss=loss_f(output,trainy.to(torch.float32))
    model.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss.append(loss.detach().numpy())
    with torch.no_grad():
        model.eval()
        pred=model(testx.to(torch.float32))
        test_loss.append(loss_f(pred,testy.to(torch.float32)).detach().numpy())
    if _%100==0:
        print(f'epoch:{_}')
        print(f'train_mse:{train_loss[-1]}')
        print(f'test_mse:{test_loss[-1]}')
batch_size=1
input_c=4
dummy_input = torch.randn(batch_size, input_c).to(torch.float32)
# 导出模型为ONNX格式

plt.figure()

plt.plot(train_loss,label='train_loss')
# plt.plot(test_loss,label='valid_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
onnx_path = "/Users/apple/git/ROM_code/data_test/model_0401.onnx"
torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'],verbose=True)
test_fea_HWFET=pd.read_csv('HWFET_new.csv').loc[:,fea_data].drop([0])
test_target_HWFET=pd.read_csv('HWFET_new.csv').loc[:,lab_data].drop([0])
test_fea_SC03=pd.read_csv('SC03_new.csv').loc[:,fea_data].drop([0])
test_target_SC03=pd.read_csv('SC03_new.csv').loc[:,lab_data].drop([0])
test_fea_US06=pd.read_csv('US06_new.csv').loc[:,fea_data].drop([0])
test_target_US06=pd.read_csv('US06_new.csv').loc[:,lab_data].drop([0])
fea_12=pd.merge(test_fea_HWFET,test_fea_SC03,how='outer')
fea_123=pd.merge(fea_12,test_fea_US06,how='outer')
lab_12=pd.merge(test_target_HWFET,test_target_SC03,how='outer')
lab_123=pd.merge(lab_12,test_target_US06,how='outer')
test_fea_123_S=torch.from_numpy(fea_123.values)
test_pred_123_S=model(test_fea_123_S.to(torch.float32))

# 检验2：加载 ONNX 模型
onnx_model = onnxruntime.InferenceSession("/Users/apple/git/ROM_code/data_test/model_0401.onnx")

# 存储预测结果
predictions = []

# 逐个样本进行推断
for i in range(len(test_fea_123_S)):
    # 准备输入数据并调整形状
    input_data = test_fea_123_S[i].to(torch.float32).numpy().reshape(1, -1)

    # 进行推断
    output = onnx_model.run(None, {"input": input_data})
    # 提取结果并添加到列表中
    predictions.append(output[0])

# 将列表转换为 NumPy 数组
predictions = np.array(predictions)

predictions = np.squeeze(predictions, axis=1)

# 打印结果的形状
print(predictions.shape)
unstandard_1=predictions
plt.figure()

plt.plot(np.arange(len(unstandard_1)),lab_123.loc[:,'compressor power'],label='Target_G')
plt.plot(np.arange(len(unstandard_1)),unstandard_1[:,2],label='Predict_G')
plt.legend()
plt.show()
plt.figure()

plt.plot(np.arange(len(unstandard_1)),lab_123.loc[:,'evapo relative humidity'],label='Target_F')
plt.plot(np.arange(len(unstandard_1)),unstandard_1[:,1],label='Predict_F')
plt.legend()
plt.show(block=False)
plt.figure()

plt.plot(np.arange(len(unstandard_1)),lab_123.loc[:,'evapo temperature'],label='Target_E')
plt.plot(np.arange(len(unstandard_1)),unstandard_1[:,0],label='Predict_E')
plt.legend()
plt.show()
def mape(y,p):
    return np.mean(np.abs(y-p)/np.abs(y))*100
mape1=mape(lab_123.loc[:,'evapo temperature'],unstandard_1[:,0])
mape2=mape(lab_123.loc[:,'evapo relative humidity'],unstandard_1[:,1])
mape3=mape(lab_123.loc[:,'compressor power'],unstandard_1[:,2])
metr=(mape1+mape2+mape3)/3
# In[]
from sklearn.metrics import mean_absolute_error
mae1=mean_absolute_error(lab_123.loc[:,'evapo temperature'],unstandard_1[:,0])
mae2=mean_absolute_error(lab_123.loc[:,'evapo relative humidity'],unstandard_1[:,1])
mae3=mean_absolute_error(lab_123.loc[:,'compressor power'],unstandard_1[:,2])