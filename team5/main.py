import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['E:\\Python\\BidClassify_Team5\\','E:/Python/BidClassify_Team5/'])

import pandas as pd
from bidtype_classify_team5 import BidTypeClassifierTeam15
from bidtype_classify_team5 import BidTypeCreatModel
import time

#读取数据
df = pd.read_csv("bidtype_train5.csv", encoding="utf-8")


#分隔数据
from sklearn.model_selection import  train_test_split
dfx_train , dfx_test ,dfy_train , dfy_test = train_test_split(df,df["bidType"],test_size=0.14, random_state=40)


start = time.clock()
#训练模型
bidCreateModel = BidTypeCreatModel()
bidCreateModel.CreateModel(df)
end = time.clock()
print(end-start)


start = time.clock()
#测试数据
bidClassify = BidTypeClassifierTeam15()
df = bidClassify.predict(dfx_test)
end = time.clock()
print(end-start)

df[["htmlContent", "predict_y", "bidType", "IsPredict"]].to_csv('df.csv', encoding="utf-8")
df[["predict_y", "bidType", "IsPredict", "bidName"]].to_excel("df.xls", sheet_name='Random Data',
                                                                      encoding='utf-8')
from sklearn.metrics import accuracy_score
score = accuracy_score(df["bidType"], df["predict_y"])
print(score)