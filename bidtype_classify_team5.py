from sklearn.externals import joblib
from PreCleanData import DataConverter

class BidTypeClassifierTeam15:

    def __init__(self):
        self.classifier = joblib.load('model.m')  # 加载分类模型
        self.vectorizer = joblib.load('vectorizer.m')  # 加载分词特征向量话模型
        return

    def predict(self, df):

        df = df.reset_index()
        conentList = df["htmlContent"].values.tolist()
        # 格式化数据
        converter = DataConverter();
        df["content"] = converter.CleanData(conentList);
        df["Name"] = converter.CleanData(df["bidName"].values.tolist())

        # 将分词用空格隔开
        row_count = df.shape[0]
        join_words = []
        for index in range(row_count):
            join_words.append(' '.join(df.loc[index, 'content']))
        df['content'] = join_words

        x_test = self.vectorizer.transform(df["content"])
        y_test = df["bidType"]

        result = []
        isPredicet = []
        df = df.reset_index()
        y_test = y_test.reset_index()
        num = df.shape[0]

        keword = ["变更","更正","更改","撤销","补遗","答疑","澄清","中标","候选","流标","废标","终止","中止","预告","预公告"]
        for index in range(num):
            bidtype = ""
            KW = []
            projName = df.loc[index, "Name"]
            for name in projName:
                if(name in keword):
                    KW.append(name)
            ispredict = False
            if ("变更" in KW or "更正" in KW or "更改" in KW or "撤销" in KW or "补遗" in KW or "答疑" in KW or "澄清" in KW ):
                bidtype = 3
            elif ("中标" in KW or "候选" in KW or "流标" in KW or "废标" in KW or "终止" in KW or "中止" in KW):
                bidtype = 2
            elif ("预告" in KW  or "预公告" in KW):
                bidtype = 1
            else:
                bidtype = self.classifier.predict(x_test[index])[0]
                ispredict = True
            isPredicet.append(ispredict)
            result.append(bidtype)

        df["predict_y"] = result;
        df["IsPredict"] = isPredicet;
        '''
        IsSame = []
        for index in range(num):
            same = False
            if (df.loc[index, "predict"] == df.loc[index, "bidType"]):
                same = True
            IsSame.append(same)
        df["IsSame"] = IsSame
        '''

        return df




class BidTypeCreatModel:

    def __init__(self):
        pass

    def CreateModel(self,df):

        df = df.reset_index()
        conentList = df["htmlContent"].values.tolist()
        # 格式化数据
        converter = DataConverter();
        df["content"] = converter.CleanData(conentList)

        # 将分词用空格隔开
        row_count = df.shape[0]
        join_words = []
        for index in range(row_count):
            join_words.append(' '.join(df.loc[index, 'content']))
        df['content'] = join_words

        #转化为向量
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(ngram_range=(1, 4))
        vectorizer.fit(df["content"])
        joblib.dump(vectorizer,"vectorizer.m")
        x_train = vectorizer.transform(df["content"])
        y_train = df["bidType"]

        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression()
        classifier.fit(x_train, y_train)
        joblib.dump(classifier,"model.m")
        return
