from team5.fiterHtml import FilterTag
import jieba


class DataConverter():
    def __init__(self):
        pass

    # 清洗停用词方法
    def drop_stopwords(self,contents, stopwords):
        contents_clean = []
        for line in contents:
            line_Clean = []
            for word in line:
                word = word.strip()
                if (word == '') or (word == ' ') or (word in stopwords):
                    continue
                else:
                    line_Clean.append(word)
            contents_clean.append(line_Clean)
        return contents_clean

    def CleanData(self,conentList):
        # 清除html标签
        afterfiter = []
        fiterTag = FilterTag()
        for item in conentList:
            afterfiter.append(fiterTag.filterHtmlTag(item))

        # 结巴分词
        content_S = []
        for line in afterfiter:
            current_segment = jieba.lcut(line)
            if len(current_segment) > 1 and current_segment != '\r\n':
                content_S.append(current_segment)

        # 读取停用词
        stopwords = []
        with open("newstopwords.txt", encoding="utf-8") as f:
            for line in f.readlines():
                a = line.rstrip("\n")
                stopwords.append(a)

        # 清洗停用词
        convert = DataConverter()
        content_clean =convert.drop_stopwords(content_S, stopwords)
        return content_clean

