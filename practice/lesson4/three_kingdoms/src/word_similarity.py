from utils import segment
from utils import files_processing
from gensim.models import word2vec
import multiprocessing

# 源文件所在目录
file_dir = '.\\source'
segment_out_dir = '.\\segment'

#files_list = files_processing.get_files_list(file_dir,postfix='*.txt')
#segment.batch_processing_files(files_list, segment_out_dir, batchSize=1000, stopwords=['\n','',' ','\n\n'])

# 如果目录中有多个文件，可以使用PathLineSentences
sentences = word2vec.PathLineSentences(segment_out_dir)

# 设置模型参数，进行训练
model = word2vec.Word2Vec(sentences, size=100, window=3, min_count=1)
print(model.wv.similarity('曹操', '诸葛亮'))
print(model.wv.similarity('诸葛亮', '周瑜'))

print('和曹操相关：' + str(model.wv.most_similar(positive=['曹操'])))
print('曹操+刘备-张飞:' + str(model.wv.most_similar(positive=['曹操','刘备'], negative=['张飞'])))
# 设置模型参数，进行训练

model2 = word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=multiprocessing.cpu_count())
# 保存模型
#model2.save('./models/word2Vec.model')
print(model2.wv.similarity('曹操', '诸葛亮'))
print(model2.wv.similarity('诸葛亮', '周瑜'))
print('和曹操相关：' + str(model2.wv.most_similar(positive=['曹操'])))
print('曹操+刘备-张飞:' + str(model2.wv.most_similar(positive=['曹操','刘备'], negative=['张飞'])))
