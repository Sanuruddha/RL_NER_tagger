import codecs
from gensim.models import FastText

with codecs.open('train.txt', 'rb', encoding='utf-16', errors='ignore') as infile:
    fin = infile.read()

lines = [[]]

word = ""

for letter in fin:
    if letter == "\t":
        if not word == "":
            lines[len(lines)-1].append(word)
            word = ""
        continue
    elif letter == ".":
        lines.append([])
        continue
    elif not (u'\u0d80' <= letter <= u'\u0dff'):
        continue
    word += letter

f = codecs.open("out.txt", "w+", encoding ='utf-8')
for i in lines:
    f.write("%s\n" % (' '.join(i)))

print(lines[0])

model_ted = FastText(lines, size=30, window=5, min_count=5, workers=4, sg=1)
model_ted.save("word2vec.model")

print(model_ted.wv.most_similar(u'\u0dc4\u0dba\u0dd2\u0da7\u0dd2\u0dba'))