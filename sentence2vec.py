import codecs
sentences = [[]]
labels = [[]]
with codecs.open('train.txt', 'rb', encoding='utf-16', errors='ignore') as infile:
    lines = infile.readlines()
    EOF = False
    count = 0
    for line in lines:
        try:
            word, label = line.split()
            sentences[len(sentences)-1].append(word)
            if label == 'O':
                label = 0
            else:
                label = 1
            labels[len(labels)-1].append(label)
            count += 1
            if count == 10:
                count = 0
                sentences.append([])
                labels.append([])
        except:
            continue

print(sentences[0])
print(labels[0])