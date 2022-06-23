import pandas as pd

def describe_data(path, file):
    data = pd.read_csv(path + file, header = 0)

    class1 = data[data['Sentiment'] == 1]
    class0 = data[data['Sentiment'] == 0]

    textl0 = class0.sum(axis = 1).mean(axis = 0)
    textl1 = class1.sum(axis = 1).mean(axis = 0)

    words0 = class0.sum(axis = 0).astype(bool).sum(axis = 0)
    words1 = class1.sum(axis = 0).astype(bool).sum(axis = 0)

    print('# instances:', len(data.index))
    print('avg. text lenght class 1:', textl1, '| avg. text lenght class 0:', textl0)
    print('words class 1:', words0, '| words class 0:', words1)
    print('# words:', len(data.columns[1:]))
    print('Vocabulary:', list(data.columns[1:]))

path = 'C:/Users/hp/Desktop/codigo/'
file = 'data_wo.csv'
describe_data(path, file)
