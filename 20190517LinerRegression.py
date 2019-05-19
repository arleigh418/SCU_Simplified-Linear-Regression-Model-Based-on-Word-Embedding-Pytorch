
import pandas as pd 
import jieba
import numpy  as np
import torch
from torch import nn
from torch import optim
import torch.autograd as autograd


jieba.load_userdict('new_words.txt')

#====================training data=========================
x = pd.read_excel('GHW2_train.xls', encoding = 'utf-8')

trend = x['trend'].tolist()
trade = x['trade'].tolist()
text = x['summary'].tolist()

x_text_gg=[]
for i in text:
    seg_full = jieba.cut(i , cut_all=False)
    x = ' '.join(seg_full)
    x_text_gg.append(x)


#====================testing data=========================
g = pd.read_excel('GHW2_test.xls', encoding = 'utf-8')

text_test = g['summary'].tolist()
x_text_yy=[]
for i in text_test:
    seg_full_test = jieba.cut(i , cut_all=False)
    x = ' '.join(seg_full_test)
    x_text_yy.append(x)






def prepare_sequence(seq, to_ix, cuda=True):
    seq_list = seq.split(' ')
    
    remainderWords = list(filter(lambda a: a in to_ix, seq_list))

    var = autograd.Variable(torch.LongTensor([to_ix[w] for w in remainderWords]))
#     print('var:',var)
    return var

def build_token_to_ix(sentences):
    token_to_ix = dict()
    # print(len(sentences))
    for sent in sentences:
        sent = str(sent)
        for token in sent.split(' '):
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix



word_to_ix = build_token_to_ix([s for s in x_text_gg])



print('building over -- ix')


class LinerRegression(nn.Module):

    def __init__(self, embedding_dim, vocab_size):
        super(LinerRegression, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden1label = nn.Linear(embedding_dim , 2)
        # self.hidden1labe2 = nn.Linear(embedding_dim , 1)
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # x = embeds.view(len(sentence), 1, -1)
#         print(embeds.size())
#         print(x.size())
        y  = self.hidden1label(embeds[-1])
       
        return y



def train():
    epochs = 30
    model = LinerRegression(200,vocab_size = len(word_to_ix))
    model.cuda()
    
    
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3) 
    loss_function = nn.MSELoss()
    # test_trade_store = []
    # test_trend_store = []

    for epoch in range(epochs):
        c=0
        loss_avg_trade = 0 
        loss_avg_trend = 0
        loss_trade,loss_trend = train_every_epochs(loss_avg_trade,loss_avg_trend,c,model,x_text_gg,trade,trend,loss_function,optimizer,word_to_ix)
        # if epoch % 10 == 0 and epoch != 0:
        #         for param_group in model.parameters():
        #                 param_group['lr'] = param_group['lr'] * 0.8
        print ('Epoch [{}], Loss_trend: {:.4f},Loss_trade: {:.4f}'.format(epoch, loss_trend,loss_trade))
    b=0
    test_trade,test_trend = test(b,model,x_text_yy,word_to_ix)
    trade_pre = pd.DataFrame(test_trade)
    trade_pre.to_csv('trade_pred.csv')
    trend_pre = pd.DataFrame(test_trend)
    trend_pre.to_csv('trend_pred.csv')
    
        


def train_every_epochs(loss_avg_trade,loss_avg_trend,count,model,text,trade,trend,loss_function,optimizer,word_to_ix):
    
#     model.train()
    for i in text:

        to_ix = prepare_sequence(i,word_to_ix).cuda()
        pred_trade,pred_trend = model(to_ix)
        
        a = np.array(trade[count])
        b = np.array(trend[count])
        trade_real = torch.from_numpy(a).type(torch.FloatTensor).cuda()
        trend_real = torch.from_numpy(b).type(torch.FloatTensor).cuda()
        trade_real = trade_real.view(-1,1,1)
        trend_real = trend_real.view(-1, 1,1)
        
        model.zero_grad()
        loss_trade = loss_function(pred_trade,trade_real)
    
        loss_trend = loss_function(pred_trend,trend_real)
     
        loss_trade.backward(retain_graph = True)
        loss_trend.backward()
        optimizer.step()
        count += 1
        loss_avg_trade +=loss_trade.item()
        loss_avg_trend +=loss_trend.item()
        
    loss_fuc_trade = loss_avg_trade/count
    loss_fuc_trend = loss_avg_trend/count
    return loss_fuc_trade,loss_fuc_trend


def test(count2,model,text2,word_to_ix2):
    model.eval()
    storge= []
    storge2 = []
    for x in text2:    
        to_ix = prepare_sequence(x,word_to_ix2).cuda()
        
        count2+=1
        test_trade,test_trend = model(to_ix)
        storge.append(test_trade.item())
        storge2.append(test_trend.item())
        
    return storge,storge2
    
    
    
model = train()




