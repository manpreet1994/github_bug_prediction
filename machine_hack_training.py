#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
from fastai.text import *
from fastai.metrics import *
from transformers import RobertaTokenizer
import time
# from fastai.vision import *


# In[58]:


train_df = pd.read_json("embold_train.json").reset_index(drop=True)
train_df.head()


# In[59]:


set(train['label'])


# In[60]:


# Creating a config object to store task specific information
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
        
config = Config(
    testing=False,
    seed = 2019,
    roberta_model_name='distilroberta-base', #'roberta-base', # can also be exchnaged with roberta-large 
    max_lr=1e-5,
    epochs=1,
    use_fp16=True,
    bs=8, 
    max_seq_len=128, 
    num_labels = 3,
    hidden_dropout_prob=.05,
    hidden_size=768, # 1024 for roberta-large
    start_tok = "<s>",
    end_tok = "</s>",
)


# In[61]:


# train = pd.read_csv("data/train.csv",usecols=['body','label']).dropna()
train = pd.read_json("embold_train.json").reset_index(drop=True).dropna()


# In[75]:


feat_cols = 'title', 'body'
label_cols = 'label'


# In[76]:


class FastAiRobertaTokenizer(BaseTokenizer):
    """Wrapper around RobertaTokenizer to be compatible with fastai"""
    def __init__(self, tokenizer: RobertaTokenizer, max_seq_len: int=128, **kwargs): 
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len 
    def __call__(self, *args, **kwargs): 
        return self 
    def tokenizer(self, t:str) -> List[str]: 
        """Adds Roberta bos and eos tokens and limits the maximum sequence length""" 
        return [config.start_tok] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + [config.end_tok]


# In[77]:


# create fastai tokenizer for roberta
roberta_tok = RobertaTokenizer.from_pretrained("roberta-base")
# roberta_tok = RobertaTokenizer.from_pretrained("distilroberta-base")

fastai_tokenizer = Tokenizer(tok_func=FastAiRobertaTokenizer(roberta_tok, max_seq_len=config.max_seq_len), 
                             pre_rules=[], post_rules=[])


# In[78]:


# create fastai vocabulary for roberta
path = Path()
roberta_tok.save_vocabulary(path)

with open('vocab.json', 'r') as f:
    roberta_vocab_dict = json.load(f)
    
fastai_roberta_vocab = Vocab(list(roberta_vocab_dict.keys()))


# In[79]:


# Setting up pre-processors
class RobertaTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
         super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)

class RobertaNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_roberta_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):
    """
    Constructing preprocessors for Roberta
    We remove sos and eos tokens since we add that ourselves in the tokenizer.
    We also use a custom vocabulary to match the numericalization with the original Roberta model.
    """
    return [RobertaTokenizeProcessor(tokenizer=tokenizer), RobertaNumericalizeProcessor(vocab=vocab)]


# In[80]:


# Creating a Roberta specific DataBunch class
class RobertaDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training Roberta"
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds, path:PathOrStr='.', bs:int=64, val_bs:int=None, pad_idx=1,
               pad_first=True, device:torch.device=None, no_check:bool=False, backwards:bool=False, 
               dl_tfms:Optional[Collection[Callable]]=None, **dl_kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **dl_kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs))
        return cls(*dataloaders, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)


# In[81]:


class RobertaTextList(TextList):
    _bunch = RobertaDataBunch
    _label_cls = TextList


# In[82]:


# loading the tokenizer and vocab processors
processor = get_roberta_processor(tokenizer=fastai_tokenizer, vocab=fastai_roberta_vocab)

# creating our databunch 
data = RobertaTextList.from_df(train, ".", cols=feat_cols, processor=processor)     .split_by_rand_pct(0.2,seed=config.seed)     .label_from_df(cols=label_cols,label_cls=CategoryList)     .databunch(bs=config.bs, pad_first=False, pad_idx=0)
#     .add_test(RobertaTextList.from_df(test, ".", cols=feat_cols, processor=processor)) \
    


# In[102]:


data


# In[83]:


### MODEL


# In[84]:


import torch
import torch.nn as nn
from transformers import RobertaModel

# defining our model architecture 
class CustomRobertaModel(nn.Module):
    def __init__(self,num_labels=5):
        super(CustomRobertaModel,self).__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(config.roberta_model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels) # defining final output layer
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _ , pooled_output = self.roberta(input_ids, token_type_ids, attention_mask) # 
        logits = self.classifier(pooled_output)        
        return logits
    
    def predict(self, inputtext):
        return torch.sigmoid(self.forward(input_ids = torch.tensor([roberta_tok.encode(inputtext)], dtype = torch.long ,device='cpu')))


# In[85]:


roberta_model = CustomRobertaModel(num_labels=config.num_labels)


# In[86]:


learn = Learner(data, roberta_model, metrics=[accuracy])
learn.model.roberta.train() 
learn.freeze_to(-1)


# In[120]:


learn.fit_one_cycle(config.epochs, max_lr=config.max_lr,moms=(0.8,0.9))


# In[88]:


learn.export('machinehack2.pkl')


# ## Predictions

# In[109]:


test = pd.read_json("embold_test.json").reset_index(drop=True)


# In[110]:


test.iloc[0]


# In[111]:


learn.predict(test.iloc[0])[1].tolist()


# In[112]:


[x for x in test]


# In[121]:


predictions =  []
for x in range(len(test)):
    predictions.append(learn.predict(test.iloc[x])[1].tolist())


# In[122]:


predictions


# In[123]:


#create a submission dataframe
submission_df = pd.DataFrame(predictions, columns=['label'])

#write a .csv file for submission
submission_df.to_csv('submission06.csv', index=False)


# In[ ]:




