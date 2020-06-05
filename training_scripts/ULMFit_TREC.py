'''
ULMFit_TREC.py
Training with TREC 
'''

# Fine-tuning a forward and backward langauge model to get to 95.4% accuracy on the IMDB movie reviews dataset. This tutorial is done with fastai v1.0.53.

# In[ ]:


get_ipython().system('pip install -U spacy')


# In[ ]:


get_ipython().system('pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz')


# In[ ]:


get_ipython().system('python -m spacy download en')


# In[1]:


from fastai.text import *
import spacy


# TOKENIZATION

# In[ ]:


lang = 'en'
# TODO: CHUNKSIZE 
chunksize = 50
n_lbls = 1

spacy.load(lang)

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

re1 = re.compile(r'  +')


# In[ ]:


dir_path = Path('/storage/TREC')

df_trn = pd.read_csv(dir_path / 'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(dir_path / 'val.csv', header=None, chunksize=chunksize)

# for i, r in enumerate(df_val):
#     print(i)


# In[ ]:


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls, lang='en'):
    if len(df.columns) == 1:
        labels = []
        texts = f'\n{BOS} {FLD} 1 ' + df[0].astype(str)
    else:
        labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
#         print(labels)
        texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
        for i in range(n_lbls, len(df.columns)): 
            texts += f' {FLD} {i-n_lbls+1} ' + df[i].astype(str)
#         print(texts)
        
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer(lang=lang).process_all(partition_by_cores(texts, n_cpus=1))
    return tok, list(labels)


def get_all(df, n_lbls, lang='en'):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls, lang=lang)
        tok += tok_
        labels += labels_
    return tok, labels


# In[ ]:


tmp_path = dir_path / 'tmp'
tmp_path.mkdir(exist_ok=True)

tok_trn, trn_labels = get_all(df_trn, n_lbls=1, lang=lang)

tok_val, val_labels = get_all(df_val, n_lbls=1, lang=lang)
print(len(tok_val))

np.save(tmp_path / 'tok_trn.npy', tok_trn)
np.save(tmp_path / 'tok_val.npy', tok_val)
np.save(tmp_path / 'lbl_trn.npy', trn_labels)
np.save(tmp_path / 'lbl_val.npy', val_labels)

trn_joined = [' '.join(o) for o in tok_trn]
open(tmp_path / 'joined.txt', 'w', encoding='utf-8').writelines(trn_joined)


# MAPPING TOKENS TO IDS

# In[ ]:


import collections


# In[ ]:


max_vocab=30000
min_freq=0


# In[ ]:


trn_tok = np.load(tmp_path / 'tok_trn.npy')
val_tok = np.load(tmp_path / 'tok_val.npy')

# print(val_tok[:10])
freq = collections.Counter(p for o in trn_tok for p in o)
freq2 = collections.Counter(p for o in val_tok for p in o)
print(len(freq), len(freq2))
dict.update(freq, freq2)
print(len(freq))
# print(freq.most_common(25))

itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})

trn_lm = np.array([[stoi[o] for o in p] for p in trn_tok])
val_lm = np.array([[stoi[o] for o in p] for p in val_tok])
print(len(trn_tok))
print(len(trn_lm))
print(len(val_tok))
print(len(val_lm))

np.save(tmp_path / 'trn_ids.npy', trn_lm)
np.save(tmp_path / 'val_ids.npy', val_lm)
pickle.dump(itos, open(tmp_path / 'itos.pkl', 'wb'))


# In[ ]:


dir_path = '/storage/TREC'
pretrain_path = '/storage/wt103'
cuda_id=0
cl=15
pretrain_id='wt103'
lm_id=''
bs=64,
dropmult=1.0
backwards=False
lr=4e-3
preload=True
bpe=False
startat=0,
use_clr=True
use_regular_schedule=False
use_discriminative=True
notrain=False
joined=False
train_file_id=''


# In[ ]:


PRE  = 'bwd_' if backwards else 'fwd_'
PRE = 'bpe_' + PRE if bpe else PRE
IDS = 'bpe' if bpe else 'ids'
train_file_id = train_file_id if train_file_id == '' else f'_{train_file_id}'
joined_id = 'lm_' if joined else ''
lm_id = lm_id if lm_id == '' else f'{lm_id}_'
lm_path=f'{PRE}{lm_id}lm'
enc_path=f'{PRE}{lm_id}lm_enc'

dir_path = Path(dir_path)
pretrain_path = Path(pretrain_path)
pre_lm_path = pretrain_path / 'models' / f'{PRE}{pretrain_id}.h5'
for p in [dir_path, pretrain_path, pre_lm_path]:
    assert p.exists(), f'Error: {p} does not exist.'

bptt=70
em_sz,nh,nl = 400,1150,3
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

if backwards:
    trn_lm_path = dir_path / 'tmp' / f'trn_{joined_id}{IDS}{train_file_id}_bwd.npy'
    val_lm_path = dir_path / 'tmp' / f'val_{joined_id}{IDS}_bwd.npy'
else:
    trn_lm_path = dir_path / 'tmp' / f'trn_{joined_id}{IDS}{train_file_id}.npy'
    val_lm_path = dir_path / 'tmp' / f'val_{joined_id}{IDS}.npy'

print(f'Loading {trn_lm_path} and {val_lm_path}')
trn_lm = np.load(trn_lm_path)
trn_lm = np.concatenate(trn_lm)
# print(len(trn_lm))
val_lm = np.load(val_lm_path)
val_lm = np.concatenate(val_lm)
# print(len(val_lm))

if bpe:
    vs=30002
else:
    itos = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))
    vs = len(itos)

trn_dl = LanguageModelPreLoader(trn_lm, bs, bptt)
val_dl = LanguageModelPreLoader(val_lm, bs, bptt)
md = TextLMDataBunch(dir_path, 1, vs, trn_dl, val_dl)

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*dropmult

learner = md.get_model(opt_fn, em_sz, nh, nl,
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])
learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learner.clip=0.3
learner.metrics = [accuracy]
wd=1e-7

lrs = np.array([lr/6,lr/3,lr,lr/2]) if use_discriminative else lr
if preload and startat == 0:
    wgts = torch.load(pre_lm_path, map_location=lambda storage, loc: storage)
    if bpe:
        learner.model.load_state_dict(wgts)
    else:
        print(f'Loading pretrained weights...')
        ew = to_np(wgts['0.encoder.weight'])
        row_m = ew.mean(0)

        itos2 = pickle.load(open(pretrain_path / 'tmp' / f'itos.pkl', 'rb'))
        stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})
        nw = np.zeros((vs, em_sz), dtype=np.float32)
        nb = np.zeros((vs,), dtype=np.float32)
        for i,w in enumerate(itos):
            r = stoi2[w]
            if r>=0:
                nw[i] = ew[r]
            else:
                nw[i] = row_m

        wgts['0.encoder.weight'] = T(nw)
        wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(nw))
        wgts['1.decoder.weight'] = T(np.copy(nw))
        learner.model.load_state_dict(wgts)
        #learner.freeze_to(-1)
        #learner.fit(lrs, 1, wds=wd, use_clr=(6,4), cycle_len=1)
elif preload:
    print('Loading LM that was already fine-tuned on the target data...')
    learner.load(lm_path)

if not notrain:
    learner.unfreeze()
    if use_regular_schedule:
        print('Using regular schedule. Setting use_clr=None, n_cycles=cl, cycle_len=None.')
        use_clr = None
        n_cycles=cl
        cl=None
    else:
        n_cycles=1
    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(learner, lm_path, enc_path, patience=5))
        print('Using early stopping...')
    learner.fit(lrs, n_cycles, wds=wd, use_clr=(32,10) if use_clr else None, cycle_len=cl,
                callbacks=callbacks)
    learner.save(lm_path)
    learner.save_encoder(enc_path)
else:
    print('No more fine-tuning used. Saving original LM...')
    learner.save(lm_path)
    learner.save_encoder(enc_path)


# In[ ]:


new_path = untar_data('/storage/TREC')


# ## From a language model...

# ### Data collection

# This was run on a Titan RTX (24 GB of RAM) so you will probably need to adjust the batch size accordinly. If you divide it by 2, don't forget to divide the learning rate by 2 as well in the following cells. You can also reduce a little bit the bptt to gain a bit of memory. 

# In[3]:


bs,bptt=256,80


# This will download and untar the file containing the IMDB dataset, returning a `Pathlib` object pointing to the directory it's in (default is ~/.fastai/data/imdb0). You can specify another folder with the `dest` argument.

# In[2]:


path = untar_data('/storage/TREC')
# print(TextList.from_csv(path, csv_name='train.csv'))


# In[ ]:


path2 = untar_data(URLs.IMDB)
print(path2.ls())


# We then gather the data we will use to fine-tune the language model using the [data block API](https://docs.fast.ai/data_block.html). For this step, we want all the texts available (even the ones that don't have lables in the unsup folder) and we won't use the IMDB validation set (we will do this for the classification part later only). Instead, we set aside a random 10% of all the texts to build our validation set.
# 
# The fastai library will automatically launch the tokenization process with the [spacy tokenizer](https://spacy.io/api/tokenizer/) and a few [default rules](https://docs.fast.ai/text.transform.html#Rules) for pre and post-processing before numericalizing the tokens, with a vocab of maximum size 60,000. Tokens are sorted by their frequency and only the 60,000 most commom are kept, the other ones being replace by an unkown token. This cell takes a few minutes to run, so we save the result.

# In[ ]:


# data_lm = (TextList.from_folder(path, extensions='.csv', include=[path/'full_train.csv'], exclude=[path/'train.csv', path/'test.csv', path/'val.csv'])
#             .split_by_rand_pct(0.1)
#            #We randomly split and keep 10% (10,000 reviews) for validation
#             .label_for_lm()           
#            #We want to do a language model so we label accordingly
#             .databunch(bs=bs, bptt=bptt))

data_lm = (TextList.from_csv(path, 'full_train.csv', header=None, cols=1)
           #Inputs: all the text files in path
            .split_by_rand_pct(0.1)
           #We randomly split and keep 10% (10,000 reviews) for validation
            .label_for_lm()           
           #We want to do a language model so we label accordingly
            .databunch(bs=bs, bptt=bptt))
data_lm.save('data_lm.pkl')


# When restarting the notebook, as long as the previous cell was executed once, you can skip it and directly load your data again with the following.

# In[4]:


data_lm = load_data(path, 'data_lm.pkl', bs=bs, bptt=bptt)


# Since we are training a language model, all the texts are concatenated together (with a random shuffle between them at each new epoch). The model is trained to guess what the next word in the sentence is.

# In[ ]:


data_lm.show_batch()


# For a backward model, the only difference is we'll have to pqss the flag `backwards=True`.

# In[5]:


data_bwd = load_data(path, 'data_lm.pkl', bs=bs, bptt=bptt, backwards=True)


# In[ ]:


data_bwd.show_batch()


# ### Fine-tuning the forward language model

# The idea behind the [ULMFit paper](https://arxiv.org/abs/1801.06146) is to use transfer learning for this classification task. Our language model isn't randomly initialized but with the weights of a model pretrained on a larger corpus, [Wikitext 103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/). The vocabulary of the two datasets are slightly different, so when loading the weights, we take care to put the embedding weights at the right place, and we rando;ly initiliaze the embeddings for words in the IMDB vocabulary that weren't in the wikitext-103 vocabulary of our pretrained model.
# 
# This is all done by the first line of code that will download the pretrained model for you at the first use. The second line is to use [Mixed Precision Training](), which enables us to use a higher batch size by training part of our model in FP16 precision, and also speeds up training by a factor 2 to 3 on modern GPUs. 

# In[6]:


learn = language_model_learner(data_lm, AWD_LSTM)
learn = learn.to_fp16(clip=0.1)


# The `Learner` object we get is frozen by default, which means we only train the embeddings at first (since some of them are random).

# In[7]:


learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7), wd=0.1)


# Then we unfreeze the model and fine-tune the whole thing.

# In[8]:


learn.unfreeze()


# In[9]:


learn.fit_one_cycle(15, 2e-3, moms=(0.8,0.7), wd=0.1)


# Once done, we jsut save the encoder of the model (everything except the last linear layer that was decoding our final hidden states to words) because this is what we will use for the classifier.

# In[10]:


learn.save_encoder('fwd_enc_15e')


# ### The same but backwards

# You can't directly train a bidirectional RNN for language modeling, but you can always enseble a forward and backward model. fastai provides a pretrained forward and backawrd model, so we can repeat the previous step to fine-tune the pretrained backward model. The command `language_model_learner` checks the `data` object you pass to automatically decide if it should use the pretrained forward or backward model.

# In[11]:


learn = language_model_learner(data_bwd, AWD_LSTM)
learn = learn.to_fp16(clip=0.1)


# Then the training is the same:

# In[12]:


learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7), wd=0.1)


# In[13]:


learn.unfreeze()


# In[14]:


learn.fit_one_cycle(15, 2e-3, moms=(0.8,0.7), wd=0.1)


# In[15]:


learn.save_encoder('bwd_enc_15e')


# ## ... to a classifier

# ### Data Collection

# The classifier is a model that is a bit heavier, so we have lower the batch size.

# In[ ]:


import pandas as pd


# In[16]:


# path = untar_data(URLs.IMDB)
bs = 128


# We use the data block API again to gather all the texts for classification. This time, we only keep the ones in the trainind and validation folderm and label then by the folder they are in. Since this step takes a bit of time, we save the result.

# In[ ]:


df_train = pd.read_csv(path/'full_train.csv', header=None)
df_train['is_valid'] = False
# df_train.head()
print(df_train.shape)


# In[ ]:


df_test = pd.read_csv(path/'test.csv', header=None)
df_test['is_valid'] = True
df_test.head()
print(df_test.shape)


# In[ ]:


df_clas = df_train.append(df_test, ignore_index=True)
print(df_clas.shape)
df_clas.head()


# In[ ]:


data_clas = (TextList.from_df(df_clas, path, cols=1, vocab=data_lm.vocab)
             .split_from_df(col='is_valid')
             .label_from_df(cols=0)
             .databunch(bs=bs))

data_clas.save('data_clas.pkl')


# As long as the previous cell was executed once, you can skip it and directly do this.

# In[17]:


data_clas = load_data(path, 'data_clas.pkl', bs=bs)


# In[ ]:


data_clas.show_batch()


# Like before, you only have to add `backwards=True` to load the data for a backward model.

# In[18]:


data_clas_bwd = load_data(path, 'data_clas.pkl', bs=bs, backwards=True)


# In[ ]:


data_clas_bwd.show_batch()


# ### Fine-tuning the forward classifier

# The classifier needs a little less dropout, so we pass `drop_mult=0.5` to multiply all the dropouts by this amount (it's easier than adjusting all the five different values manually). We don't load the pretrained model, but instead our fine-tuned encoder from the previous section.

# In[38]:


learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, pretrained=False)
learn.load_encoder('fwd_enc_15e')


# Then we train the model using gradual unfreezing (partially training the model from everything but the classification head frozen to the whole model trianing by unfreezing one layer at a time) and differential learning rate (deeper layer gets a lower learning rate).

# In[39]:


lr = 1e-1


# In[40]:


learn.fit_one_cycle(1, lr, moms=(0.8,0.7), wd=0.1)


# In[41]:


learn.freeze_to(-2)
lr /= 2
learn.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)


# In[42]:


learn.freeze_to(-3)
lr /= 2
learn.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)


# In[43]:


learn.unfreeze()
lr /= 5
learn.fit_one_cycle(10, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)


# In[26]:


learn.save('fwd_clas_15e_5e')


# ### The same but backwards

# Then we do the same thing for the backward model, the only thigns to adjust are the names of the data object and the fine-tuned encoder we load.

# In[27]:


learn_bwd = text_classifier_learner(data_clas_bwd, AWD_LSTM, drop_mult=0.5, pretrained=False)
learn_bwd.load_encoder('bwd_enc_15e')


# In[28]:


learn_bwd.fit_one_cycle(1, lr, moms=(0.8,0.7), wd=0.1)


# In[29]:


learn_bwd.freeze_to(-2)
lr /= 2
learn_bwd.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)


# In[30]:


learn_bwd.freeze_to(-3)
lr /= 2
learn_bwd.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)


# In[31]:


learn_bwd.unfreeze()
lr /= 5
learn_bwd.fit_one_cycle(5, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)


# In[32]:


learn_bwd.save('bwd_clas_15e_5e')


# ### Ensembling the two models

# For our final results, we'll take the average of the predictions of the forward and the backward models. SInce the samples are sorted by text lengths for batching, we pass the argument `ordered=True` to get the predictions in the order of the texts.

# In[33]:


pred_fwd,lbl_fwd = learn.get_preds(ordered=True)


# In[34]:


pred_bwd,lbl_bwd = learn_bwd.get_preds(ordered=True)


# In[35]:


final_pred = (pred_fwd+pred_bwd)/2


# In[36]:


accuracy(final_pred, lbl_fwd)


# And we get the 95.4% accuracy reported in the paper!
