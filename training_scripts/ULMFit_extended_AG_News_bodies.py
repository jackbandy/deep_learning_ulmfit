'''
ULMFit_extended_AG_News_bodies.py
Uses AG News text bodies, trains for 20 epochs 
'''

# Fine-tuning a forward and backward language model to get to 94.99% accuracy on the AG News dataset. This tutorial is done with fastai v1.0.53.

# In[1]:


from fastai.text import *


# ## From a language model...

# ### Data collection

# This was run on a Titan RTX (24 GB of RAM) so you will probably need to adjust the batch size accordinly. If you divide it by 2, don't forget to divide the learning rate by 2 as well in the following cells. You can also reduce a little bit the bptt to gain a bit of memory. 

# In[2]:


bs,bptt=256,80


# This will download and untar the file containing the AG News dataset, returning a `Pathlib` object pointing to the directory it's in (default is /notebooks/storage/ag_news_csv). You can specify another folder with the `dest` argument.

# In[3]:


path = untar_data(URLs.AG_NEWS)


# We then gather the data we will use to fine-tune the language model using the [data block API](https://docs.fast.ai/data_block.html). For this step, we want all the texts available (even the ones that don't have lables in the unsup folder) and we won't use the AG News validation set (we will do this for the classification part later only). Instead, we set aside a random 10% of all the texts to build our validation set.
# 
# The fastai library will automatically launch the tokenization process with the [spacy tokenizer](https://spacy.io/api/tokenizer/) and a few [default rules](https://docs.fast.ai/text.transform.html#Rules) for pre and post-processing before numericalizing the tokens, with a vocab of maximum size 60,000. Tokens are sorted by their frequency and only the 60,000 most commom are kept, the other ones being replace by an unkown token. This cell takes a few minutes to run, so we save the result.

# In[4]:


data_lm = (TextList.from_csv(path, 'train.csv', cols=2)
           #Inputs: all the text files in path
            .split_by_rand_pct(0.1)
           #We randomly split and keep 10% (10,000 reviews) for validation
            .label_for_lm()           
           #We want to do a language model so we label accordingly
            .databunch(bs=bs, bptt=bptt))
data_lm.save('data_lm_bodies.pkl')


# When restarting the notebook, as long as the previous cell was executed once, you can skip it and directly load your data again with the following.

# In[4]:


data_lm = load_data(path, 'data_lm_bodies.pkl', bs=bs, bptt=bptt)


# Since we are training a language model, all the texts are concatenated together (with a random shuffle between them at each new epoch). The model is trained to guess what the next word in the sentence is.

# In[6]:


data_lm.show_batch()


# For a backward model, the only difference is we'll have to pqss the flag `backwards=True`.

# In[5]:


data_bwd = load_data(path, 'data_lm_bodies.pkl', bs=bs, bptt=bptt, backwards=True)


# In[6]:


data_bwd.show_batch()


# ### Fine-tuning the forward language model

# The idea behind the [ULMFit paper](https://arxiv.org/abs/1801.06146) is to use transfer learning for this classification task. Our language model isn't randomly initialized but with the weights of a model pretrained on a larger corpus, [Wikitext 103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/). The vocabulary of the two datasets are slightly different, so when loading the weights, we take care to put the embedding weights at the right place, and we rando;ly initiliaze the embeddings for words in the IMDB vocabulary that weren't in the wikitext-103 vocabulary of our pretrained model.
# 
# This is all done by the first line of code that will download the pretrained model for you at the first use. The second line is to use [Mixed Precision Training](), which enables us to use a higher batch size by training part of our model in FP16 precision, and also speeds up training by a factor 2 to 3 on modern GPUs. 

# In[9]:


learn = language_model_learner(data_lm, AWD_LSTM)
learn = learn.to_fp16(clip=0.1)


# The `Learner` object we get is frozen by default, which means we only train the embeddings at first (since some of them are random).

# In[10]:


learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7), wd=0.1)


# Then we unfreeze the model and fine-tune the whole thing.

# In[11]:


learn.unfreeze()


# In[12]:


learn.fit_one_cycle(20, 2e-3, moms=(0.8,0.7), wd=0.1)


# Once done, we jsut save the encoder of the model (everything except the last linear layer that was decoding our final hidden states to words) because this is what we will use for the classifier.

# In[13]:


learn.save_encoder('fwd_enc_20e_ag_news_bodies')


# ### The same but backwards

# You can't directly train a bidirectional RNN for language modeling, but you can always enseble a forward and backward model. fastai provides a pretrained forward and backawrd model, so we can repeat the previous step to fine-tune the pretrained backward model. The command `language_model_learner` checks the `data` object you pass to automatically decide if it should use the pretrained forward or backward model.

# In[7]:


learn = language_model_learner(data_bwd, AWD_LSTM)
learn = learn.to_fp16(clip=0.1)


# Then the training is the same:

# In[8]:


learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7), wd=0.1)


# In[9]:


learn.unfreeze()


# In[10]:


learn.fit_one_cycle(20, 2e-3, moms=(0.8,0.7), wd=0.1)


# In[11]:


learn.save_encoder('bwd_enc_20e_ag_news_bodies')


# ## ... to a classifier

# ### Data Collection

# The classifier is a model that is a bit heavier, so we have lower the batch size.

# In[12]:


path = untar_data(URLs.AG_NEWS)
bs = 128


# We use the data block API again to gather all the texts for classification. This time, we only keep the ones in the trainind and validation folderm and label then by the folder they are in. Since this step takes a bit of time, we save the result.

# In[13]:


df_train = pd.read_csv(Path(path)/'train.csv', header=None)
df_train['is_valid'] = False
df_train.head()


# In[14]:


df_test = pd.read_csv(Path(path)/'test.csv', header=None)
df_test['is_valid'] = True
df_test.head()


# In[15]:


len(df_train), len(df_test)


# In[16]:


df_clas = pd.concat([df_train, df_test])


# In[17]:


df_clas.head()


# In[18]:


df_clas.tail()


# In[19]:


len(df_clas)


# In[20]:


data_clas = (TextList.from_df(df_clas, path, cols=2, vocab=data_lm.vocab)
             .split_from_df(col='is_valid')
             .label_from_df(cols=0)
             .databunch(bs=bs))

data_clas.save('data_clas_bodies.pkl')


# As long as the previous cell was executed once, you can skip it and directly do this.

# In[21]:


data_clas = load_data(path, 'data_clas_bodies.pkl', bs=bs)


# In[22]:


data_clas.show_batch()


# Like before, you only have to add `backwards=True` to load the data for a backward model.

# In[23]:


data_clas_bwd = load_data(path, 'data_clas_bodies.pkl', bs=bs, backwards=True)


# In[24]:


data_clas_bwd.show_batch()


# ### Fine-tuning the forward classifier

# The classifier needs a little less dropout, so we pass `drop_mult=0.5` to multiply all the dropouts by this amount (it's easier than adjusting all the five different values manually). We don't load the pretrained model, but instead our fine-tuned encoder from the previous section.

# In[25]:


learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, pretrained=False)
learn.load_encoder('fwd_enc_20e_ag_news_bodies')


# Then we train the model using gradual unfreezing (partially training the model from everything but the classification head frozen to the whole model trianing by unfreezing one layer at a time) and differential learning rate (deeper layer gets a lower learning rate).

# In[26]:


lr = 1e-1


# In[27]:


learn.fit_one_cycle(1, lr, moms=(0.8,0.7), wd=0.1)


# In[28]:


learn.freeze_to(-2)
lr /= 2
learn.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)


# In[29]:


learn.freeze_to(-3)
lr /= 2
learn.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)


# In[30]:


learn.unfreeze()
lr /= 5
learn.fit_one_cycle(4, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)


# In[31]:


learn.save('fwd_clas_20e_ag_news_bodies')


# ### The same but backwards

# Then we do the same thing for the backward model, the only thigns to adjust are the names of the data object and the fine-tuned encoder we load.

# In[32]:


learn_bwd = text_classifier_learner(data_clas_bwd, AWD_LSTM, drop_mult=0.5, pretrained=False)
learn_bwd.load_encoder('bwd_enc_20e_ag_news_bodies')


# In[33]:


learn_bwd.fit_one_cycle(1, lr, moms=(0.8,0.7), wd=0.1)


# In[34]:


learn_bwd.freeze_to(-2)
lr /= 2
learn_bwd.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)


# In[35]:


learn_bwd.freeze_to(-3)
lr /= 2
learn_bwd.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)


# In[36]:


learn_bwd.unfreeze()
lr /= 5
learn_bwd.fit_one_cycle(4, slice(lr/(2.6**4),lr), moms=(0.8,0.7), wd=0.1)


# In[37]:


learn_bwd.save('bwd_clas_20e_ag_news_bodies')


# ### Ensembling the two models

# For our final results, we'll take the average of the predictions of the forward and the backward models. SInce the samples are sorted by text lengths for batching, we pass the argument `ordered=True` to get the predictions in the order of the texts.

# In[38]:


pred_fwd,lbl_fwd = learn.get_preds(ordered=True)


# In[39]:


pred_bwd,lbl_bwd = learn_bwd.get_preds(ordered=True)


# In[40]:


final_pred = (pred_fwd+pred_bwd)/2


# In[41]:


accuracy(final_pred, lbl_fwd)


# And we get the 94.99% accuracy reported in the paper!
