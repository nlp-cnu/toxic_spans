# Sourcing for this code is from the following:
# https://www.youtube.com/watch?v=MqQ7rqRllIc
# https://www.youtube.com/watch?v=oreIJQZ40H0&t=1594s
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics, preprocessing, model_selection
import pandas as pd
import joblib

# Dataset object that will yeild data to the neural network
class BERTDataset:
    # Constructor - Stores necessary fields for training/inference
    def __init__(self, texts, tags, max_len=128, tokenizer='bert-base-uncased'):
        self.texts = texts
        self.tags = tags
        self.tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer, 
                                                                    do_lower_case=True)
        self.max_len = max_len
    # method to grab the length of the texts field
    def __len__(self):
        return len(self.texts)
    
    # method to yield items to the neural network
    # item is essentially a uid for an item in the dataset
    def __getitem__(self, item):
        text = self.texts[item]
        tags = self.tags[item]
        
        ids = []
        target_tag = []
        # Encoding data from the dataset. If I messed up big time, it would be here
        for i, s in enumerate(text):
            inputs = self.tokenizer.encode(s, add_special_tokens=False)     
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)
        # Adding CLS tokens
        ids = ids[:self.max_len - 2]
        target = target_tag[:self.max_len - 2]
        
        ids = [101] + ids + [102]
        target_tag = [0] + target_tag + [0]
        # Creating attention mask and binary mask
        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
        # Padding the data to fit the maximum sequence length requirement
        padding_len = self.max_len - len(ids)
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)
        # returns data in the form of a dictionary
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target_tag': torch.tensor(target_tag, dtype=torch.long)
        }

# EntityModel is a child of the Pytorch neural network (nn) architecture
class EntityModel(nn.Module):
    # A number of things can be altered here
    def __init__(self, num_train_steps, num_tag):
        super(EntityModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        # Determines dropout for the nn, acts as a normalization
        self.bert_drop = nn.Dropout(0.3)
        self.num_tag = num_tag
        # creates a linear layer for the model, should be experimented with
        self.out = nn.Linear(768, self.num_tag)
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = 'batch'
        
    # Model optimizer, learning rate (lr) acts like a 'unit of time' for integration
    def fetch_optimizer(self):
        opt = AdamW(self.parameters(), lr=1e-4)
        return opt
    
    # Model scheduler
    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch
    
    # Loss function calculated through the network. "How well did the network do?" Used for reweighting the nodes
    def loss(self, output, target, mask, num_labels):
        lfn = nn.CrossEntropyLoss()
        active_loss = mask.view(-1) == 1
        active_logits = output.view(-1, num_labels)
        active_labels = torch.where(
            active_loss,
            target.view(-1),
            torch.tensor(lfn.ignore_index).type_as(target)
        )
        loss = lfn(active_logits, active_labels)
        return loss
    
    # Here is where I think I can add in F1 accuracy score while training. Untested so far
    '''def monitor_metrics(self, outputs, targets):
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        return {
            'accuracy': metrics.accuracy_score(targets, outputs)}
    '''
    
    # How the model passes data through itself
    def forward(self, ids, mask, token_type_ids, target_tag):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo_tag = self.bert_drop(o1)
        tag = self.out(bo_tag)
        loss = self.loss(tag, target_tag, mask, self.num_tag)
        # met = self.monitor_metrics(x, targets)
        return tag, loss
    
# Different training method, not tested, supposed to be simpler
"""
def train_model(fold):
    data_path = ''
    df = pd.read_csv(data_path)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = BERTDataset(df_train.review.values, df_train.sentiment_values)
    valid_dataset = BERTDataset(df_valid.review.values, df_valid.sentiment_values)
    
    n_train_steps = int(len(df_train) / 32 * 3)
    model = TextModel(num_classes=1, num_train_steps=n_train_steps)
    
    es = tez.callbacks.EarlyStopping(monitor='valid_loss', patience=3, model_path='toxic_test_model')
    model.fit(train_dataset, valid_dataset=valid_dataset, epochs=3, train_bs=32, callbacks=[es])
    
    model.load('toxic_test_model')
    preds = model.predict(valid_dataset)
    for p in preds:
        print(p)
"""

# Converts the data (again) from the txt in my Jupyter labs to a manner that the BERTDataset can read
def preprocess_data(data_path):
    df = pd.read_csv(data_path, sep='\t', keep_default_na=False)
    enc_tag = preprocessing.LabelEncoder()
    
    df.loc[:, 'tags'] = enc_tag.fit_transform(df['tags'])
    
    sentences = df.groupby('Text #')['tokens'].apply(list).values
    tag = df.groupby('Text #')['tags'].apply(list).values
    return sentences, tag, enc_tag


# training function. Important
def train_fn(data_loader, model, optimizer, device, scheduler):
    # Sets the model to training mode
    model.train()
    # final loss: how well the model does over the batch of data
    final_loss = 0
    # Using tqdm, a library that shows a loading bar so I can track time
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        # Resets the calculated gradient???
        optimizer.zero_grad()
        # Grabs the loss from the data
        _, loss = model(**data)
        # Performs backpropogation
        loss.backward()
        # Steps the optimizer and scheduler
        optimizer.step()
        scheduler.step()
        final_loss = final_loss + float(loss.item())
    return final_loss / len(data_loader)


# Evaluation function, not used yet.
def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in data_loader:
        for k, v in data.items():
            data[k] = v.to(device)
        _, loss = model(**data)
        final_loss = final_loss + loss.item()
    return final_loss / len(data_loader)

'''
def preprocess_predict(data_path):
    df = pd.read_csv(data_path, sep='\t')
    enc_tag = preprocessing.LabelEncoder()
    
    df.loc[:, 'tags'] = enc_tag.fit_transform(df['tags'])
    
    sentences = df.groupby('Text #')['tokens'].apply(list).values
    tag = df.groupby('Text #')['tags'].apply(list).values
    return sentences, tag, enc_tag
'''

if __name__ == '__main__':
    # Loading data, tags, and encoder
    data_path = os.path.join('toxic_data', 'train_full.txt')
    sentences, tag, enc_tag = preprocess_data(data_path)
    
    meta_data = {
        'enc_tag': enc_tag 
    }
    # print(enc_tag.classes_)
    num_tag = len(list(enc_tag.classes_))
    # Saving encoder to bin
    joblib.dump(meta_data, 'meta_data_full.bin')
    # Setting batch size and number of epochs
    BATCH_SIZE = 32
    EPOCHS = 2
    '''
    (
        train_sentences,
        test_sentences,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, tag, random_state=42, test_size=0.1)
    '''
    # Creating a BERTDataset object, 338 is the largest item in our dataset
    train_sentences = sentences
    train_tag = tag
    train_dataset = BERTDataset(
        train_sentences, train_tag, max_len=338
    )
    # collate_fn=lambda x: x, collate_fn=lambda x: x
    # data loader is like an interface between BERTDataset and the nn
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE
    )
    '''
    test_dataset = BERTDataset(
        test_sentences, test_tag, max_len=256
    )
    
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE
    )
    '''
    # Setting the number of training steps
    num_train_steps = int(len(train_sentences) / BATCH_SIZE * EPOCHS)
    
    # Setting the device
    device = torch.device('cpu')
    # Creating the model
    model = EntityModel(num_train_steps, num_tag)
    # Can load model from the dict if necessary
    # model.load_state_dict(torch.load('toxic_model_full_trial_2/toxic_bert_model_full'))
    # Pushing model to the device
    model.to(device)
    
    # I really don't know why this is here because the model has an optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {
            'params': [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.001
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                
            ],
            'weight_decay': 0.0
        }
    ]
    # Setting optimizer
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    # Setting the scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    # Capturing the losses after each epoch
    losses = []
    print('----TRAINING------')
    # Supposed to train until this loss isn't beat
    best_loss = 0.0839775150296864
    # Train the model over the dataset EPOCH number of times
    for epoch in range(EPOCHS):
        # Calls the training function, stores the loss
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        losses.append(train_loss)
        print('Train Loss:', train_loss)
        # Stores the model in a dictionary
        if train_loss < best_loss:
            torch.save(model.state_dict(), 'toxic_bert_model_full')
            best_loss = train_loss
        else:
            break
            
    print('------TRAINING DONE------')
    print('MODEL SAVED AS toxic_bert_model_full')
    
    loss_df = pd.DataFrame({'loss':losses})
    loss_df.to_csv('losses.csv')
    # Excess, should be deleted later
    '''
    print('------PREDICTING---------')
    
    
    meta_data = joblib.load('toxic_model_2345/meta_data_2345.bin')
    enc_tag = meta_data['enc_tag']
    num_tag = len(list(enc_tag.classes_))
    
    sentence1 = "These idiots deserve death."
    sentence1 = sentence1.split()
    
    tokenized_sentence1 = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True).encode(sentence1)
    
    eval_df = pd.to_csv(os.path.join('toxic_data', 'eval_1.csv'))
    
    test_dataset = BERTDataset([sentence1], [[1] * len(sentence1)])
    num_train_steps = 20
    
    
    device = torch.device('cpu')
    model = EntityModel(num_train_steps, num_tag)
    model.load_state_dict(torch.load('toxic_model_2345/toxic_bert_model_2345'))
    model.to(device)
    
    with torch.no_grad():
        model.eval()
        data = test_dataset[0]
        for k, v in data.items():
            print(k)
            data[k] = v.to(device).unsqueeze(0)
        tag, _ = model(**data)
        print(len(tokenized_sentence1[1:-1]), tokenized_sentence1[1:-1])
        print(len(sentence1), sentence1)
        print(len(enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[1:len(tokenized_sentence1) - 1]),
              enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[1:len(tokenized_sentence1) - 1])
        pred_df = pd.DataFrame({'tags':enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[1:len(tokenized_sentence1) - 1]})
        pred_df.to_csv('./example_pred', sep='\t', columns=['tags'], index=False)
    '''
    
    
    
