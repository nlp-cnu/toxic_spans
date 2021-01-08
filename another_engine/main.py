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

class BERTDataset:
    def __init__(self, texts, tags, max_len=128, tokenizer='bert-base-uncased'):
        self.texts = texts
        self.tags = tags
        self.tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer, 
                                                                    do_lower_case=True)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        tags = self.tags[item]
        ids = []
        target_tag = []
        for i, s in enumerate(text):
            inputs = self.tokenizer.encode(s, add_special_tokens=False)     
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)
            
        ids = ids[:self.max_len - 2]
        target = target_tag[:self.max_len - 2]
        
        ids = [101] + ids + [102]
        target_tag = [0] + target_tag + [0]
        
        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
        
        padding_len = self.max_len - len(ids)
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target_tag': torch.tensor(target_tag, dtype=torch.long)
        }


class EntityModel(nn.Module):
    def __init__(self, num_train_steps, num_tag):
        super(EntityModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.bert_drop = nn.Dropout(0.3)
        self.num_tag = num_tag
        self.out = nn.Linear(768, self.num_tag)
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = 'batch'
        
    def fetch_optimizer(self):
        opt = AdamW(self.parameters(), lr=1e-4)
        return opt
    
    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch
    
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
    
    '''def monitor_metrics(self, outputs, targets):
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        return {
            'accuracy': metrics.accuracy_score(targets, outputs)}
    '''
    
    def forward(self, ids, mask, token_type_ids, target_tag):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo_tag = self.bert_drop(o1)
        tag = self.out(bo_tag)
        loss = self.loss(tag, target_tag, mask, self.num_tag)
        # met = self.monitor_metrics(x, targets)
        return tag, loss        
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
def preprocess_data(data_path):
    df = pd.read_csv(data_path, sep='\t', keep_default_na=False)
    enc_tag = preprocessing.LabelEncoder()
    
    df.loc[:, 'tags'] = enc_tag.fit_transform(df['tags'])
    
    sentences = df.groupby('Text #')['tokens'].apply(list).values
    tag = df.groupby('Text #')['tags'].apply(list).values
    return sentences, tag, enc_tag
    
        
def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss = final_loss + loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in data_loader:
        for k, v in data.items():
            data[k] = v.to(device)
        _, loss = model(**data)
        final_loss = final_loss + loss.item()
    return final_loss / len(data_loader)


def preprocess_predict(data_path):
    df = pd.read_csv(data_path, sep='\t')
    enc_tag = preprocessing.LabelEncoder()
    
    df.loc[:, 'tags'] = enc_tag.fit_transform(df['tags'])
    
    sentences = df.groupby('Text #')['tokens'].apply(list).values
    tag = df.groupby('Text #')['tags'].apply(list).values
    return sentences, tag, enc_tag


if __name__ == '__main__':

    data_path = os.path.join('toxic_data', 'train_2345.txt')
    sentences, tag, enc_tag = preprocess_data(data_path)
    
    meta_data = {
        'enc_tag': enc_tag 
    }
    # print(enc_tag.classes_)
    num_tag = len(list(enc_tag.classes_))
    
    joblib.dump(meta_data, 'meta_data.bin')
    
    BATCH_SIZE = 32
    EPOCHS = 3
    '''
    (
        train_sentences,
        test_sentences,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, tag, random_state=42, test_size=0.1)
    '''
    train_sentences = sentences
    train_tag = tag
    train_dataset = BERTDataset(
        train_sentences, train_tag, max_len=338
    )
    # collate_fn=lambda x: x, collate_fn=lambda x: x
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
    num_train_steps = int(len(train_sentences) / BATCH_SIZE * EPOCHS)
    
    device = torch.device('cpu')
    model = EntityModel(num_train_steps, num_tag)
    model.to(device)
    
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
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    print('----TRAINING------')
    # best_loss = np.inf
    for epoch in range(EPOCHS):
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        print('Train Loss:', train_loss)
        '''
        test_loss = eval_fn(test_data_loader, model, device)
        print('Train Loss:', train_loss, '| Test Loss:', test_loss)
        if test_loss < best_loss:
            torch.save(model.state_dict(), 'toxic_test_model')
            best_loss = test_loss
        '''
    print('------TRAINING DONE------')
    torch.save(model.state_dict(), 'toxic_bert_model')
    print('MODEL SAVED AS toxic_bert_model')
    '''
    print('------PREDICTING---------')
    
    
    meta_data = joblib.load('meta_data.bin')
    enc_tag = meta_data['enc_tag']
    num_tag = len(list(enc_tag.classes_))
    
    sentence1 = "samtsirhc is Veryh Great( lol)."
    sentence1 = sentence1.split()
    
    tokenized_sentence1 = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True).encode(sentence1)
    
    
    
    test_dataset = BERTDataset([sentence1], [[1] * len(sentence1)])
    num_train_steps = 20
    
    
    device = torch.device('cpu')
    model = EntityModel(num_train_steps, num_tag)
    model.load_state_dict(torch.load('toxic_bert_model'))
    model.to(device)
    
    with torch.no_grad():
        model.eval()
        data = test_dataset
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, _ = model(**data)
        print(tokenized_sentence1)
        print(sentence1)
        print(enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[1:len(tokenized_sentence1) - 1])
    '''
    
    
    
