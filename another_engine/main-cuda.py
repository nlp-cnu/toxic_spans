# Sourcing for this code is from the following:
# https://www.youtube.com/watch?v=MqQ7rqRllIc
# https://www.youtube.com/watch?v=oreIJQZ40H0&t=1594s
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
        target_tag = target_tag[:self.max_len - 2]

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
    def monitor_metrics(self, outputs, targets):
        outputs = outputs.argmax(2).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        pass_f1_average = np.average([f1(np.where(targets[i] == 1)[0], np.where(outputs == 1)[0])
                                      for i, outputs in enumerate(outputs)])
        pass_accuracy_average = np.average([metrics.accuracy_score(targets[i], outputs)
                                            for i, outputs in enumerate(outputs)])
        pass_recall_average = np.average([recall(np.where(targets[i] == 1)[0], np.where(outputs == 1)[0])
                                          for i, outputs in enumerate(outputs)])
        pass_precision_average = np.average([precision(np.where(targets[i] == 1)[0], np.where(outputs == 1)[0])
                                             for i, outputs in enumerate(outputs)])

        return {
            'f1': pass_f1_average,
            'precision': pass_precision_average,
            'recall': pass_recall_average,
            'accuracy': pass_accuracy_average
        }

    # How the model passes data through itself
    def forward(self, ids, mask, token_type_ids, target_tag):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo_tag = self.bert_drop(o1)
        tag = self.out(bo_tag)
        # print('\ntag transform:', tag.argmax(2).cpu().numpy(), np.shape(tag.argmax(2).cpu().numpy()))
        loss = self.loss(tag, target_tag, mask, self.num_tag)
        met = self.monitor_metrics(tag, target_tag)
        return tag, loss, met


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


# training function. Important. Probably wrong lol
def train_fn(data_loader, model, optimizer, device, scheduler):
    # Sets the model to training mode
    model.train()
    # final loss: how well the model does over the batch of data
    final_loss = 0
    final_f1 = 0
    final_accuracy = 0
    final_precision = 0
    final_recall = 0
    # Using tqdm, a library that shows a loading bar so I can track time
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        # Resets the calculated gradient???
        optimizer.zero_grad()
        # Grabs the loss from the model
        tag, loss, met = model(**data)
        # Performs backpropogation
        loss.backward()
        # Steps the optimizer and scheduler
        optimizer.step()
        scheduler.step()
        final_loss += float(loss.item())
        final_f1 += met['f1']
        final_accuracy += met['accuracy']
        final_precision += met['precision']
        final_recall += met['recall']
    average_loss = final_loss / len(data_loader)
    average_f1 = final_f1 / len(data_loader)
    average_precision = final_precision / len(data_loader)
    average_recall = final_recall / len(data_loader)
    average_accuracy = final_accuracy / len(data_loader)
    mets = {
        'loss': average_loss,
        'f1': average_f1,
        'precision': average_precision,
        'recall': average_recall,
        'accuracy': average_accuracy
    }
    return mets


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


def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1 if len(predictions) == 0 else 0
    nom = 2 * len(set(predictions).intersection(set(gold)))
    denom = len(set(predictions)) + len(set(gold))
    return nom / denom


def precision(predictions, gold):
    """
    Precision
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1 if len(predictions) == 0 else 0
    if len(predictions) == 0:
        return 0
    above = len(set(predictions).intersection(set(gold)))
    below = len(set(predictions))
    return above / below


def recall(predictions, gold):
    """
    Recall
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1 if len(predictions) == 0 else 0
    top = len(set(predictions).intersection(set(gold)))
    bottom = len(set(gold))
    return top / bottom


if __name__ == '__main__':
    # Establishing write paths for all the glorious files
    data_path = 'train_full.txt'
    meta_data_path = 'meta_data_test.bin'
    model_dict_path = 'toxic_bert_model_full_test'

    # Loading data, tags, and encoder
    sentences, tag, enc_tag = preprocess_data(data_path)
    meta_data = {
        'enc_tag': enc_tag
    }
    num_tag = len(list(enc_tag.classes_))
    # Saving encoder to bin
    joblib.dump(meta_data, meta_data_path)
    # Setting batch size and number of epochs
    BATCH_SIZE = 16
    EPOCHS = 15
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
        train_sentences, train_tag, max_len=128
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
    device = torch.device('cuda')
    # Creating the model

    model = EntityModel(num_train_steps, num_tag)
    # Can load model from the dict if necessary
    # model.load_state_dict(torch.load('toxic_model_full_trial_2/toxic_bert_model_full'))
    # Pushing model to the device
    model.to(device)

    # I really don't know why this is here because the model has an optimizer defined in that class
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
    optimizer = AdamW(optimizer_parameters, lr=1e-4)
    # Setting the scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    # Capturing the losses after each epoch
    losses = []
    f1s = []
    precisions = []
    recalls = []
    accuracies = []
    print('----TRAINING------')
    # Supposed to train until this loss isn't beat
    best_loss = np.inf
    # Train the model over the dataset EPOCH number of times
    for epoch in range(EPOCHS):
        # Calls the training function, stores the loss
        mets = train_fn(train_data_loader, model, optimizer, device, scheduler)
        losses.append(mets['loss'])
        f1s.append(mets['f1'])
        precisions.append(mets['precision'])
        recalls.append(mets['recall'])
        accuracies.append(mets['accuracy'])
        print('Train Loss:', mets['loss'])
        print('Train f1:', mets['f1'])
        print('Train precision:', mets['precision'])
        print('Train recall:', mets['recall'])
        print('Train accuracy:', mets['accuracy'])
        # Stores the model in a dictionary
        if mets['loss'] < best_loss:
            torch.save(model.state_dict(), model_dict_path)
            best_loss = mets['loss']

    metrics_df = pd.DataFrame(
        {'loss': losses, 'f1': f1s, 'precision': precisions, 'recall': recalls, 'accuracy': accuracies})
    metrics_df.to_csv('metrics_example.csv')

    print('------TRAINING DONE------')
    print('MODEL SAVED AS {}'.format(model_dict_path))

