import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import transformers
import joblib
from main import BERTDataset, EntityModel


if __name__ == '__main__':
    # Gets file paths for the encoder, model, eval data, and a write path for all the tags
    meta_path = os.path.join('full_model_seq_128', 'meta_data_test.bin')
    model_path = os.path.join('full_model_seq_128', 'toxic_bert_model_full_test')
    eval_path = os.path.join('toxic_data', 'tsd_test.csv')
    write_path = os.path.join('tsd_eval_tags_seq_128.csv')
    
    print('------PREDICTING---------')
    
    # Loads the encoder
    meta_data = joblib.load(meta_path)
    enc_tag = meta_data['enc_tag']
    num_tag = len(list(enc_tag.classes_))
    # Train steps is arbitrary, not training here, but this is necessary for the BERTDataset constructor
    num_train_steps = 1
    
    eval_df = pd.read_csv(eval_path, keep_default_na=False)
    # Loading the model and sending it to the device
    device = torch.device('cpu')
    model = EntityModel(num_train_steps, num_tag)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Predicts each item in the evaluation dataset individually :/ couldn't figure this out
    tags = []
    for text in tqdm(eval_df['text']):
        # Splits data
        text = text.split()
        # Lazily creates BERTDataset for one item at a time
        test_dataset = BERTDataset([text], [[1] * len(text)], max_len=294)
        # Tokenizes data for prediction
        tokenized_sentence = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True).encode(text)
        # Sets the gradient of the model to 0
        with torch.no_grad():
            # Sets the model to evaluation mode
            model.eval()
            data = test_dataset[0]
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)
            tag, _, _ = model(**data)
            # print(len(tokenized_sentence1[1:-1]), tokenized_sentence1[1:-1])
            # print(len(sentence1), sentence1)
            # print(len(enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[1:len(tokenized_sentence1) - 1]),
            #       enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[1:len(tokenized_sentence1) - 1])
            tags.append(enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[1:len(tokenized_sentence) - 1])
    
    # Writes the data to file
    pred_df = pd.DataFrame({'tags':tags})
    pred_df.to_csv(write_path, sep='\t', columns=['tags'], index=False)
    print('------DONE--------')
    print('Tags written to', str(write_path))
    
    
    