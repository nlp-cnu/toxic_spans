import os
import numpy as np
import pandas as pd
import convert_to_bert as ctb
from ast import literal_eval

# train -> preprocess -> bert
# eval -> bert -> postprocess


def read_data(eval_file, bert_output_file):
    """
    reads in the eval file and formats a dataframe as [tokens, tags, pred_tags]
    """
    eval_df = pd.read_csv(eval_file)
    eval_df = pd.read_csv(read_file)
    eval_df['spans'] = df.spans.apply(literal_eval)
    
    tag_df = pd.read_csv(bert_output_file)
    
    return eval_df, tag_df