from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
import random
from torch.utils.data.distributed import DistributedSampler
import pandas as pd

#####################################################################################

class ColdDataset(data_utils.Dataset):
    def __init__(self, x, y, max_len, mask_token,num_item = 0 ,Eval = False   ):
        self.seqs = x
        self.targets = y
        self.max_len = max_len
        self.mask_token = mask_token
        self.Eval = Eval 
        self.num_item =num_item +1

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        target = self.targets[index]
        seq = seq[-self.max_len:]
        seq_len = len(seq)
        seq_mask_len = self.max_len - seq_len
        
        if self.Eval : 
            labels = [0] * self.num_item
            labels[target] = 1
            seq = [self.mask_token] * seq_mask_len + seq
            return torch.LongTensor(seq), torch.LongTensor(labels)
       
        seq = [0] * seq_mask_len + seq   
        return torch.LongTensor(seq), torch.LongTensor([target])

#####################################################################################################
class cold_reset_df(object):

    def __init__(self):
        print("=" * 10, "Initialize Reset DataFrame Object", "=" * 10)
        self.item_enc1 = LabelEncoder()
        self.item_enc2 = LabelEncoder()
        self.user_enc = LabelEncoder()

    def fit_transform(self, df1, df2):
        print("=" * 10, "Resetting user ids and item ids in DataFrame", "=" * 10)
        df = df1['user_id']._append(df2['user_id'])
        df = self.user_enc.fit_transform(df) + 1
        df1['item_id'] = self.item_enc1.fit_transform(df1['item_id']) + 1
        df1['user_id'] = df[:len(df1)]
        df2['item_id'] = self.item_enc2.fit_transform(df2['item_id']) + 1
        df2['user_id'] = df[len(df1):]
        return df1, df2

    def inverse_transform(self, df):
        df['item_id'] = self.item_enc.inverse_transform(df['item_id']) - 1
        df['user_id'] = self.user_enc.inverse_transform(df['user_id']) - 1
        return df
##########################################################################################################
class item_reset_df(object):

    def __init__(self):
        print("=" * 10, "Initialize Reset DataFrame Object", "=" * 10)
        self.item_enc = LabelEncoder()

    def fit_transform(self, df):
        print("=" * 10, "Resetting item ids in DataFrame", "=" * 10)
        df['item_id'] = self.item_enc.fit_transform(df['item_id']) + 1
        return df

    def inverse_transform(self, df):
        df['item_id'] = self.item_enc.inverse_transform(df['item_id']) - 1
        return df
################################################################################################################
def get_loader(dataset, args):
    
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, pin_memory=True)
    return dataloader
###################################################################################################################

# First way of pre processing

def process_data(args, item_min=50):
    """The function demonstrates a pipeline for preprocessing and preparing data for a recommendation system that aims to handle the cold start problem.The function created to deal with tenrec dataset.It work at all users on target  and source files to create train & valid & test datasets. The function filtered the users in the DFs with less than item_min interactions."""
    
    # Load data from files
    df1 = pd.read_csv(args.target_path, usecols=['user_id', 'item_id', 'click'])
    df2 = pd.read_csv(args.source_path, usecols=['user_id', 'item_id', 'click'])

    # Filter out users with less than item_min interactions
    user_counts = df2.groupby('user_id').size()
    hot_users = user_counts[user_counts >= item_min].index
    df2 = df2[df2.user_id.isin(hot_users)].reset_index(drop=True)

    vocab_size = len(set(df2['item_id']))

    # Encode Userid and item ID using label encoder
    reset_ob = cold_reset_df()  # Assuming cold_reset_df is defined somewhere
    df2, df1 = reset_ob.fit_transform(df2, df1)

    # Get set of users who appear in both datasets
    user_set = set(df1['user_id']).intersection(set(df2['user_id']))

    # Filter out users who appear in only one dataset
    df1 = df1[df1['user_id'].isin(user_set)]
    df2 = df2[df2['user_id'].isin(user_set)]

    # Split users into cold and hot based on interaction count
    cold_user = set(df1['user_id'][df1.groupby('user_id')['item_id'].transform('count') <= 5])
    hot_user = set(df1['user_id'][df1.groupby('user_id')['item_id'].transform('count') > 5])

    # Create new dataframes with user-item interactions
    new_data1 = []
    new_data2 = []
    for u in user_set:
        tmp_data2 = df2[df2.user_id == u][:-3].values.tolist()
        tmp_data1 = df1[df1.user_id == u].values.tolist()
        new_data1.extend(tmp_data1)
        new_data2.extend(tmp_data2)

    new_data1 = pd.DataFrame(new_data1, columns=df1.columns)
    new_data2 = pd.DataFrame(new_data2, columns=df2.columns)

    # Determine number of users in new_data1
    user_count = len(set(new_data1['user_id']))

    reset_item = item_reset_df()  # Assuming item_reset_df is defined somewhere
    new_data1 = reset_item.fit_transform(new_data1)
    item_count = len(set(new_data1['item_id']))

    # Create example lists for hot and cold users
    source_user_history = new_data2.groupby('user_id')['item_id'].apply(list).to_dict()
    target_user_history = new_data1.groupby('user_id')['item_id'].apply(list).to_dict()
    hot_examples = []
    cold_examples = []
    for user_id, target_items in target_user_history.items():
        if user_id in cold_user:
            example_list = cold_examples
        else:
            example_list = hot_examples
        for target_item in target_items:
            source_items = source_user_history.get(user_id, []) + [0]
            example = [source_items, target_item]
            example_list.append(example)
    cold_data = pd.DataFrame(cold_examples, columns=['source', 'target'])
    hot_data = pd.DataFrame(hot_examples, columns=['source', 'target'])

    # Split cold_data into train, validation, and test sets
    size1 = len(cold_data) // 2
    cold_1 = cold_data[:size1]
    cold_2 = cold_data[size1:]
    val_len = len(cold_2) // 2
    train_data = hot_data._append(cold_1)
    val_data = cold_2[:val_len]
    test_data = cold_2[val_len:]
    x_train, y_train = train_data.source.values.tolist(), train_data.target.values.tolist()
    x_val, y_val = val_data.source.values.tolist(), val_data.target.values.tolist()
    x_test, y_test = test_data.source.values.tolist(), test_data.target.values.tolist()

    args.num_users = user_count
    args.num_items = item_count
    args.num_embedding = vocab_size

    # Create dataloader for each dataset
    train_dataset = ColdDataset(x_train, y_train, args.max_len, args.pad_token)
    valid_dataset = ColdDataset(x_val, y_val, args.max_len, args.pad_token, args.num_items, Eval=True)
    test_dataset = ColdDataset(x_test, y_test, args.max_len, args.pad_token, args.num_items, Eval=True)

    train_loader = get_loader(train_dataset, args)
    val_loader = get_loader(valid_dataset, args)
    test_loader = get_loader(test_dataset, args)

    return train_loader, val_loader, test_loader
###############################################################################################################

# Second way of pre processing

def process_dataV2(args, item_min=50):
    """The function demonstrates a pipeline for preprocessing and preparing data for a recommendation system that aims to handle the cold start problem.The function created to deal with tenrec dataset.It filters items on target  and source files using like column to create train & valid & test datasets. The function filtered the users in the DFs with less than item_min interactions."""
    
    # Load data from files
    df1 = pd.read_csv(args.target_path, usecols=['user_id', 'item_id', 'like'])
    df1 = df1[df1.like.isin([1])]
    df2 = pd.read_csv(args.source_path, usecols=['user_id', 'item_id', 'like'])
    df2 = df2[df2.like.isin([1])]

    # Filter out users with less than item_min interactions
    user_counts = df2.groupby('user_id').size()
    hot_users = user_counts[user_counts >= item_min].index
    df2 = df2[df2.user_id.isin(hot_users)].reset_index(drop=True)

    vocab_size = len(set(df2['item_id']))

    # Encode Userid and item ID using label encoder
    reset_ob = cold_reset_df()  # Assuming cold_reset_df is defined somewhere
    df2, df1 = reset_ob.fit_transform(df2, df1)

    # Get set of users who appear in both datasets
    user_set = set(df1['user_id']).intersection(set(df2['user_id']))

    # Filter out users who appear in only one dataset
    df1 = df1[df1['user_id'].isin(user_set)]
    df2 = df2[df2['user_id'].isin(user_set)]

    # Split users into cold and hot based on interaction count
    cold_user = set(df1['user_id'][df1.groupby('user_id')['item_id'].transform('count') <= 5])
    hot_user = set(df1['user_id'][df1.groupby('user_id')['item_id'].transform('count') > 5])

    # Create new dataframes with user-item interactions
    new_data1 = []
    new_data2 = []
    for u in user_set:
        tmp_data2 = df2[df2.user_id == u][:-3].values.tolist()
        tmp_data1 = df1[df1.user_id == u].values.tolist()
        new_data1.extend(tmp_data1)
        new_data2.extend(tmp_data2)

    new_data1 = pd.DataFrame(new_data1, columns=df1.columns)
    new_data2 = pd.DataFrame(new_data2, columns=df2.columns)

    # Determine number of users in new_data1
    user_count = len(set(new_data1['user_id']))

    reset_item = item_reset_df()  # Assuming item_reset_df is defined somewhere
    new_data1 = reset_item.fit_transform(new_data1)
    item_count = len(set(new_data1['item_id']))

    # Create example lists for hot and cold users
    source_user_history = new_data2.groupby('user_id')['item_id'].apply(list).to_dict()
    target_user_history = new_data1.groupby('user_id')['item_id'].apply(list).to_dict()
    hot_examples = []
    cold_examples = []
    for user_id, target_items in target_user_history.items():
        if user_id in cold_user:
            example_list = cold_examples
        else:
            example_list = hot_examples
        for target_item in target_items:
            source_items = source_user_history.get(user_id, []) + [0]
            example = [source_items, target_item]
            example_list.append(example)
    cold_data = pd.DataFrame(cold_examples, columns=['source', 'target'])
    hot_data = pd.DataFrame(hot_examples, columns=['source', 'target'])

    # Split cold_data into train, validation, and test sets
    size1 = len(cold_data) // 2
    cold_1 = cold_data[:size1]
    cold_2 = cold_data[size1:]
    val_len = len(cold_2) // 2
    train_data = hot_data._append(cold_1)
    val_data = cold_2[:val_len]
    test_data = cold_2[val_len:]
    x_train, y_train = train_data.source.values.tolist(), train_data.target.values.tolist()
    x_val, y_val = val_data.source.values.tolist(), val_data.target.values.tolist()
    x_test, y_test = test_data.source.values.tolist(), test_data.target.values.tolist()

    args.num_users = user_count
    args.num_items = item_count
    args.num_embedding = vocab_size

    # Create dataloader for each dataset
    train_dataset = ColdDataset(x_train, y_train, args.max_len, args.pad_token)
    valid_dataset = ColdDataset(x_val, y_val, args.max_len, args.pad_token, args.num_items, Eval=True)
    test_dataset = ColdDataset(x_test, y_test, args.max_len, args.pad_token, args.num_items, Eval=True)

    train_loader = get_loader(train_dataset, args)
    val_loader = get_loader(valid_dataset, args)
    test_loader = get_loader(test_dataset, args)

    return train_loader, val_loader, test_loader

###############################################################################################################


class cold_reset_dfV2(object):

    def __init__(self):
        print("=" * 10, "Initialize Reset DataFrame Object", "=" * 10)
        self.item_enc1 = LabelEncoder()
        self.user_enc = LabelEncoder()

    def fit_transform(self, df1):
        print("=" * 10, "Resetting user ids and item ids in DataFrame", "=" * 10)
        df = df1.copy()
        df['user_id'] = self.user_enc.fit_transform(df1['user_id']) + 1
        df['item_id'] = self.item_enc1.fit_transform(df1['item_id']) + 1
        return df

    def inverse_transform(self, df):
        df['item_id'] = self.item_enc.inverse_transform(df['item_id']) - 1
        df['user_id'] = self.user_enc.inverse_transform(df['user_id']) - 1
        return df

##################################################################################################################    

# Third way of pre processing

def process_dataV3(args, item_min=2):
    """The function demonstrates a pipeline for preprocessing and preparing data for a recommendation system that aims to handle the cold start problem.The function created to deal with tenrec dataset.It works on one file to create train & valid & test datasets. It takes the last item_min items as the target and the rest items as asource."""
    
    # Load data from files
    df = pd.read_csv(args.source_path, usecols=['user_id', 'item_id', 'click'])
    

    # Filter out users with less than item_min interactions
    user_counts = df.groupby('user_id').size()
    users = user_counts[user_counts >= item_min].index
    df = df[df.user_id.isin(users)].reset_index(drop=True)

    vocab_size = len(set(df['item_id']))

    # Encode Userid and item ID using label encoder
    reset_ob = cold_reset_dfV2()  
    df = reset_ob.fit_transform(df)

    # Get set of users who appear in both datasets
    user_set = set(df['user_id'])


    # Split users into cold and hot based on interaction count
    cold_user = set(df['user_id'][df.groupby('user_id')['item_id'].transform('count') <= 5])
    hot_user = set(df['user_id'][df.groupby('user_id')['item_id'].transform('count') > 5])

    # Create new dataframes with user-item interactions
    new_data1 = []
    new_data2 = []
    for u in user_set:
        tmp_data2 = df[df.user_id == u][:-args.target_num].values.tolist()
        tmp_data1 = df[df.user_id == u].tail(args.target_num).values.tolist()
        new_data1.extend(tmp_data1)
        new_data2.extend(tmp_data2)

    new_data1 = pd.DataFrame(new_data1, columns=df.columns)
    new_data2 = pd.DataFrame(new_data2, columns=df.columns)

    # Determine number of users in new_data1
    user_count = len(set(new_data1['user_id']))

    reset_item = item_reset_df() 
    new_data1 = reset_item.fit_transform(new_data1)
    item_count = len(set(new_data1['item_id']))

    # Create example lists for hot and cold users
    source_user_history = new_data2.groupby('user_id')['item_id'].apply(list).to_dict()
    target_user_history = new_data1.groupby('user_id')['item_id'].apply(list).to_dict()
    hot_examples = []
    cold_examples = []
    for user_id, target_items in target_user_history.items():
        if user_id in cold_user:
            example_list = cold_examples
        else:
            example_list = hot_examples
        for target_item in target_items:
            source_items = source_user_history.get(user_id, []) + [0]
            example = [source_items, target_item]
            example_list.append(example)
    cold_data = pd.DataFrame(cold_examples, columns=['source', 'target'])
    hot_data = pd.DataFrame(hot_examples, columns=['source', 'target'])

    # Split cold_data into train, validation, and test sets
    size1 = len(cold_data) // 2
    cold_1 = cold_data[:size1]
    cold_2 = cold_data[size1:]
    val_len = len(cold_2) // 2
    train_data = hot_data._append(cold_1)
    val_data = cold_2[:val_len]
    test_data = cold_2[val_len:]
    x_train, y_train = train_data.source.values.tolist(), train_data.target.values.tolist()
    x_val, y_val = val_data.source.values.tolist(), val_data.target.values.tolist()
    x_test, y_test = test_data.source.values.tolist(), test_data.target.values.tolist()

    args.num_users = user_count
    args.num_items = item_count
    args.num_embedding = vocab_size

    # Create dataloader for each dataset
    train_dataset = ColdDataset(x_train, y_train, args.max_len, args.pad_token)
    valid_dataset = ColdDataset(x_val, y_val, args.max_len, args.pad_token, args.num_items, Eval=True)
    test_dataset = ColdDataset(x_test, y_test, args.max_len, args.pad_token, args.num_items, Eval=True)

    train_loader = get_loader(train_dataset, args)
    val_loader = get_loader(valid_dataset, args)
    test_loader = get_loader(test_dataset, args)

    return train_loader, val_loader, test_loader

########################################################################################################################

# Fourth way of preprocessing

def process_dataV4(args, item_min=2):
    
    """The function demonstrates a pipeline for preprocessing and preparing data for a recommendation system that aims to handle the cold start problem.The function created to deal with tenrec dataset.It works on one file to create train & valid & test datasets. It takes the from random index item_min items as the target and the rest items as a source."""
    
    # Load data from files
    df = pd.read_csv(args.source_path, usecols=['user_id', 'item_id', 'click'])
    

    # Filter out users with less than item_min interactions
    user_counts = df.groupby('user_id').size()
    users = user_counts[user_counts >= item_min].index
    df = df[df.user_id.isin(users)].reset_index(drop=True)

    vocab_size = len(set(df['item_id']))

    # Encode Userid and item ID using label encoder
    reset_ob = cold_reset_dfV2()  
    df = reset_ob.fit_transform(df)

    # Get set of users who appear in both datasets
    user_set = set(df['user_id'])


    # Split users into cold and hot based on interaction count
    cold_user = set(df['user_id'][df.groupby('user_id')['item_id'].transform('count') <= 5])
    hot_user = set(df['user_id'][df.groupby('user_id')['item_id'].transform('count') > 5])

    # Create new dataframes with user-item interactions
    new_data1 = []
    new_data2 = []
    for u in user_set:
        
        # Select multiple random rows from the DataFrame
        random_rows = df[df.user_id == u].sample(n=args.target_num)

        # Drop the selected random rows and get the rest of the rows
        rest_of_rows = df[df.user_id == u].drop(random_rows.index)

        tmp_data2 = rest_of_rows.values.tolist()
        tmp_data1 = random_rows.values.tolist()
        
        
        new_data1.extend(tmp_data1)
        new_data2.extend(tmp_data2)

    new_data1 = pd.DataFrame(new_data1, columns=df.columns)
    new_data2 = pd.DataFrame(new_data2, columns=df.columns)

    # Determine number of users in new_data1
    user_count = len(set(new_data1['user_id']))

    reset_item = item_reset_df() 
    new_data1 = reset_item.fit_transform(new_data1)
    item_count = len(set(new_data1['item_id']))

    # Create example lists for hot and cold users
    source_user_history = new_data2.groupby('user_id')['item_id'].apply(list).to_dict()
    target_user_history = new_data1.groupby('user_id')['item_id'].apply(list).to_dict()
    hot_examples = []
    cold_examples = []
    for user_id, target_items in target_user_history.items():
        if user_id in cold_user:
            example_list = cold_examples
        else:
            example_list = hot_examples
        for target_item in target_items:
            source_items = source_user_history.get(user_id, []) + [0]
            example = [source_items, target_item]
            example_list.append(example)
    cold_data = pd.DataFrame(cold_examples, columns=['source', 'target'])
    hot_data = pd.DataFrame(hot_examples, columns=['source', 'target'])

    # Split cold_data into train, validation, and test sets
    size1 = len(cold_data) // 2
    cold_1 = cold_data[:size1]
    cold_2 = cold_data[size1:]
    val_len = len(cold_2) // 2
    train_data = hot_data._append(cold_1)
    val_data = cold_2[:val_len]
    test_data = cold_2[val_len:]
    x_train, y_train = train_data.source.values.tolist(), train_data.target.values.tolist()
    x_val, y_val = val_data.source.values.tolist(), val_data.target.values.tolist()
    x_test, y_test = test_data.source.values.tolist(), test_data.target.values.tolist()

    args.num_users = user_count
    args.num_items = item_count
    args.num_embedding = vocab_size

    # Create dataloader for each dataset
    train_dataset = ColdDataset(x_train, y_train, args.max_len, args.pad_token)
    valid_dataset = ColdDataset(x_val, y_val, args.max_len, args.pad_token, args.num_items, Eval=True)
    test_dataset = ColdDataset(x_test, y_test, args.max_len, args.pad_token, args.num_items, Eval=True)

    train_loader = get_loader(train_dataset, args)
    val_loader = get_loader(valid_dataset, args)
    test_loader = get_loader(test_dataset, args)

    return train_loader, val_loader, test_loader
########################################################################################################################

# Fifth way of preprocessing


class cold_reset_dfV3(object):

    def __init__(self):
        print("=" * 10, "Initialize Reset DataFrame Object", "=" * 10)
        self.item_enc1 = LabelEncoder()
        self.user_enc = LabelEncoder()

    def fit_transform(self, df1):
        print("=" * 10, "Resetting user ids and item ids in DataFrame", "=" * 10)
        df = df1.copy()
        df['UserId'] = self.user_enc.fit_transform(df1['UserId']) + 1
        df['ProductId'] = self.item_enc1.fit_transform(df1['ProductId']) + 1
        return df

    def inverse_transform(self, df):
        df['ProductId'] = self.item_enc.inverse_transform(df['ProductId']) - 1
        df['UserId'] = self.user_enc.inverse_transform(df['UserId']) - 1
        return df

    
    
class item_reset_dfV2(object):

    def __init__(self):
        print("=" * 10, "Initialize Reset DataFrame Object", "=" * 10)
        self.item_enc = LabelEncoder()

    def fit_transform(self, df):
        print("=" * 10, "Resetting item ids in DataFrame", "=" * 10)
        df['ProductId'] = self.item_enc.fit_transform(df['ProductId']) + 1
        return df

    def inverse_transform(self, df):
        df['ProductId'] = self.item_enc.inverse_transform(df['ProductId']) - 1
        return df
    
    
    

def process_dataV5(args, item_min=2):
    
    """The function demonstrates a pipeline for preprocessing and preparing data for a recommendation system that aims to handle the cold start problem.The function created to deal with amazon dataset.It works on one file to create train & valid & test datasets. It takes the from random index item_min items as the target and the rest items as a source."""
    
    # Load data from files
    df = pd.read_csv(args.source_path, usecols=['UserId', 'ProductId','Rating'])
    df = df[ df.Rating == 5]

    # Filter out users with less than item_min interactions
    user_counts = df.groupby('UserId').size()
    users = user_counts[user_counts >= item_min].index
    df = df[df.UserId.isin(users)  ].reset_index(drop=True)

    vocab_size = len(set(df['ProductId']))

    # Encode Userid and item ID using label encoder
    reset_ob = cold_reset_dfV3()  
    df = reset_ob.fit_transform(df)

    # Get set of users who appear in both datasets
    user_set = set(df['UserId'])


    # Split users into cold and hot based on interaction count
    cold_user = set(df['UserId'][df.groupby('UserId')['ProductId'].transform('count') <= 5])
    hot_user = set(df['UserId'][df.groupby('UserId')['ProductId'].transform('count') > 5])

    # Create new dataframes with user-item interactions
    new_data1 = []
    new_data2 = []
    for u in user_set:
        
        # Select multiple random rows from the DataFrame
        random_rows = df[df.UserId == u].sample(n=args.target_num)

        # Drop the selected random rows and get the rest of the rows
        rest_of_rows = df[df.UserId == u].drop(random_rows.index)

        tmp_data2 = rest_of_rows.values.tolist()
        tmp_data1 = random_rows.values.tolist()
        
        
        new_data1.extend(tmp_data1)
        new_data2.extend(tmp_data2)

    new_data1 = pd.DataFrame(new_data1, columns=df.columns)
    new_data2 = pd.DataFrame(new_data2, columns=df.columns)

    # Determine number of users in new_data1
    user_count = len(set(new_data1['UserId']))

    reset_item = item_reset_dfV2() 
    new_data1 = reset_item.fit_transform(new_data1)
    item_count = len(set(new_data1['ProductId']))

    # Create example lists for hot and cold users
    source_user_history = new_data2.groupby('UserId')['ProductId'].apply(list).to_dict()
    target_user_history = new_data1.groupby('UserId')['ProductId'].apply(list).to_dict()
    hot_examples = []
    cold_examples = []
    for user_id, target_items in target_user_history.items():
        if user_id in cold_user:
            example_list = cold_examples
        else:
            example_list = hot_examples
        for target_item in target_items:
            source_items = source_user_history.get(user_id, []) + [0]
            example = [source_items, target_item]
            example_list.append(example)
    cold_data = pd.DataFrame(cold_examples, columns=['source', 'target'])
    hot_data = pd.DataFrame(hot_examples, columns=['source', 'target'])

    # Split cold_data into train, validation, and test sets
    size1 = len(cold_data) // 2
    cold_1 = cold_data[:size1]
    cold_2 = cold_data[size1:]
    val_len = len(cold_2) // 2
    train_data = hot_data._append(cold_1)
    val_data = cold_2[:val_len]
    test_data = cold_2[val_len:]
    x_train, y_train = train_data.source.values.tolist(), train_data.target.values.tolist()
    x_val, y_val = val_data.source.values.tolist(), val_data.target.values.tolist()
    x_test, y_test = test_data.source.values.tolist(), test_data.target.values.tolist()

    args.num_users = user_count
    args.num_items = item_count
    args.num_embedding = vocab_size

    # Create dataloader for each dataset
    train_dataset = ColdDataset(x_train, y_train, args.max_len, args.pad_token)
    valid_dataset = ColdDataset(x_val, y_val, args.max_len, args.pad_token, args.num_items, Eval=True)
    test_dataset = ColdDataset(x_test, y_test, args.max_len, args.pad_token, args.num_items, Eval=True)

    train_loader = get_loader(train_dataset, args)
    val_loader = get_loader(valid_dataset, args)
    test_loader = get_loader(test_dataset, args)

    return train_loader, val_loader, test_loader

######################################################################################################################

# Six way of preprocessing 


def process_dataV6(args, item_min=2):
    
    """The function demonstrates a pipeline for preprocessing and preparing data for a recommendation system that aims to handle the cold start problem.The function created to deal with amazon dataset.It works on one file to create train & valid & test datasets. It takes the last item_min items as the target and the rest items as a source."""
    
    # Load data from files
    df = pd.read_csv(args.source_path, usecols=['ProductId', 'UserId','Rating'])
    df = df[ df.Rating == 5]
    

    # Filter out users with less than item_min interactions
    user_counts = df.groupby('UserId').size()
    users = user_counts[user_counts >= item_min].index
    df = df[df.UserId.isin(users) ].reset_index(drop=True)

    vocab_size = len(set(df['ProductId']))

    # Encode Userid and item ID using label encoder
    reset_ob = cold_reset_dfV3()  
    df = reset_ob.fit_transform(df)

    # Get set of users who appear in both datasets
    user_set = set(df['UserId'])


    # Split users into cold and hot based on interaction count
    cold_user = set(df['UserId'][df.groupby('UserId')['ProductId'].transform('count') <= 5])
    hot_user = set(df['UserId'][df.groupby('UserId')['ProductId'].transform('count') > 5])

    # Create new dataframes with user-item interactions
    new_data1 = []
    new_data2 = []
    for u in user_set:
        tmp_data2 = df[df.UserId == u][:-args.target_num].values.tolist()
        tmp_data1 = df[df.UserId == u].tail(args.target_num).values.tolist()
        new_data1.extend(tmp_data1)
        new_data2.extend(tmp_data2)

    new_data1 = pd.DataFrame(new_data1, columns=df.columns)
    new_data2 = pd.DataFrame(new_data2, columns=df.columns)

    # Determine number of users in new_data1
    user_count = len(set(new_data1['UserId']))

    reset_item = item_reset_dfV2() 
    new_data1 = reset_item.fit_transform(new_data1)
    item_count = len(set(new_data1['ProductId']))

    # Create example lists for hot and cold users
    source_user_history = new_data2.groupby('UserId')['ProductId'].apply(list).to_dict()
    target_user_history = new_data1.groupby('UserId')['ProductId'].apply(list).to_dict()
    hot_examples = []
    cold_examples = []
    for user_id, target_items in target_user_history.items():
        if user_id in cold_user:
            example_list = cold_examples
        else:
            example_list = hot_examples
        for target_item in target_items:
            source_items = source_user_history.get(user_id, []) + [0]
            example = [source_items, target_item]
            example_list.append(example)
    cold_data = pd.DataFrame(cold_examples, columns=['source', 'target'])
    hot_data = pd.DataFrame(hot_examples, columns=['source', 'target'])

    # Split cold_data into train, validation, and test sets
    size1 = len(cold_data) // 2
    cold_1 = cold_data[:size1]
    cold_2 = cold_data[size1:]
    val_len = len(cold_2) // 2
    train_data = hot_data._append(cold_1)
    val_data = cold_2[:val_len]
    test_data = cold_2[val_len:]
    x_train, y_train = train_data.source.values.tolist(), train_data.target.values.tolist()
    x_val, y_val = val_data.source.values.tolist(), val_data.target.values.tolist()
    x_test, y_test = test_data.source.values.tolist(), test_data.target.values.tolist()

    args.num_users = user_count
    args.num_items = item_count
    args.num_embedding = vocab_size

    # Create dataloader for each dataset
    train_dataset = ColdDataset(x_train, y_train, args.max_len, args.pad_token)
    valid_dataset = ColdDataset(x_val, y_val, args.max_len, args.pad_token, args.num_items, Eval=True)
    test_dataset = ColdDataset(x_test, y_test, args.max_len, args.pad_token, args.num_items, Eval=True)

    train_loader = get_loader(train_dataset, args)
    val_loader = get_loader(valid_dataset, args)
    test_loader = get_loader(test_dataset, args)

    return train_loader, val_loader, test_loader