import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class SoccerDataset(Dataset):
    def __init__(self, data, max_episode_length=128, cont_cols=None, cat_cols=None, mode='train'):
        self.data = data
        self.max_episode_length = max_episode_length
        self.cont_cols = cont_cols if cont_cols else ["time_seconds", "start_x", "start_y", "end_x", "end_y"]
        self.cat_cols = cat_cols if cat_cols else ["team_id", "player_id", "type_name", "result_name"]
        self.mode = mode

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        episode = self.data[idx]

        # episode의 길이가 max_episode_length보다 길면 뒤에서부터 자르기
        if len(episode) > self.max_episode_length:
            episode = episode.iloc[-self.max_episode_length :]

        target_row = episode.iloc[-1]
        if self.mode == 'train':
            target_tensor = torch.tensor(target_row[["end_x", "end_y"]].values.astype(np.float32), dtype=torch.float32)
        else:
            target_tensor = torch.zeros(2, dtype=torch.float32)

        cont_values = episode[self.cont_cols].values.astype(np.float32)
        cat_values = episode[self.cat_cols].values.astype(np.int64)

        # episode 마지막 행의 end_x, end_y 값을 마스킹 (모델이 예측해야하는 값)
        masking_end_x = self.cont_cols.index("end_x")
        masking_end_y = self.cont_cols.index("end_y")
        cont_values[-1, masking_end_x] = 0.0
        cont_values[-1, masking_end_y] = 0.0

        cont_tensor = torch.tensor(cont_values, dtype=torch.float32)
        cat_tensor = torch.tensor(cat_values, dtype=torch.long)
        
        game_episode_id = str(episode['game_episode'].iloc[0])

        return cont_tensor, cat_tensor, target_tensor, game_episode_id


def collate_fn(batch):
    cont_list, cat_list, target_list, id_list = zip(*batch)

    cont_padded = pad_sequence(cont_list, batch_first=True, padding_value=0.0)
    cat_padded = pad_sequence(cat_list, batch_first=True, padding_value=0)

    targets = torch.stack(target_list)

    length = torch.tensor([len(x) for x in cont_list])
    max_len = cont_padded.size(1)

    mask = torch.arange(max_len)[None, :] >= length[:, None]

    return cont_padded, cat_padded, targets, mask, id_list


def make_loader(train_episodes, val_episodes, test_episodes, max_episode_length=128,
                cont_cols=None, cat_cols=None, batch_size=32, shuffle=True):

    train_dataset = SoccerDataset(
        data=train_episodes, 
        max_episode_length=max_episode_length, 
        cont_cols=cont_cols, 
        cat_cols=cat_cols,
        mode='train'
    )
    val_dataset = SoccerDataset(
        data=val_episodes, 
        max_episode_length=max_episode_length, 
        cont_cols=cont_cols, 
        cat_cols=cat_cols,
        mode='train'
    )
    test_dataset = SoccerDataset(
        data=test_episodes,
        max_episode_length=max_episode_length, 
        cont_cols=cont_cols, 
        cat_cols=cat_cols,
        mode='test'
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader