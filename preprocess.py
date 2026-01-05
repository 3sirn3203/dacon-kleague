import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit


def preprocess_data(train_df, test_df, data_dir, 
                    cont_cols=None, cat_cols=None, val_split=0.1):

    # DataFrame을 episode 단위로 나누어 리스트로 변환
    train_val_episodes = [group for _, group in train_df.groupby(['game_id', 'period_id', 'episode_id'])]
    test_episodes = []
    for row in test_df.itertuples():
        episode_path = os.path.normpath(os.path.join(data_dir, row.path))
        if not os.path.exists(episode_path):
            raise FileNotFoundError(f"The file {episode_path} does not exist.")
        episode = pd.read_csv(episode_path)
        test_episodes.append(episode)

    # train, val 분할: game_id 기준으로 그룹화하여 분할
    game_ids = [ep['game_id'].iloc[0] for ep in train_val_episodes]
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(splitter.split(train_val_episodes, groups=game_ids))
    
    train_episodes = [train_val_episodes[i] for i in train_idx]
    val_episodes = [train_val_episodes[i] for i in val_idx]
    
    # 좌표값을 0에서 1 사이의 값으로 매핑
    x_max = 105.0
    y_max = 68.0

    for episode in train_episodes + val_episodes + test_episodes:
        # test episode의 마지막 sequence에는 end_x, end_y가 NaN이므로 0으로 채움
        episode[['end_x', 'end_y']] = episode[['end_x', 'end_y']].fillna(0.0)

        episode['start_x'] = episode['start_x'] / x_max
        episode['end_x'] = episode['end_x'] / x_max
        episode['start_y'] = episode['start_y'] / y_max
        episode['end_y'] = episode['end_y'] / y_max

    # continuous features 스케일링 적용
    scaler = None
    if cont_cols:
        scaler = StandardScaler()
        all_cont_data = pd.concat([episode[cont_cols] for episode in train_episodes], ignore_index=True)
        scaler.fit(all_cont_data)

        for i in range(len(train_episodes)):
            train_episodes[i][cont_cols] = scaler.transform(train_episodes[i][cont_cols])

        for i in range(len(val_episodes)):
            val_episodes[i][cont_cols] = scaler.transform(val_episodes[i][cont_cols])

        for i in range(len(test_episodes)):
            test_episodes[i][cont_cols] = scaler.transform(test_episodes[i][cont_cols])

    # categorical features에 Label Encoding 적용
    encoders = {}
    if cat_cols:
        all_cat_data = pd.concat([episode[cat_cols] for episode in train_episodes], ignore_index=True)
        
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(all_cat_data[col].astype(str)) # player_id, team_id 등을 위해 문자열로 변환 후 fit
            
            # Unseen Label 처리를 위한 매핑 딕셔너리
            label_map = dict(zip(le.classes_, le.transform(le.classes_)))
            
            def safe_transform(series):
                series = series.astype(str)
                # unknown label은 -1로 매핑
                return series.map(lambda x: label_map.get(x, -1))

            encoders[col] = le

            # 결과적으로 unknown label은 0, 기존 라벨들은 1부터 시작하도록 변환
            for i in range(len(train_episodes)):
                encoded = safe_transform(train_episodes[i][col])
                train_episodes[i][col] = encoded + 1

            for i in range(len(val_episodes)):
                encoded = safe_transform(val_episodes[i][col])
                val_episodes[i][col] = encoded + 1

            for i in range(len(test_episodes)):
                encoded = safe_transform(test_episodes[i][col])
                test_episodes[i][col] = encoded + 1

    return train_episodes, val_episodes, test_episodes, scaler, encoders