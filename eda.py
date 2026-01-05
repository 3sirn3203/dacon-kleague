import os
import numpy as np
import pandas as pd


DATA_DIR = "open_track1/"


def read_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)


if __name__ == "__main__":
    train_df = read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = read_csv(os.path.join(DATA_DIR, "test.csv"))

    train_episodes = [group for _, group in train_df.groupby(['game_id', 'period_id', 'episode_id'])]
    train_episode_lengths = [len(episode) for episode in train_episodes]

    test_episodes = []
    for row in test_df.itertuples():
        episode_path = os.path.normpath(os.path.join(DATA_DIR, row.path))
        if not os.path.exists(episode_path):
            raise FileNotFoundError(f"The file {episode_path} does not exist.")
        episode = pd.read_csv(episode_path)
        test_episodes.append(episode)
    test_episode_lengths = [len(episode) for episode in test_episodes]

    print("[EDA] Train Episode Lengths:")
    print(f"  Max: {max(train_episode_lengths)}")
    print(f"  Min: {min(train_episode_lengths)}")
    print(f"  Average: {int(np.mean(train_episode_lengths))}")
    print(f"  95%ile: {int(np.percentile(train_episode_lengths, 95))}")
    print(f"  99%ile: {int(np.percentile(train_episode_lengths, 99))}\n")

    print("[EDA] Test Episode Lengths:")
    print(f"  Max: {max(test_episode_lengths)}")
    print(f"  Min: {min(test_episode_lengths)}")
    print(f"  Average: {int(np.mean(test_episode_lengths))}")
    print(f"  95%ile: {int(np.percentile(test_episode_lengths, 95))}")
    print(f"  99%ile: {int(np.percentile(test_episode_lengths, 99))}")
