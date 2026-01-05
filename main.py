import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser

from preprocess import preprocess_data
from load_data import make_loader
from model import MessiTransformer

DATA_DIR = "open_track1/"
SUBMISSION_DIR = "submissions/"


def read_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, "r") as f:
        return json.load(f)

def read_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Data directory path")
    parser.add_argument("--config", type=str, default="config.json", help="Path to model config file")
    args = parser.parse_args()

    config = read_json(args.config)
    cat_cols = config.get("cat_cols", [])
    cont_cols = config.get("cont_cols", [])
    val_split = config.get("val_split", 0.1)
    max_episode_length = config.get("max_episode_length", 128)
    batch_size = config.get("batch_size", 32)
    shuffle = config.get("shuffle", True)

    d_model = config.get("d_model", 128)
    n_head = config.get("n_head", 4)
    num_layers = config.get("num_layers", 2)
    emb_dim = config.get("emb_dim", 16)
    dropout = config.get("dropout", 0.1)

    epochs = config.get("epochs", 10)
    lr = config.get("learning_rate", 1e-3)
    submission_name = config.get("submission_name", "submission.csv")

    train_df = read_csv(os.path.join(args.data_dir, "train.csv"))
    test_df = read_csv(os.path.join(args.data_dir, "test.csv"))
    match_info_df = read_csv(os.path.join(args.data_dir, "match_info.csv"))

    print("[Main] Preprocessing data...")
    train_episodes, val_episodes, test_episodes, scaler, encoders = preprocess_data(
        train_df=train_df, 
        test_df=test_df, 
        data_dir=args.data_dir,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        val_split=val_split,
    )
    print("[Main] Data preprocessed successfully.\n")

    print("[Main] Categorical feature dimension:")
    cat_dims = {}
    for col, le in encoders.items():
        print(f"  {col}: {len(le.classes_)} unique classes (vocab size: {len(le.classes_) + 1})")
        cat_dims[col] = len(le.classes_)

    print("\n[Main] Creating data loaders...")
    train_loader, val_loader, test_loader = make_loader(
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        test_episodes=test_episodes,
        max_episode_length=max_episode_length,
        cont_cols=cont_cols + ["start_x", "start_y", "end_x", "end_y"],
        cat_cols=cat_cols,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    
    print("[Main] Data loaders created successfully.")
    print(f"  Number of training episodes: {len(train_loader.dataset)}")
    print(f"  Number of validation episodes: {len(val_loader.dataset)}")
    print(f"  Number of test episodes: {len(test_loader.dataset)}\n")

    print("[Main] Loading model...")
    model = MessiTransformer(
        cat_dims=cat_dims,
        cont_dim=len(cont_cols) + 4,
        d_model=d_model,
        nhead=n_head,
        num_layers=num_layers,
        emb_dim=emb_dim,
        dropout=dropout,
        max_len=max_episode_length,
    )
    print("[Main] Model loaded successfully.")

    print("\n[Main] Model Summary:")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Training Loop
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[Main] Using device: {device}")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n[Main] Starting training...")
    best_val_loss = float('inf')
    
    for epoch_idx in range(epochs):
        model.train()
        train_loss = 0.0
        for cont_x, cat_x, target, mask, _ in train_loader:
            cont_x = cont_x.to(device)
            cat_x = cat_x.to(device)
            target = target.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            output = model(cont_x, cat_x, padding_mask=mask)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * cont_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cont_x, cat_x, target, mask, _ in val_loader:
                cont_x = cont_x.to(device)
                cat_x = cat_x.to(device)
                target = target.to(device)
                mask = mask.to(device)
                
                output = model(cont_x, cat_x, padding_mask=mask)
                loss = criterion(output, target)
                val_loss += loss.item() * cont_x.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch_idx+1}/{epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  Best model saved.")
    
    print("\n[Main] Training finished.")

    # Inference
    print("\n[Main] Starting inference...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for cont_x, cat_x, _, mask, episode_ids in test_loader:
            cont_x = cont_x.to(device)
            cat_x = cat_x.to(device)
            mask = mask.to(device)
            
            outputs = model(cont_x, cat_x, padding_mask=mask)
            outputs = outputs.cpu().numpy()
            
            # Inverse scaling
            # x: * 105, y: * 68
            outputs[:, 0] *= 105.0
            outputs[:, 1] *= 68.0
            
            for episode_id, pred in zip(episode_ids, outputs):
                results.append({
                    "game_episode": episode_id,
                    "end_x": pred[0],
                    "end_y": pred[1]
                })
    
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(os.path.join(SUBMISSION_DIR, submission_name), index=False)
    print(f"[Main] Submission file saved to {os.path.join(SUBMISSION_DIR, submission_name)}. (Rows: {len(submission_df)})")
    

if __name__ == "__main__":
    main()