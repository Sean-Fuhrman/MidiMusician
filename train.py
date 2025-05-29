
#%%
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MaestroFeatureOneHotDataset, collate_onehot
from tqdm import tqdm
# -------------------------
# Model definitions
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,
                 num_types, num_pitches, num_ctrls):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True)
        self.type_head  = nn.Linear(hidden_dim, num_types)
        self.pitch_head = nn.Linear(hidden_dim, num_pitches)
        self.ctrl_head  = nn.Linear(hidden_dim, num_ctrls)
        # continuous heads: dt, dur, val, bpm
        self.dt_head   = nn.Linear(hidden_dim, 1)
        self.dur_head  = nn.Linear(hidden_dim, 1)
        self.val_head  = nn.Linear(hidden_dim, 1)
        self.bpm_head  = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # x: [B, T, input_dim]

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # [B, T, hidden_dim]
        # heads
        type_logits  = self.type_head(out)
        pitch_logits = self.pitch_head(out)
        ctrl_logits  = self.ctrl_head(out)
        dt_pred      = self.dt_head(out).squeeze(-1)
        dur_pred     = self.dur_head(out).squeeze(-1)
        val_pred     = self.val_head(out).squeeze(-1)
        bpm_pred     = self.bpm_head(out).squeeze(-1)
        return {
            "type": type_logits,
            "pitch": pitch_logits,
            "ctrl": ctrl_logits,
            "dt": dt_pred,
            "dur": dur_pred,
            "val": val_pred,
            "bpm": bpm_pred
        }

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers,
                 num_types, num_pitches, num_ctrls, dim_feedforward=2048):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.type_head  = nn.Linear(d_model, num_types)
        self.pitch_head = nn.Linear(d_model, num_pitches)
        self.ctrl_head  = nn.Linear(d_model, num_ctrls)
        self.dt_head   = nn.Linear(d_model, 1)
        self.dur_head  = nn.Linear(d_model, 1)
        self.val_head  = nn.Linear(d_model, 1)
        self.bpm_head  = nn.Linear(d_model, 1)

    def forward(self, x, lengths):
        # x: [B, T, input_dim]
        B, T, _ = x.shape
        mask = (torch.arange(T, device=lengths.device)
                .unsqueeze(0).expand(B, T)
                >= lengths.unsqueeze(1))
        x_proj = self.input_fc(x)  # [B, T, d_model]
        out = self.transformer(x_proj, src_key_padding_mask=mask)
        type_logits  = self.type_head(out)
        pitch_logits = self.pitch_head(out)
        ctrl_logits  = self.ctrl_head(out)
        dt_pred      = self.dt_head(out).squeeze(-1)
        dur_pred     = self.dur_head(out).squeeze(-1)
        val_pred     = self.val_head(out).squeeze(-1)
        bpm_pred     = self.bpm_head(out).squeeze(-1)
        return {
            "type": type_logits,
            "pitch": pitch_logits,
            "ctrl": ctrl_logits,
            "dt": dt_pred,
            "dur": dur_pred,
            "val": val_pred,
            "bpm": bpm_pred
        }

# -------------------------
# Training loop
# -------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    total_loss = 0.0
    model_path = 'model.pt'

    for batch in tqdm(loader):
        # move to device
        for k in batch:
            batch[k] = batch[k].to(device)
        # prepare inputs & targets
        lengths = batch["lengths"] - 1
        print(f"Batch lengths: {lengths}")
        x_inputs = torch.cat([
            batch["type_oh"][:,:-1],
            batch["pitch_oh"][:,:-1],
            batch["ctrl_oh"][:,:-1],
            batch["dt"][:,:-1].unsqueeze(-1),
            batch["dur"][:,:-1].unsqueeze(-1),
            batch["val"][:,:-1].unsqueeze(-1),
            batch["bpm"][:,:-1].unsqueeze(-1),
        ], dim=2)  # [B, T-1, input_dim]
        # targets
        target_type  = batch["type_oh"][:,1:].argmax(dim=-1)
        target_pitch = batch["pitch_oh"][:,1:].argmax(dim=-1)
        target_ctrl  = batch["ctrl_oh"][:,1:].argmax(dim=-1)
        target_dt    = batch["dt"][:,1:]
        target_dur   = batch["dur"][:,1:]
        target_val   = batch["val"][:,1:]
        target_bpm   = batch["bpm"][:,1:]


        # forward
        preds = model(x_inputs, lengths)


        # compute losses
        loss_type  = ce(preds["type"].transpose(1,2), target_type)
        loss_pitch = ce(preds["pitch"].transpose(1,2), target_pitch)
        loss_ctrl  = ce(preds["ctrl"].transpose(1,2), target_ctrl)
        loss_dt    = mse(preds["dt"],  target_dt)
        loss_dur   = mse(preds["dur"], target_dur)
        loss_val   = mse(preds["val"], target_val)
        loss_bpm   = mse(preds["bpm"], target_bpm)

        loss = (loss_type + loss_pitch + loss_ctrl +
                loss_dt + loss_dur + loss_val + loss_bpm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # Save model 
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path} with loss {total_loss:.4f/ len(loader)}")
    
    #Generate a sample output
    #TODO:
    
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='maestro-v3.0.0/maestro-v3.0.0.csv',)
    parser.add_argument('--midi_dir', type=str, default='maestro-v3.0.0/',)
    parser.add_argument('--model_type', choices=['lstm','transformer'], default='lstm')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers_trf', type=int, default=4)
    args = parser.parse_args()

    # Dataset & Loader
    ds = MaestroFeatureOneHotDataset(args.manifest, args.midi_dir, time_bin=0.01)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_onehot)

    # Determine feature dims
    sample = next(iter(loader))
    B, T, _ = sample["type_oh"].shape
    input_dim = sample["type_oh"].shape[2] + sample["pitch_oh"].shape[2] + \
                sample["ctrl_oh"].shape[2] + 4  # dt,dur,val,bpm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate model
    if args.model_type == 'lstm':
        model = LSTMModel(input_dim, args.hidden_dim,
                          args.num_layers,
                          ds.num_types,
                          ds.num_pitches,
                          ds.num_ctrls)
    else:
        model = TransformerModel(input_dim, args.d_model,
                                 args.nhead, args.num_layers_trf,
                                 ds.num_types, ds.num_pitches,
                                 ds.num_ctrls)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch}/{args.epochs}  Loss: {loss:.4f}")

    # Save
    out_name = f"{args.model_type}_model.pt"
    torch.save(model.state_dict(), out_name)
    print(f"Saved model to {out_name}")


if __name__ == "__main__":
    main()
#%%