#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import torch
import utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pretty_midi
from utils import midi_to_event_list

class MaestroFeatureOneHotDataset(Dataset):
    """
    Dataset returning one-hot encoded categorical features
    plus continuous features for each MIDI event sequence.
    """
    def __init__(self, manifest_csv, midi_dir, time_bin=None):
        self.df       = pd.read_csv(manifest_csv)
        self.midi_dir = midi_dir
        self.time_bin = time_bin

        # Categorical vocabs
        self.type2id   = {"note": 0, "control_change": 1, "set_tempo": 2}
        self.num_types = len(self.type2id)

        self.pitch2id   = {p: p+1 for p in range(128)}
        self.pitch2id[None] = 0
        self.num_pitches = len(self.pitch2id)

        self.ctrl2id    = {64:0, 66:1, 67:2, None:3}
        self.num_ctrls  = len(self.ctrl2id)
        self.max_seq_len = 30_000

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        pm        = pretty_midi.PrettyMIDI(os.path.join(self.midi_dir, row["midi_filename"]))
        events    = midi_to_event_list(pm)

        last_t = 0.0
        # temporary lists
        type_ids, pitch_ids, ctrl_ids = [], [], []
        dts, durs, vals, bpms       = [], [], [], []

        for e in events:
            if len(type_ids) >= self.max_seq_len:
                break
            # time delta
            dt = e["time"] - last_t
            last_t = e["time"]
            if self.time_bin:
                dt = round(dt / self.time_bin) * self.time_bin
            dts.append(dt)

            # categorical
            t_id = self.type2id[e["type"]]
            type_ids.append(t_id)

            if e["type"] == "note":
                pitch_ids.append(self.pitch2id[e["pitch"]])
                ctrl_ids.append(self.ctrl2id[None])
                durs.append(e["end_time"] - e["time"])
                vals.append(e["velocity"])
                bpms.append(0.0)
            elif e["type"] == "control_change":
                pitch_ids.append(self.pitch2id[None])
                ctrl_ids.append(self.ctrl2id[e["controller"]])  # <<— controller, not value
                durs.append(0.0)
                vals.append(e["value"])                         # <<— store the pedal pressure separately
                bpms.append(0.0)
            else:  # set_tempo
                pitch_ids.append(self.pitch2id[None])
                ctrl_ids.append(self.ctrl2id[None])
                durs.append(0.0)
                vals.append(0.0)
                bpms.append(e["bpm"])

        # tensorize and one-hot encode
        type_idx  = torch.tensor(type_ids, dtype=torch.long)
        pitch_idx = torch.tensor(pitch_ids, dtype=torch.long)
        ctrl_idx  = torch.tensor(ctrl_ids, dtype=torch.long)

        type_oh   = F.one_hot(type_idx,  num_classes=self.num_types).float()
        pitch_oh  = F.one_hot(pitch_idx, num_classes=self.num_pitches).float()
        ctrl_oh   = F.one_hot(ctrl_idx,  num_classes=self.num_ctrls).float()


        return {
            "type_oh": type_oh,    # (L, num_types)
            "pitch_oh": pitch_oh,  # (L, num_pitches)
            "ctrl_oh": ctrl_oh,    # (L, num_ctrls)
            "dt": torch.tensor(dts, dtype=torch.float),    # (L,)
            "dur": torch.tensor(durs, dtype=torch.float),  # (L,)
            "val": torch.tensor(vals, dtype=torch.float),  # (L,)
            "bpm": torch.tensor(bpms, dtype=torch.float),  # (L,)
            "ticks_per_beat": torch.tensor(pm.resolution, dtype=torch.long)  # scalar
        }

# Example DataLoader
if __name__ == "__main__":
    manifest = "maestro-v3.0.0/maestro-v3.0.0.csv"
    midi_dir = "maestro-v3.0.0/"
    
    ds = MaestroFeatureOneHotDataset(manifest, midi_dir, time_bin=0.01)
    loader = DataLoader(ds, batch_size=4, shuffle=True,
                        collate_fn=utils.collate_onehot, num_workers=2)

    batch = next(iter(loader))
    print({k: v.shape for k, v in batch.items()})  # inspect shapes
#%%