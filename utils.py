#%%
import pandas as pd
import pretty_midi
import os
import torch
import torch.nn.functional as F
import pretty_midi
from mido import MidiFile, MidiTrack, Message, MetaMessage


def midi_to_event_list(midi_data):
    """
    Convert a PrettyMIDI object into a list of events conforming to:
    
      "note":           ["start_time","end_time","track","duration","channel","pitch","velocity"]
      "control_change": ["time","track", "value"]
      "set_tempo":      ["time","track","bpm"]
      "ticks_per_beat": ["ticks_per_beat"]
    """
    events = []
    track_idx = 0  # MAESTRO is single-instrument, so one “track”

    inst = midi_data.instruments[0]


    
    # 2) tempo changes
    times, tempi = midi_data.get_tempo_changes()  # times in sec, tempi in µs per beat
    for t, uspb in zip(times, tempi):
        bpm = 60_000_000 / uspb
        events.append({
            "type":  "set_tempo",
            "time": t,
            "track": track_idx,
            "bpm":   bpm
        })
    
    # 3) control changes (pedals etc.)
    for cc in inst.control_changes:
        events.append({
            "type":       "control_change",
            "time":      cc.time,
            "controller": cc.number,  # MIDI controller number
            "track":      track_idx,
            "value":      cc.value
        })
    
    # 4) notes (automatically paired start/end by pretty_midi)
    for note in inst.notes:
        events.append({
            "type":     "note",
            "time":    note.start,
            "end_time":    note.end,
            "pitch":    note.pitch,
            "velocity": note.velocity
        })
    
    # Sort events by time
    events.sort(key=lambda x: x["time"])
    return events

from mido import MidiFile, MidiTrack, Message, MetaMessage

def events_to_midi(events, output_path='reconstructed.mid'):
    """
    Reconstruct a .mid from your event dicts, reading PPQN from the first token.

    events[0] must be:
        {"type": "ticks_per_beat", "ticks_per_beat": <int>}

    Subsequent events are dicts with keys:
        - type: 'set_tempo', 'control_change', 'note', (optional 'patch_change')
        - time: float seconds
        For 'set_tempo':        'bpm'
        For 'control_change':  'value' (and optionally 'controller', 'channel')
        For 'note':            'pitch', 'velocity', 'end_time' (and optionally 'channel')
        For 'patch_change':    'patch' (and optionally 'channel')
    """
    # 0) Pull out PPQN from the first event
    first = events[0]
    if first.get("type") != "ticks_per_beat":
        raise ValueError("First event must be a ticks_per_beat token")
    ticks_per_beat = first["ticks_per_beat"]
    payload = events[1:]  # the rest

    # 1) Create type-1 MIDI with two tracks: meta & performance
    mid = MidiFile(type=1)
    mid.ticks_per_beat = ticks_per_beat
    meta_track = MidiTrack(); perf_track = MidiTrack()
    mid.tracks.append(meta_track)
    mid.tracks.append(perf_track)

    # 2) Expand events into mido‐ready messages
    expanded = []
    for e in payload:
        t = e['time']
        if e['type'] == 'note':
            expanded.append({
                'time':     t,
                'kind':     'note_on',
                'pitch':    e['pitch'],
                'velocity': e['velocity'],
                'channel':  e.get('channel', 0)
            })
            expanded.append({
                'time':     e['end_time'],
                'kind':     'note_off',
                'pitch':    e['pitch'],
                'velocity': 0,
                'channel':  e.get('channel', 0)
            })
        elif e['type'] == 'set_tempo':
            uspb = int(60_000_000 / e['bpm'])
            expanded.append({'time': t, 'kind': 'set_tempo', 'tempo': uspb})
        elif e['type'] == 'control_change':
            ctrl = e.get('controller', 64)
            expanded.append({
                'time':       t,
                'kind':       'control_change',
                'controller': ctrl,
                'value':      e['value'],
                'channel':    e.get('channel', 0)
            })
        elif e['type'] == 'patch_change':
            expanded.append({
                'time':    t,
                'kind':    'program_change',
                'program': e['patch'],
                'channel': e.get('channel', 0)
            })

    # 3) Sort by time
    expanded.sort(key=lambda x: x['time'])

    # 4) Emit into tracks, converting seconds → ticks via current tempo
    last_meta_time = 0.0
    last_perf_time = 0.0
    last_tempo_us  = None

    for ev in expanded:
        if ev['kind'] == 'set_tempo':
            now = ev['time']
            dt_sec = now - last_meta_time
            dt_ticks = 0 if last_tempo_us is None else round(dt_sec * ticks_per_beat * 1e6 / last_tempo_us)
            msg = MetaMessage('set_tempo', tempo=ev['tempo'], time=dt_ticks)
            meta_track.append(msg)
            last_meta_time = now
            last_tempo_us  = ev['tempo']
        else:
            now = ev['time']
            tempo_us = last_tempo_us or 500_000
            dt_sec   = now - last_perf_time
            dt_ticks = round(dt_sec * ticks_per_beat * 1e6 / tempo_us)

            kind = ev['kind']
            if kind == 'note_on':
                msg = Message('note_on',
                              note=ev['pitch'],
                              velocity=ev['velocity'],
                              time=dt_ticks,
                              channel=ev['channel'])
            elif kind == 'note_off':
                msg = Message('note_off',
                              note=ev['pitch'],
                              velocity=ev['velocity'],
                              time=dt_ticks,
                              channel=ev['channel'])
            elif kind == 'control_change':
                msg = Message('control_change',
                              control=ev['controller'],
                              value=ev['value'],
                              time=dt_ticks,
                              channel=ev['channel'])
            elif kind == 'program_change':
                msg = Message('program_change',
                              program=ev['program'],
                              time=dt_ticks,
                              channel=ev['channel'])
            else:
                continue

            perf_track.append(msg)
            last_perf_time = now

    # 5) End‐of‐track markers
    meta_track.append(MetaMessage('end_of_track', time=0))
    perf_track.append(MetaMessage('end_of_track', time=0))

    # 6) Save file
    mid.save(output_path)
    print(f"Reconstructed MIDI saved to: {output_path}")
    return mid

# utils.py


def _events_to_features(events, dataset):
    """
    Given a list of event‐dicts, produce a single‐item batch in the
    exact same format that your Dataset/DataLoader collate_onehot emits.
    """
    last_t = 0.0
    type_ids, pitch_ids, ctrl_ids = [], [], []
    dts, durs, vals, bpms = [], [], [], []

    for e in events:
        # 1) time‐delta
        dt = e["time"] - last_t
        last_t = e["time"]
        if dataset.time_bin:
            dt = round(dt / dataset.time_bin) * dataset.time_bin
        dts.append(dt)

        # 2) categorical slots
        t_id = dataset.type2id[e["type"]]
        type_ids.append(t_id)

        if e["type"] == "note":
            pitch_ids.append(dataset.pitch2id[e["pitch"]])
            ctrl_ids.append(dataset.ctrl2id[None])
            durs.append(e["end_time"] - e["time"])
            vals.append(e["velocity"])
            bpms.append(0.0)

        elif e["type"] == "control_change":
            pitch_ids.append(dataset.pitch2id[None])
            # map CC‐number to index
            ctrl_ids.append(dataset.ctrl2id[e["controller"]])
            durs.append(0.0)
            vals.append(e["value"])
            bpms.append(0.0)

        else:  # set_tempo
            pitch_ids.append(dataset.pitch2id[None])
            ctrl_ids.append(dataset.ctrl2id[None])
            durs.append(0.0)
            vals.append(0.0)
            bpms.append(e["bpm"])

    # one‐hot encode categories
    type_oh  = F.one_hot(torch.tensor(type_ids),  num_classes=dataset.num_types).float()
    pitch_oh = F.one_hot(torch.tensor(pitch_ids), num_classes=dataset.num_pitches).float()
    ctrl_oh  = F.one_hot(torch.tensor(ctrl_ids),  num_classes=dataset.num_ctrls).float()

    # wrap as a batch of size=1
    batch = {
        "type_oh":  type_oh.unsqueeze(0),   # (1, L, C_type)
        "pitch_oh": pitch_oh.unsqueeze(0),  # (1, L, C_pitch)
        "ctrl_oh":  ctrl_oh.unsqueeze(0),   # (1, L, C_ctrl)
        "dt":       torch.tensor(dts).unsqueeze(0),   # (1, L)
        "dur":      torch.tensor(durs).unsqueeze(0),  # (1, L)
        "val":      torch.tensor(vals).unsqueeze(0),  # (1, L)
        "bpm":      torch.tensor(bpms).unsqueeze(0),  # (1, L)
        "lengths":  torch.tensor([len(dts)], dtype=torch.long)
    }
    return batch

def _sample_next_event(outputs, dataset, last_time):
    """
    Given model outputs (dict of tensors) and the dataset vocabs,
    pick the next event by argmax (you could also sample).
    """
    b = 0
    t = outputs["type"].shape[1] - 1  # last time‐step idx

    # categorical
    type_id  = outputs["type"][b,t].argmax().item()
    pitch_id = outputs["pitch"][b,t].argmax().item()
    ctrl_id  = outputs["ctrl"][b,t].argmax().item()

    # continuous
    dt  = outputs["dt"][b,t].item()
    dur = outputs["dur"][b,t].item()
    val = outputs["val"][b,t].item()
    bpm = outputs["bpm"][b,t].item()

    next_time = last_time + dt

    # reverse‐map IDs → strings/numbers
    # build reverse dicts once
    id2type   = {v:k for k,v in dataset.type2id.items()}
    id2pitch  = {v:k for k,v in dataset.pitch2id.items()}
    id2ctrl   = {v:k for k,v in dataset.ctrl2id.items()}

    ev_type = id2type[type_id]
    if ev_type == "note":
        return {
            "type":     "note",
            "time":     next_time,
            "end_time": next_time + dur,
            "track":    0,
            "duration": dur,
            "channel":  0,
            "pitch":    id2pitch[pitch_id],
            "velocity": int(round(val))
        }
    elif ev_type == "control_change":
        return {
            "type":       "control_change",
            "time":        next_time,
            "track":       0,
            "controller": id2ctrl[ctrl_id],
            "value":       val
        }
    else:  # set_tempo
        return {
            "type":  "set_tempo",
            "time":   next_time,
            "track":  0,
            "bpm":    bpm
        }

def collate_onehot(batch):
    """
    Pads each field in the batch to shape (B, T_max, C) or (B, T_max)
    """
    B = len(batch)
    Ls = [b["dt"].shape[0] for b in batch]
    T = max(Ls)
    out = {}

    for key, tensor in batch[0].items():
        if key == "ticks_per_beat":
            # scalar: just return the first one
            out[key] = tensor
            continue
        if tensor.dim() == 2:
            # categorical one-hot: (L, C) -> (B, T, C)
            _, C = tensor.shape
            pad = torch.zeros(B, T, C, dtype=tensor.dtype)
            for i, b in enumerate(batch):
                L = b[key].shape[0]
                pad[i, :L] = b[key]
            out[key] = pad
        else:
            # continuous: (L,) -> (B, T)
            pad = torch.zeros(B, T, dtype=tensor.dtype)
            for i, b in enumerate(batch):
                L = b[key].shape[0]
                pad[i, :L] = b[key]
            out[key] = pad

    out["lengths"] = torch.tensor(Ls, dtype=torch.long)
    return out

def create_song(model, dataset, output_path, max_steps=500, device=None):
    """
    Roll out `model` autoregressively to produce up to `max_steps` events,
    then write a MIDI file to `output_path`.

    Args:
      model:      your trained LSTM/Transformer (in eval() mode)
      dataset:    the MaestroFeatureOneHotDataset instance (for vocabs & time_bin)
      output_path:str path where the .mid will be saved
      max_steps:  how many new events to generate
      device:     torch device (if None, uses CPU)
    """
    device = device or torch.device("cpu")
    model.eval()

    # 1) seed with ticks_per_beat from the first file in the manifest
    row = dataset.df.iloc[0]
    pm  = pretty_midi.PrettyMIDI(os.path.join(dataset.midi_dir, row["midi_filename"]))
    tpb = pm.resolution if hasattr(pm, "resolution") else pm.ticks_per_beat

    events = [{"type":"ticks_per_beat","ticks_per_beat":tpb}]
    last_time = 0.0

    # 2) autoregressive loop
    for _ in range(max_steps):
        batch = _events_to_features(events, dataset)
        batch = collate_onehot([batch])  # reuse your collate
        # move to device
        for k in batch: batch[k] = batch[k].to(device)

        # build model input exactly as in Train.py
        lengths = batch["lengths"] - 1
        x_in = torch.cat([
            batch["type_oh"] [:,:-1],
            batch["pitch_oh"][:,:-1],
            batch["ctrl_oh"] [:,:-1],
            batch["dt"]     [:,:-1].unsqueeze(-1),
            batch["dur"]    [:,:-1].unsqueeze(-1),
            batch["val"]    [:,:-1].unsqueeze(-1),
            batch["bpm"]    [:,:-1].unsqueeze(-1),
        ], dim=2)

        with torch.no_grad():
            out = model(x_in, lengths)

        # sample next event
        ev = _sample_next_event(out, dataset, last_time)
        events.append(ev)
        last_time = ev["time"]

    # 3) write MIDI
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    events_to_midi(events, output_path=output_path)


#%%