#%%
import pandas as pd
import pretty_midi
import os
import torch
import torch.nn.functional as F
import pretty_midi
from mido import MidiFile, MidiTrack, Message, MetaMessage
from tqdm import tqdm

def midi_to_event_list(midi_data):
    """
    Convert a PrettyMIDI object into a list of events conforming to:
    
      "note":           ["start_time","end_time","track","duration","channel","pitch","velocity"]
      "control_change": ["time","track", "value"]
      "start-token":      ["track","bpm", "resolution"]
      "end-token":    []
    """
    events = []
    track_idx = 0  # MAESTRO is single-instrument, so one “track”

    inst = midi_data.instruments[0]
    
    
    # 3) control changes (pedals etc.)
    for cc in inst.control_changes:
        events.append({
            "type":       "control_change",
            "time":      cc.time,
            "controller": cc.number,  # MIDI controller number
            "value":      cc.value
        })
    
    # 4) notes (automatically paired start/end by pretty_midi)
    for note in inst.notes:
        events.append({
            "type":     "note",
            "time":    note.start,
            "duration":    note.end - note.start,
            "pitch":    note.pitch,
            "velocity": note.velocity
        })
    
    # Sort events by time
    events.sort(key=lambda x: x["time"])
    
     # 5) start-token with tempo info
    times, tempi = midi_data.get_tempo_changes()  # times in sec, tempi in µs per beat
    for t, uspb in zip(times, tempi):
        bpm = 60_000_000 / uspb
        events.insert(0, {  # insert at the start
            "type":  "start-token",
            "bpm":   bpm,
            "resolution": midi_data.resolution  # ticks per beat
        })
        break  # only one start-token per file
    
    # 6) end-of-track marker
    events.append({
        "type": "end-token",
        "time": midi_data.get_end_time(),  # end time of the track
    })
    
    return events

import pretty_midi

def events_to_midi(events, 
                          output_path='reconstructed.mid'):
    """
    Reconstruct a .mid from your event dicts using PrettyMIDI.

    Expects:
      - events[0] == {'type':'start-token','bpm':…, 'resolution':…}
      - all other events have 'time' in seconds.
    """
    if not events or events[0]['type'] != 'start-token':
        raise ValueError("events[0] must be the start-token with 'bpm' & 'resolution'.")

    # 1) Pull out header info
    start   = events[0]
    resolution = int(start.get('resolution', 480))
    bpm        = float(start.get('bpm', 120.0))

    # 2) Create PrettyMIDI, set resolution & initial tempo
    pm = pretty_midi.PrettyMIDI(resolution=resolution)
    pm.initial_tempo = bpm

    # 3) Build a single instrument (track 0)
    inst = pretty_midi.Instrument(program=0)  # you can choose a different program if desired

    # 4) Walk your events and append Notes / ControlChanges
    for e in events[1:]:
        t = float(e.get('time', 0.0))

        if e['type'] == 'note' and e.get('pitch') is not None:
            start_sec = t
            end_sec   = t + float(e['duration'])
            vel       = int(round(float(e['velocity']) * 127))
            vel       = max(1, min(127, vel))  # avoid zero-velocity notes
            note = pretty_midi.Note(
                velocity=vel,
                pitch=int(e['pitch']),
                start=start_sec,
                end=end_sec
            )
            inst.notes.append(note)

        elif e['type'] == 'control_change':
            cc_time = t
            cc_num  = int(e.get('controller', 64))
            cc_val  = int(round(float(e['value']) * 127))
            cc_val  = max(0, min(127, cc_val))
            ctrl = pretty_midi.ControlChange(
                number=cc_num,
                value=cc_val,
                time=cc_time
            )
            inst.control_changes.append(ctrl)

        # skip any other types (e.g. end-token)

    # 5) Append instrument and write file
    pm.instruments.append(inst)
    pm.write(output_path)
    print(f"Wrote MIDI → {output_path}")
    return pm

def _events_to_features(events, dataset):
    """
    Given a list of event-dicts, produce a single‐item batch in the
    exact same format that your Dataset/DataLoader collate_onehot emits.
    """
    last_t = 0.0
    # temporary lists
    type_ids, pitch_ids, ctrl_ids = [], [], []
    
    # delta times, durations, values
    # values/durs are used for notes and control changes
    # are are bpm / ticks per beat for start token
    dts, durs, vals       = [], [], []

    for e in events:
        if len(type_ids) >= dataset.max_seq_len:
            break
        # time delta
        if "time" in e:
            dt = e["time"] - last_t
            last_t = e["time"]
            if dataset.time_bin:
                dt = round(dt / dataset.time_bin) * dataset.time_bin
            dts.append(dt)
        else:
            dt = 0.0
            dts.append(dt)

        # categorical
        t_id = dataset.type2id[e["type"]]
        type_ids.append(t_id)
        
        assert t_id != "start-token", "Start token should not be in the event list from features."

        
        if e["type"] == "note":
            pitch_ids.append(dataset.pitch2id[e["pitch"]])
            ctrl_ids.append(dataset.ctrl2id[None])
            durs.append(e["duration"])
            vals.append(int(e["velocity"]) / 127.0)  # normalize velocity to [0, 1]
        elif e["type"] == "control_change":
            pitch_ids.append(dataset.pitch2id[None])
            ctrl_ids.append(dataset.ctrl2id[e["controller"]])  # <<— controller, not value
            durs.append(0.0)
            vals.append(int(e["value"]) / 127.0)                         # <<— store the pedal pressure separately
        elif e["type"] == "start-token":
            pitch_ids.append(dataset.pitch2id[None])
            ctrl_ids.append(dataset.ctrl2id[None])
            durs.append(e["bpm"])  # store the bpm for start token
            vals.append(e["resolution"])  # store the ticks per beat for start token
        elif e["type"] == "end-token":
            # end token does not have pitch or ctrl
            pitch_ids.append(dataset.pitch2id[None])
            ctrl_ids.append(dataset.ctrl2id[None])
            durs.append(0.0)  # no duration for end token
            vals.append(0.0)  # no value for end token
        else:
            raise ValueError(f"Unknown event type: {e['type']}")
        
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
    next_time = last_time + abs(dt)
    dur = abs(dur)  # duration should be non-negative

    # reverse‐map IDs → strings/numbers
    # build reverse dicts once
    id2type   = {v:k for k,v in dataset.type2id.items()}
    id2pitch  = {v:k for k,v in dataset.pitch2id.items()}
    id2ctrl   = {v:k for k,v in dataset.ctrl2id.items()}

    ev_type = id2type[type_id]
    if ev_type == "note":
        return {
            "type":      "note",
            "time":       next_time,
            "duration":   dur,
            "pitch":      id2pitch[pitch_id] if id2pitch[pitch_id] is not None else 60,  # default to middle C
            "velocity":   int(round(min(max(val, 0.0), 1.0) * 127))
        }
    elif ev_type == "control_change":
        return {
            "type":       "control_change",
            "time":        next_time,
            "controller": id2ctrl[ctrl_id] if id2ctrl[ctrl_id] is not None else 64,
            "value":  int(round(min(max(val, 0.0), 1.0) * 127))
        }
    elif ev_type == "start-token":
        return None
    elif ev_type == "end-token":
        return {
            "type":       "end-token",
            "time":        next_time
            }

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

    # 1) seed with first event from the first file in the manifest
    row = dataset.df.iloc[0]
    pm  = pretty_midi.PrettyMIDI(os.path.join(dataset.midi_dir, row["midi_filename"]))
    ticks_per_beat = pm.resolution
    events = midi_to_event_list(pm)
    if not events:
        raise ValueError("No events found in the MIDI file. Cannot generate song.")
    
    events = events[:1]  # start with the start token only
    last_time = 0.0

    # 2) autoregressive loop
    for _ in tqdm(range(max_steps), desc="Generating example"):
        batch = _events_to_features(events, dataset)
        # batch = collate_onehot([batch])  # reuse your collate
        # move to device
        for k in batch: batch[k] = batch[k].float().to(device)
        # build model input exactly as in Train.py
        lengths = batch["lengths"] - 1
        if lengths > 0:
            x_in = torch.cat([
                batch["type_oh"] [:,:-1],
                batch["pitch_oh"][:,:-1],
                batch["ctrl_oh"] [:,:-1],
                batch["dt"]     [:,:-1].unsqueeze(-1),
                batch["dur"]    [:,:-1].unsqueeze(-1),
                batch["val"]    [:,:-1].unsqueeze(-1),
            ], dim=2)
        else:
            x_in = torch.cat([
                batch["type_oh"] ,
                batch["pitch_oh"],
                batch["ctrl_oh"],
                batch["dt"].unsqueeze(-1),
                batch["dur"].unsqueeze(-1),
                batch["val"].unsqueeze(-1),
            ], dim=2)
            lengths = batch["lengths"]
        with torch.no_grad():
            out = model(x_in, lengths)

        # sample next event
        ev = _sample_next_event(out, dataset, last_time)
        if ev is not None:
            events.append(ev)
        
            if "time" in ev:
                # update last_time only if the event has a time field
                last_time = ev["time"]
            else:
                last_time += 0.0  # no time update for start/end tokens
        if ev and ev["type"] == "end-token":
            print("End token reached, stopping generation.")
            break

    # 3) write MIDI
    print(f"Generated {len(events)} events, writing to {output_path}")
    # For testing remove end-tokens
    events_to_midi(events, output_path=output_path)


##%%