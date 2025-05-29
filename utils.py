#%%
import pandas as pd
import pretty_midi

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

# def create_song(model)

#%%