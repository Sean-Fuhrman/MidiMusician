#%%
import utils
import pandas as pd
import mido
import pretty_midi
maestro_csv = "maestro-v3.0.0/maestro-v3.0.0.csv"
df = pd.read_csv(maestro_csv)

#convert first few rows to event list
for i in range(18):
    print(f"Processing row {i}: {df.iloc[i]['midi_filename']}")
    midi = pretty_midi.PrettyMIDI("maestro-v3.0.0/" + df.iloc[i]["midi_filename"])
    events = utils.midi_to_event_list(midi)
    
    types = [e["type"] for e in events]
    #get count of types
    unique_types = set(types)
    print(f"Unique types: {unique_types}")
    for t in unique_types:
        print(f"{t}: {types.count(t)}")
        
    
        


# %%
import pretty_midi
midi = pretty_midi.PrettyMIDI("generated_song_50.mid")
#print out all notes
for instrument in midi.instruments:
    for note in instrument.notes:
        print(f"Note: {note.pitch}, Start: {note.start}, End: {note.end}, Velocity: {note.velocity}")
    for control in instrument.control_changes:
        print(f"Control: {control.number}, Time: {control.time}, Value: {control.value}")
# %%
