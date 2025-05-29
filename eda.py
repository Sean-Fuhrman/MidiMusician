#%%
import pandas as pd
import mido
import pretty_midi
# 1. Load manifest and pick first file
df       = pd.read_csv('maestro-v3.0.0/maestro-v3.0.0.csv')
pre_path = 'maestro-v3.0.0/'
file_path = pre_path + df['midi_filename'][125]

# 2. Parse with pretty_midi
midi = pretty_midi.PrettyMIDI(file_path)
ticks_per_beat = midi.resolution

print(f'Ticks per beat: {ticks_per_beat}')

#print out all the ticks per beat
for i in range(10):
    file_path = pre_path + df['midi_filename'][i]
    midi = pretty_midi.PrettyMIDI(file_path)
    ticks_per_beat = midi.resolution
    print(f'Ticks per beat for {df["midi_filename"][i]}: {ticks_per_beat}')
import utils
#Save the original midi file
midi.write('original.mid')

# 3. Convert to events

events = utils.midi_to_event_list(midi)
print(events[0])
# 4. Convert back to midi
reconstructed_midi = utils.events_to_midi(events)
# 5. Save reconstructed midi
reconstructed_midi.save('reconstructed.mid')

# %%
