import miditoolkit
import numpy as np
import copy

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 4
DEFAULT_DURATION_BINS = np.arange(27.5, 1761, 27.5)

# parameters for output
DEFAULT_RESOLUTION = 220

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, value):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.value = value

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, value={})'.format(
            self.name, self.start, self.end, self.velocity, self.value)

# read midi
def read_midi(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(Item(
            name='Note',
            start=note.start, 
            end=note.end, 
            velocity=note.velocity, 
            value=note.pitch))
    note_items.sort(key=lambda x: x.start)
    # estimated beat
    beat_items = []
    for m in midi_obj.markers:
        beat_items.append(Item(
            name='Beat',
            start=m.time,
            end=None,
            velocity=None,
            value=None
        ))
    return note_items, beat_items, midi_obj.ticks_per_beat

# quantize items
def quantize_items(items, resolution):
    # grid
    grids = np.arange(0, items[-1].start, resolution, dtype=int)
    # process
    output = []
    for i in range(len(items)):
        temp = copy.deepcopy(items[i])
        index = np.argmin(abs(grids-temp.start))
        shift = grids[index] - temp.start
        temp.start += shift
        if temp.end:
            temp.end += shift
        output.append(temp)
    return output

# group items
def group_items(items):
    beat_idx = [i for i, item in enumerate(items) if item.name == 'Beat']
    groups = []
    for i in range(len(beat_idx)-1):
        groups.append(items[beat_idx[i]:beat_idx[i+1]+1])
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

# item to event
def item2event(groups):
    events = []
    n_beat = 0
    for group in groups:
        n_beat += 1
        if len(group) == 2: # no any notes
            continue
        else:
            # beat event
            beat_st, beat_et = group[0].start, group[-1].start
            events.append(Event(
                name='Beat',
                time=None, 
                value=None,
                text='{}'.format(n_beat)))
            # note event
            for item in group[1:-1]:
                # position
                flags = np.linspace(beat_st, beat_et, DEFAULT_FRACTION, endpoint=False)
                index = np.argmin(abs(flags-item.start))
                events.append(Event(
                    name='Position', 
                    time=item.start,
                    value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                    text='{}'.format(item.start)))
                # velocity
                velocity_index = np.searchsorted(
                    DEFAULT_VELOCITY_BINS, 
                    item.velocity, 
                    side='right') - 1
                events.append(Event(
                    name='Note Velocity',
                    time=item.start, 
                    value=velocity_index,
                    text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # pitch
                events.append(Event(
                    name='Note On',
                    time=item.start, 
                    value=item.value,
                    text='{}'.format(item.value)))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
    return events

# convert word to event
def word_to_event(words, word2event):
    events = []
    for word in words:
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events

# write midi
def write_midi(words, bpm, word2event, output_path):
    events = word_to_event(words, word2event)
    print(*events, sep='\n')
    # get beat and note (no time)
    temp_notes = []
    for i in range(len(events)-3):
        if events[i].name == 'Beat':
            temp_notes.append('Beat')
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Note Velocity' and \
            events[i+2].name == 'Note On' and \
            events[i+3].name == 'Note Duration':
            # start time and end time from position
            position = int(events[i].value.split('/')[0]) - 1
            # velocity
            index = int(events[i+1].value)
            velocity = int(DEFAULT_VELOCITY_BINS[index])
            # pitch
            pitch = int(events[i+2].value)
            # duration
            index = int(events[i+3].value)
            duration = DEFAULT_DURATION_BINS[index]
            # adding
            temp_notes.append([position, velocity, pitch, duration])
    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    notes = []
    current_beat = 0
    for note in temp_notes:
        if note == 'Beat':
            current_beat += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_beat_st = current_beat * ticks_per_beat
            current_beat_et = (current_beat + 1) * ticks_per_beat
            flags = np.linspace(current_beat_st, current_beat_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, int(st), int(et)))
    # write
    midi = miditoolkit.midi.parser.MidiFile()
    midi.ticks_per_beat = DEFAULT_RESOLUTION
    # write instrument
    inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
    inst.notes = notes
    midi.instruments.append(inst)
    # write tempo
    midi.tempo_changes = [miditoolkit.midi.containers.TempoChange(bpm, 0)]
    # write
    midi.dump(output_path)