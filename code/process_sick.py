""" Download and open SICK, then reinsert the Dutch translation using the
parallel sentences."""
import os

sickLocalFN = '/Users/gijswijnholds/Code/compdisteval-private/experiment_data/SICK/SICK.txt'
sick_sentences_origin_fn = '/Users/gijswijnholds/Documents/Code/sick_nl/SICKSENTS.txt'
sick_sentences_translation_fn = '/Users/gijswijnholds/Documents/Code/sick_nl/SICKSENTSNL_CORRECTED.txt'
sick_translation_out_fn = '/Users/gijswijnholds/Documents/Code/sick_nl/SICK_NL.txt'

def load_sick(fn):
    with open(fn, 'r') as in_file:
        lines = in_file.readlines()
    return [ln.strip().split('\t') for ln in lines]


def load_parallel_sentences(fn_origin, fn_translation):
    with open(fn_origin, 'r') as origin_file:
        origin_lines = [ln.strip() for ln in origin_file.readlines()]
    with open(fn_translation, 'r') as translation_file:
        translation_lines = [ln.strip() for ln in translation_file.readlines()]
    return dict(zip(origin_lines, translation_lines))


def replace_sick_data(origin_data, translation_dict):
    header = '\t'.join(origin_data[0]) + '\n'
    new_data = '\n'.join(['\t'.join([ln[0], translation_dict[ln[1].strip()],
                                     translation_dict[ln[2].strip()]] + ln[3:7]
                                     + ln[1:3] + ln[9:12])
                                     for ln in origin_data[1:]])
    return header + new_data

print("Processing SICK translation...")
sick_data = load_sick(sickLocalFN)
sick_translation_dict = load_parallel_sentences(sick_sentences_origin_fn,
                                                sick_sentences_translation_fn)
dutch_sick = replace_sick_data(sick_data, sick_translation_dict)

if os.path.exists(sick_translation_out_fn):
    os.remove(sick_translation_out_fn)

with open(sick_translation_out_fn, 'w') as out_file:
    out_file.write(dutch_sick)
    
print("All done, translated dataset written to {}".format(sick_translation_out_fn))
