""" Download and open SICK, then reinsert the Dutch translation using the
parallel sentences."""
import os

sickLocalFN = '/Users/gijswijnholds/Code/compdisteval-private/experiment_data/SICK/SICK.txt'
sick_sentences_origin_fn = '/Users/gijswijnholds/Documents/Code/sick_nl/SICKSENTS.txt'
sick_sentences_translation_fn = '/Users/gijswijnholds/Documents/Code/sick_nl/SICKSENTSNL_CORRECTED.txt'


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


sick_data = load_sick(sickLocalFN)

sick_translation_dict = load_parallel_sentences(sick_sentences_origin_fn,
                                                sick_sentences_translation_fn)
