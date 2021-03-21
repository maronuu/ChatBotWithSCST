import logging
import os

from . import utils

### PATHS
DATA_DIR = "data/cornell"
MOVIE_METADATA_NAME = 'movie_titles_metadata.txt'
# header: movie id, title, year, rate, vote, genre list
MOVIE_LINES_NAME = 'movie_lines.txt'
# header: line id, char id, movie id,  char name, phrase
MOVIE_CONVERSATIONS_NAME = 'movie_conversations.txt'
# header: char id (first), char id (second), movie id, list of line id

### CONSTATNTS
SEP_STRING = "+++$+++"

### Logger
log = logging.getLogger("cornell_movies_corpus_handler")


def helper_iterator(data_dir, file_name):
    with open(os.path.join(data_dir, file_name, 'rb')) as f:
        for line in f:
            line = str(line, encoding='utf-8', errors='ignore')
            line = [s.strip() for s in line.split(SEP_STRING)]
            yield line

def load_movie_id_set(data_dir, genre_filter):
    id_set = set()
    for line in helper_iterator(data_dir, MOVIE_METADATA_NAME):
        movie_id = line[0]
        movie_genre_list = line[5]
        if genre_filter in movie_genre_list:
            id_set.add(movie_id)

    return id_set

def load_phrases(data_dir, movie_id_set=None):
    lineId2phrase = dict()
    for line in helper_iterator(data_dir, MOVIE_LINES_NAME):
        line_id = line[0]
        movie_id = line[2]
        phrase = line[4]
        if movie_id_set and (movie_id not in movie_id_set):
            # should be ignroed.
            continue
        tokens = utils.tokenize(phrase)
        if not tokens:
            continue
        lineId2phrase[line_id] = tokens
    
    return lineId2phrase

def load_conversations(data_dir, lineId2phrase, movie_id_set=None):
    conv_list = list()
    for line in helper_iterator(data_dir, MOVIE_CONVERSATIONS_NAME):
        movie_id = line[2]
        line_id_list = line[3]
        if movie_id_set and (movie_id not in movie_id_set):
            # should be ignored.
            continue
        line_id_list = line_id_list.strip('[]').split(', ')
        line_id_list = [s.strip("'") for s in line_id_list]
        dialogue = [lineId2phrase[line_id] for line_id in line_id_list if line_id in lineId2phrase]
        if not dialogue:
            continue
        conv_list.append(dialogue)

    return conv_list

def load_genres(data_dir):
    movie2genres = dict()
    for line in helper_iterator(data_dir, MOVIE_METADATA_NAME):
        movie_id = line[0]
        movie_genre_list = line[5].strip('[]').split(', ')
        movie_genre_list = [s.strip("'") for s in movie_genre_list]
        movie2genres[movie_id] = movie_genre_list
    
    return movie2genres

def load_dialogues(data_dir=DATA_DIR, genre_filter=''):
    """Load dialogue data from corpus. Include only dialogues included specified genres.

    Args:
        data_dir (str, optional): path to dataset directory. Defaults to DATA_DIR.
        genre_filter (str, optional): Genres to be included in dataset. Defaults to ''.
    """
    movie_id_set = None
    if genre_filter:
        movie_id_set = load_movie_id_set(data_dir, genre_filter)
        log.info(f"{len(movie_id_set)} movies loaded. Genre: {genre_filter}")
    
    log.info("Loading and tokenizing phrases...")
    lineId2phrase = load_phrases(data_dir, movie_id_set)
    log.info(f"{len(lineId2phrase)} phrases loaded.")
    dialogues = load_conversations(data_dir, lineId2phrase, movie_id_set)

    return dialogues