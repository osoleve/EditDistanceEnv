import re
import functools as ft
from typing import Callable, List, Dict, Optional

import verifiers as vf
from datasets import Dataset
from jellyfish import damerau_levenshtein_distance
from nltk.tokenize.sonority_sequencing import SyllableTokenizer
from nltk.tokenize import wordpunct_tokenize

from edenv.edenv import EditDistanceEnvironment

SYLLABIFY_PROMPT = '''Break the words in the following text into syllables separated by "{sep}", leaving all whitespace, punctuation, and any other textual structure intact.\n\nTEXT: "{text}"'''


@ft.cache
def get_syllable_tokenizer():
    return SyllableTokenizer()


def make_syllable_sample(text: str, sep: str = ".") -> Dict:
    """Create a syllabified sample using sonority sequencing"""
    tok = get_syllable_tokenizer()
    words = wordpunct_tokenize(text)
    sample = {
        "text": text,
        "sep": sep,
        "expected": text,
    }
    for word in words:
        syllables = tok.tokenize(word)
        sample["expected"] = re.sub(
            r"\b" + re.escape(word) + r"\b", sep.join(syllables), sample["expected"]
        )
    return sample


def build_syllable_sample_set() -> List[Dict]:
    """Create a default sample set for testing"""
    return [
        make_syllable_sample("coquettish, flamboyant and churlish", "-"),
        make_syllable_sample("the quick brown fox jumps over the lazy dog", "."),
        make_syllable_sample("the Rain in spain Stays mainly in the plain", "<SEP>"),
        make_syllable_sample("gregor samsa awoke one morning from uneasy dreams", "🪲"),
    ]


# Create syllabification environment
syllabify_env = EditDistanceEnvironment(
    task_prompt=SYLLABIFY_PROMPT,
    make_sample_func=make_syllable_sample,
    task_name="syllabify",
    default_samples=build_syllable_sample_set(),
)

load_environment = syllabify_env.load_environment
