import re
import functools as ft
from typing import Callable, List, Dict, Optional

import verifiers as vf
from datasets import Dataset
from jellyfish import damerau_levenshtein_distance

from nltk.tokenize.sonority_sequencing import SyllableTokenizer
from nltk.tokenize import wordpunct_tokenize

def symdl(s: str, t: str) -> float:
    """Symmetric damerau levenshtein distance as a similarity metric"""
    return 1.0 - damerau_levenshtein_distance(s, t) / max(len(s), len(t))


class EditDistanceEnvironment:
    """
    A generic environment for tasks that involve string transformations
    evaluated using Damerau-Levenshtein distance as a similarity metric.
    """

    def __init__(
        self,
        task_prompt: str,
        make_sample_func: Callable[[str, str], Dict],
        task_name: str = "transform",
        default_samples: Optional[List[Dict]] = None,
        hf_dataset_id: str = None,
        column: str = None
    ):
        """
        Args:
            task_prompt: Template string with {sep} and {text} placeholders
            make_sample_func: Function that takes (text, sep) and returns
                             {"text": str, "sep": str, "expected": str}
            task_name: Name for the task in the dataset
            default_samples: Optional list of default samples to use
        """
        self.task_prompt = task_prompt
        self.make_sample = make_sample_func
        self.task_name = task_name
        self.default_samples = default_samples or []
        self.hf_dataset_id = hf_dataset_id
        self.column = column

    def create_sample_set(self) -> List[Dict]:
        """Create default sample set - can be overridden"""
        if self.default_samples:
            return self.default_samples

        # Default samples for testing
        samples = []
        samples.append(self.make_sample("coquettish, flamboyant and churlish", "-"))
        samples.append(
            self.make_sample("the quick brown fox jumps over the lazy dog", ".")
        )
        samples.append(
            self.make_sample("the Rain in spain Stays mainly in the plain", "<SEP>")
        )
        samples.append(
            self.make_sample("gregor samsa awoke one morning from uneasy dreams", "🪲")
        )
        return samples

    def build_data(self, samples: List[Dict]) -> Dataset:
        """Format samples for training"""
        data = {"prompt": [], "answer": [], "task": []}
        for sample in samples:
            data["prompt"].append(
                [
                    {
                        "role": "user",
                        "content": self.task_prompt.format(**sample),
                    }
                ]
            )
            data["answer"].append(sample["expected"])
            data["task"].append(self.task_name)
        return Dataset.from_dict(data)

    @staticmethod
    def indel_accuracy(prompt: str, completion, answer: str) -> float:
        """Score accuracy using symmetric Damerau-Levenshtein distance"""
        # Extract the actual text from the completion structure
        if isinstance(completion, list) and len(completion) > 0:
            if isinstance(completion[0], dict) and "content" in completion[0]:
                completion_text = completion[0]["content"]
            else:
                completion_text = str(completion[0])
        elif isinstance(completion, str):
            completion_text = completion
        else:
            completion_text = str(completion)

        return symdl(completion_text, answer)

    def load_environment(
        self,
        hf_dataset_id: Optional[str] = None,
        column: Optional[str] = None,
        **kwargs,
    ) -> vf.SingleTurnEnv:
        """Load the verifier environment"""
        if (hf_dataset_id is not None and column is not None) or (self.hf_dataset_id is not None and self.column is not None):
            if column is not None:
                self.hf_dataset_id = hf_dataset_id
                self.column = column
                
            from datasets import load_dataset

            hf_dataset = load_dataset(self.hf_dataset_id)
            split_name = list(hf_dataset.keys())[0]
            hf_data = hf_dataset[split_name]
            dataset = self.build_data(
                [self.make_sample(item[self.column], **kwargs) for item in hf_data]
            )
        else:
            dataset = self.build_data(self.create_sample_set())

        rubric = vf.Rubric(funcs=[self.indel_accuracy], weights=[1.0])
        vf_env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
        return vf_env

# ---------------------- Env setup below

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
