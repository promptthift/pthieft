import re
from typing import List
from collections import Counter
from swift.plugin import ORM, orms
from swift.utils import get_logger

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import clip
import torch

logger = get_logger()

device = "cuda" if torch.cuda.is_available() else "cpu"

if True:
    clip_model, preprocess = clip.load("ViT-L/14", device=device, download_root="./.cache")
else:
    clip_model = None
    preprocess = None


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards
def jaccard_score(sentence1, sentence2):
    """
    Calculate the Jaccard score between two sentences based on their words.
    
    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.
        
    Returns:
        float: The Jaccard score between the two sentences.
    """
    sentence1 = re.sub(r'[^\w\s]', '', sentence1.lower())
    sentence2 = re.sub(r'[^\w\s]', '', sentence2.lower())
    # Tokenize sentences into words
    words1 = set(sentence1.lower().split())
    words2 = set(sentence2.lower().split())
    
    # Calculate the intersection and union
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    # Calculate the Jaccard score
    score = len(intersection) / len(union) if len(union) > 0 else 0.0
    # del sentence1, sentence2, words1, words2, intersection, union
    # gc.collect()  
    return score

def jaccard_score_multiset(sentence1, sentence2):
    """
    Calculate the Jaccard score between two sentences based on their words.
    
    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.
        
    Returns:
        float: The Jaccard score between the two sentences.
    """
    sentence1 = re.sub(r'[^\w\s]', '', sentence1.lower())
    sentence2 = re.sub(r'[^\w\s]', '', sentence2.lower())
    # Tokenize sentences into words
    
    words1 = Counter(sentence1.split())
    words2 = Counter(sentence2.split())
    
    # Calculate the intersection and union
    intersection = sum((words1 & words2).values())
    union = sum((words1 | words2).values()) 
    
    # del sentence1, sentence2, words1, words2, intersection, union
    # gc.collect()  
    return intersection / union if union > 0 else 0.0

def calculate_weighted_score(content, token_contributions):
    """
    Calculate the weighted score of a content based on token contributions.

    Args:
        content (str): The sentence to evaluate.
        token_contributions (list of dict): List of token contribution dictionaries.

    Returns:
        float: The weighted score.
    """
    content_lower = content.lower()  # Case-insensitive search
    total_score = 0

    for item in token_contributions:
        phrase = item["phrase"].lower()
        contribution_value = item["contribution_values"]
        value_sum = item["value_sum"]

        # If the phrase is in the content, add contribution_value / value_sum
        if phrase in content_lower:
            total_score += contribution_value / value_sum
    return total_score



class TokenContributionRewardFunction(ORM):
    def __call__(self,completions,token_contribution,real,**kwargs):
        rewards = [0.5*jaccard_score(content,conver) + 0.5*calculate_weighted_score(content,contri) for content,contri,conver in zip(completions,token_contribution,real)]
        return rewards

class CLIPRewardFunction(ORM):
    def __call__(self,completions,real,**kwargs):
        rewards = []
        for content,conver in zip(completions,real):
        
            with torch.no_grad():
                text_input1 = clip.tokenize([content], context_length=77, truncate=True).to(device)
                text_input2 = clip.tokenize([conver], context_length=77, truncate=True).to(device)
                text_features1 = clip_model.encode_text(text_input1)
                text_features2 = clip_model.encode_text(text_input2)
                clip_similarity = torch.cosine_similarity(text_features1, text_features2).item()
                reward = 0.5*clip_similarity + 0.5*jaccard_score(content,conver)
                reward = 2*reward - 1
            rewards.append(reward)
        return rewards
    
class JaccardRewardFunction(ORM):
    def __call__(self,completions,real,**kwargs):
        rewards = [jaccard_score(content,conver) for content,conver in zip(completions,real)]
        return rewards

class JaccardRewardMultiFunction(ORM):
    def __call__(self,completions,real,**kwargs):
        rewards = [jaccard_score_multiset(content,conver) for content,conver in zip(completions,real)]
        return rewards

class BLEURewardFunction:
    def __call__(self, completions, real, **kwargs):
        n_gram=4
        smoothing = SmoothingFunction().method1
        weights = tuple([1.0 / n_gram] * n_gram)  
        rewards = [
            sentence_bleu([r.split()], c.split(), weights=weights, smoothing_function=smoothing)
            for c, r in zip(completions, real)
        ]
        return rewards

class JacBLEUCONRewardFunction(ORM):
    def __call__(self,completions,token_contribution,real, **kwargs):
        n_gram = 4
        smoothing = SmoothingFunction().method1
        weights = tuple([1.0 / n_gram] * n_gram)
        rewards = [0.4*jaccard_score_multiset(content,conver) + 0.4*sentence_bleu([conver.split()], content.split(), weights=weights, smoothing_function=smoothing) +0.2*calculate_weighted_score(content,contri) for content,contri,conver in zip(completions,token_contribution,real)]
        return rewards

class JacCONRewardFunction(ORM):
    def __call__(self,completions,token_contribution,real, **kwargs):
        rewards = [0.5*jaccard_score_multiset(content,conver) +0.5*calculate_weighted_score(content,contri) for content,contri,conver in zip(completions,token_contribution,real)]
        return rewards

class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['token_contribution']= TokenContributionRewardFunction
orms['JaccardReward']= JaccardRewardFunction
orms['JacBLEUContri'] = JacBLEUCONRewardFunction
orms['JacContri']= JacCONRewardFunction
orms['BLEU'] = BLEURewardFunction
orms['CLIP'] = CLIPRewardFunction
