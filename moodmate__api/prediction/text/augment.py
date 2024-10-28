import random
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')

# Custom functions
from prediction.text.evaluate import make_prediction
from prediction.text.preprocess import tokenize_sent
from prediction.text.shap_extraction import get_shap_values_as_list

class SSwap():
    """
    Saliency-based Token Swap (SSwap) class that performs token swapping between sentences
    based on their saliency values for data augmentation in text classification tasks.

    Methods
    -------
    swap_words(sent1, sent2, shap_1, shap_2, thresh, p):
        Swaps non-salient tokens between two sentences based on their SHAP values and a threshold.
    
    swap_imp_words(sent1, sent2, shap_1, shap_2, thresh, p):
        Swaps important (salient) tokens between two sentences based on their SHAP values and a threshold.
    
    bisent_swap(sent, rand_sent, shap, rand_shap, thresh, same_label=None, p=0.1):
        A high-level method that applies the appropriate token swapping strategy (either non-salient or
        salient token swapping) between two sentences depending on whether they share the same label.
    """
    def swap_words(self, sent1, sent2, shap_1, shap_2, thresh, p):
        """
        Swaps non-salient tokens between two sentences.

        Parameters
        ----------
        sent1 : str
            The first sentence.
        sent2 : str
            The second sentence.
        shap_1 : list of floats
            SHAP values for tokens in the first sentence.
        shap_2 : list of floats
            SHAP values for tokens in the second sentence.
        thresh : float
            Percentile threshold for determining which tokens are salient.
        p : float
            The proportion of non-salient tokens to swap between the sentences.

        Returns
        -------
        tuple of str
            The modified sentences after non-salient token swapping.
        """
        
        # STEP 1: Extract Saliency information
        
        # extract words from sentences
        words1 = tokenize_sent(sent1)
        words2 = tokenize_sent(sent2)
        # extract salient words
        important_words_1 = [w for i, w in enumerate(words1) if shap_1[i] >= np.percentile(shap_1, thresh)]
        important_words_2 = [w for i, w in enumerate(words2) if shap_2[i] >= np.percentile(shap_2, thresh)]
        # extract indices for non-salient words
        swappable_idx_1_ns = [idx for idx, w in enumerate(words1) if w not in important_words_1]
        swappable_idx_2_ns = [idx for idx, w in enumerate(words2) if w not in important_words_2]
        
        # STEP 2: Swap non-salient words
        
        # defines how much of the sentence to swap
        n = int(p * max(1, max(len(swappable_idx_1_ns), len(swappable_idx_2_ns))))
        for _ in range(n):
            if (len(swappable_idx_1_ns) > 0) and (len(swappable_idx_2_ns) > 0):
                swap_idx_1, swap_idx_2 = random.choice(swappable_idx_1_ns), random.choice(swappable_idx_2_ns)
                words1[swap_idx_1], words2[swap_idx_2] = words2[swap_idx_2], words1[swap_idx_1]
                
        # Return 2 augmented sentences
        return (' '.join(words1), ' '.join(words2))

    def swap_imp_words(self, sent1, sent2, shap_1, shap_2 ,thresh, p):
        """
        Swap important words between two sentences based on SHAP values.
        
        Args:
            sent1 (str): First input sentence.
            sent2 (str): Second input sentence.
            shap_1 (list): SHAP values for words in the first sentence.
            shap_2 (list): SHAP values for words in the second sentence.
            thresh (float): Percentile threshold for important words.
            p (float): Proportion of swaps to perform.
        
        Returns:
            tuple: Modified versions of the two sentences after word swaps.
        """
        
        # STEP 1: Extract Saliency information
        
        # extract words from sentences
        words1 = tokenize_sent(sent1)
        words2 = tokenize_sent(sent2)
        # extract salient words
        important_words_1 = [w for i, w in enumerate(words1) if shap_1[i] >= np.percentile(shap_1, thresh)]
        important_words_2 = [w for i, w in enumerate(words2) if shap_2[i] >= np.percentile(shap_2, thresh)]
        # extract indices for salient words
        swappable_idx_1 = [idx for idx, w in enumerate(words1) if w in important_words_1]
        swappable_idx_2 = [idx for idx, w in enumerate(words2) if w in important_words_2]
        # extract indices for non-salient words
        swappable_idx_1_ns = [idx for idx, w in enumerate(words1) if w not in important_words_1]
        swappable_idx_2_ns = [idx for idx, w in enumerate(words2) if w not in important_words_2]
        
        # STEP 2: Swap words based on saliency 
        
        # defines how much of the sentence to swap        
        n = int(p * max(1, min(len(words1), len(words2))))
        for _ in range(n):
            # Swap salient words
            if (len(swappable_idx_1) > 0) and (len(swappable_idx_2) > 0):
                swap_idx_1, swap_idx_2 = random.choice(swappable_idx_1), random.choice(swappable_idx_2)
                words1[swap_idx_1], words2[swap_idx_2] = words2[swap_idx_2], words1[swap_idx_1]
            # Swap non-salient words
            if (len(swappable_idx_1_ns) > 0) and (len(swappable_idx_2_ns) > 0):
                swap_idx_1, swap_idx_2 = random.choice(swappable_idx_1_ns), random.choice(swappable_idx_2_ns)
                words1[swap_idx_1], words2[swap_idx_2] = words2[swap_idx_2], words1[swap_idx_1]
                
        # Return 2 augmented sentences
        return (' '.join(words1), ' '.join(words2))

    def bisent_swap(self, sent, rand_sent, shap, rand_shap, thresh, same_label=None, p=.1):
        """
        Swap words between two sentences based on conditions and SHAP values.
        
        Args:
            sent (str): Main input sentence.
            rand_sent (str): Randomly selected sentence for swapping.
            shap (list): SHAP values for the main sentence.
            rand_shap (list): SHAP values for the random sentence.
            thresh (float): Percentile threshold for important words.
            same_label (bool, optional): Determines the swap method. Defaults to None.
            p (float, optional): Proportion of swaps to perform. Defaults to 0.1.
        
        Returns:
            list: Tokenized version of the modified sent (main).
        """
        
        # Inter-Class swapping - swap non-salient tokens only
        if not same_label:
            sent1, sent2 = self.swap_words(
                sent,
                rand_sent,
                shap,
                rand_shap, thresh, p
            )
        else: # Inter-Class swapping - swap non-salient tokens only
            sent1, sent2= self.swap_imp_words(
                sent,
                rand_sent,
                shap,
                rand_shap, thresh, p
            )
        
        # Return 1 augmented sentence - augmentation of sent
        return tokenize_sent(sent1)
    

def extract_saliency(t, p, lang, sentence1, sentence2, s1label, s2label, labels, sintam_labels):
    
    """
    Extract important words from two sentences based on SHAP values and language-specific models.
    
    Args:
        t (float): Percentile threshold for important words.
        p (float): Placeholder parameter.
        lang (str): Language of the sentences ('en', 'si', 'tm').
        sentence1 (str): First input sentence.
        sentence2 (str): Second input sentence.
        s1label (str): Label of the first sentence.
        s2label (str): Label of the second sentence.
        labels (list): List of possible labels in English.
        sintam_labels (list): List of possible labels in other languages.
    
    Returns:
        tuple: SHAP values and lists of important words from both sentences.
    """
    
    # Custom dataset-based mapping
    if lang=='en':
        train = pd.DataFrame({'text':[sentence1, sentence2], 'label':[labels.index(s1label) ,labels.index(s2label)]})
    else:
        labels_mapping = {'🥺 Sadness':'sadness','😃 Joy':'happy', '😍 Love':'disgust', '😡 Anger':'anger','😱 Fear':'fear','😯 Surprise':'surprise'}
        other_labels1, other_labels2 = labels_mapping[s1label], labels_mapping[s2label]
        train = pd.DataFrame({'text':[sentence1, sentence2], 'label':[sintam_labels.index(other_labels1) ,sintam_labels.index(other_labels2)]})
    
    # Extract shap values based on pre-trained models
    # webapp/predictions/text/models/
    if lang == 'en':
        shaps = get_shap_values_as_list(train, 'mod_en_6000', 'tok_en_6000')
    elif lang == 'si':
        shaps = get_shap_values_as_list(train, 'mod_og_si', 'tok_og_si')
    elif lang == 'tm':
        shaps = get_shap_values_as_list(train, 'mod_og_tm', 'tok_og_tm')
    
    # Extract shap values as a list
    shaps = [s.values for s in shaps]
    
    # Extract important values
    imp1 = [w for i, w in enumerate(sentence1.strip().split()) if shaps[0][i] >= np.percentile(shaps[0], t)]
    imp2 = [w for i, w in enumerate(sentence2.strip().split()) if shaps[1][i] >= np.percentile(shaps[1], t)]
    
    # Display inputs and salient words
    if lang=='en':
        print(f"Input 1: {str(sentence1)} / {s1label} -> {imp1}")
        print(f"Input 2: {str(sentence2)} / {s2label} -> {imp2}")
    else:
        print(f"Input 1: {str(sentence1)} / {other_labels1} -> {imp1}")
        print(f"Input 2: {str(sentence2)} / {other_labels2} -> {imp2}")
    return shaps, imp1, imp2
        
def augment_sentences(factor, s1label, s2label, sentence1, sentence2, shaps, t, p):
    """
    Generate augmented sentences by swapping words based on SHAP values.
    
    Args:
        factor (int): Number of augmentations to generate.
        s1label (str): Label of the first sentence.
        s2label (str): Label of the second sentence.
        sentence1 (str): First input sentence.
        sentence2 (str): Second input sentence.
        shaps (list): SHAP values for both sentences.
        t (float): Percentile threshold for important words.
        p (float): Proportion of swaps to perform.
    
    Returns:
        list: Augmented versions of the sentences.
    """
    augmented_sents = []
    for _ in range(0,factor):
        if bool(s1label == s2label):
            aug_s = " ".join(SSwap().bisent_swap(sentence1, sentence2, shaps[0], shaps[1], t, True, p))
        else:
            aug_s = " ".join(SSwap().bisent_swap(sentence1, sentence2, shaps[0], shaps[1], t, False, p))
        augmented_sents.append(aug_s)
    return augmented_sents

def classify_augmentations(augmented_sents, labels, lang):
    """
    Classify augmented sentences and return their predicted labels.
    
    Args:
        augmented_sents (list): List of augmented sentences.
        labels (list): List of possible labels.
        lang (str): Language of the sentences.
    
    Returns:
        tuple: Augmented sentences and their predicted labels.
    """
    augmented_sentences = []
    augmented_labels = []
    for aug_s in augmented_sents:
        pred, confs= make_prediction(aug_s, lang=lang)
        augmented_sentences.append(aug_s)
        augmented_labels.append(labels[pred])
    return augmented_sentences, augmented_labels
    
