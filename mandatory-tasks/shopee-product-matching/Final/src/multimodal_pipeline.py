"""
Multimodal Product Matching Pipeline
------------------------------------
Module providing end-to-end multimodal entity resolution combining textual 
representations (TF-IDF), visual representations (ResNet-50 / CNN), and 
perceptual hash shortcuts for e-commerce product matching.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_product_title(text: str) -> str:
    """Normalizes title text by stripping promotional boilerplate brackets,
    punctuation, and excessive whitespace."""
    text = re.sub(r'\[.*?\]|\(.*?\)', ' ', str(text))
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def calculate_macro_f1(preds_list, truth_list):
    """Calculates macro-averaged Precision, Recall, and F1 score across all listings."""
    f1s, precs, recs = [], [], []
    for preds, truth in zip(preds_list, truth_list):
        preds = set(preds)
        truth = set(truth)
        inter = len(preds & truth)
        p = inter / len(preds) if len(preds) > 0 else 0.0
        r = inter / len(truth) if len(truth) > 0 else 0.0
        f1 = (2 * inter) / (len(preds) + len(truth)) if (len(preds) + len(truth)) > 0 else 0.0
        
        precs.append(p)
        recs.append(r)
        f1s.append(f1)
        
    return {
        'macro_f1': float(np.mean(f1s)),
        'precision': float(np.mean(precs)),
        'recall': float(np.mean(recs))
    }


class MultimodalProductMatcher:
    """Complete multimodal matching engine integrating perceptual hashing,
    text vectorization, and dense visual embeddings."""
    
    def __init__(self, text_max_features=25000, alpha_text_weight=0.55):
        self.text_max_features = text_max_features
        self.alpha = alpha_text_weight
        self.text_vectorizer = TfidfVectorizer(
            max_features=text_max_features,
            preprocessor=clean_product_title,
            binary=True
        )
        self.phash_lookup = {}
        self.posting_ids = None
        self.text_matrix = None
        
    def fit_text(self, titles, posting_ids, phashes):
        """Fits text representation and builds perceptual hash index."""
        self.posting_ids = np.array(posting_ids)
        self.text_matrix = self.text_vectorizer.fit_transform(titles)
        
        # Build exact perceptual hash index
        df_hash = pd.DataFrame({'posting_id': posting_ids, 'phash': phashes})
        self.phash_lookup = df_hash.groupby('phash')['posting_id'].apply(set).to_dict()
        return self
        
    def predict_unimodal_text(self, threshold=0.55, chunk_size=2000):
        """Predicts matches using only text cosine similarity."""
        n = self.text_matrix.shape[0]
        all_preds = []
        
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = self.text_matrix[start:end]
            sim = (chunk @ self.text_matrix.T).toarray()
            
            for i in range(end - start):
                global_idx = start + i
                matches = set(self.posting_ids[np.where(sim[i] >= threshold)[0]])
                matches.add(self.posting_ids[global_idx])
                all_preds.append(matches)
                
        return all_preds

    def predict_phash_baseline(self, phashes):
        """Predicts matches using exact perceptual hash equality."""
        return [self.phash_lookup.get(ph, {pid}) for ph, pid in zip(phashes, self.posting_ids)]

    def predict_decision_union(self, text_preds, visual_preds, phash_preds=None):
        """Fuses predictions via Decision-Level Union (P = P_text U P_vision [U P_phash])."""
        combined = []
        n = len(text_preds)
        for i in range(n):
            matches = set(text_preds[i]) | set(visual_preds[i])
            if phash_preds is not None:
                matches |= set(phash_preds[i])
            matches.add(self.posting_ids[i])
            combined.append(matches)
        return combined

    def predict_score_fusion(self, visual_sim_matrix, threshold=0.72, chunk_size=2000):
        """Fuses similarity scores via weighted convex combination:
        S_multimodal = alpha * S_text + (1 - alpha) * S_vision"""
        n = len(self.posting_ids)
        all_preds = []
        
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            text_chunk = (self.text_matrix[start:end] @ self.text_matrix.T).toarray()
            vis_chunk = visual_sim_matrix[start:end]
            
            joint_sim = self.alpha * text_chunk + (1.0 - self.alpha) * vis_chunk
            
            for i in range(end - start):
                global_idx = start + i
                matches = set(self.posting_ids[np.where(joint_sim[i] >= threshold)[0]])
                matches.add(self.posting_ids[global_idx])
                all_preds.append(matches)
                
        return all_preds
