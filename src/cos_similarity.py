import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityCalculator:
    def __init__(self, embeddings_path='text_embeddings.pkl'):
        self.embeddings_path = embeddings_path
        self.embeddings = None
        self.texts = None
        self.load_embeddings()
    
    def load_embeddings(self):
        with open(self.embeddings_path, 'rb') as f: data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.texts = data['texts']
    
    def calculate_similarity(self, text1, text2):
        if isinstance(text1, str): idx1 = self._find_text_index(text1)
        else: idx1 = text1
        
        if isinstance(text2, str): idx2 = self._find_text_index(text2)
        else: idx2 = text2
        
        if idx1 >= len(self.embeddings) or idx2 >= len(self.embeddings): raise ValueError("索引超出范围")
        
        vec1 = self.embeddings[idx1].reshape(1, -1)
        vec2 = self.embeddings[idx2].reshape(1, -1)
        
        similarity = cosine_similarity(vec1, vec2)[0][0]
        
        return similarity
    
    def _find_text_index(self, text):
        try: return self.texts.index(text)
        except ValueError: raise ValueError(f"文本 '{text}' 不在已加载的文本列表中")
    
    def list_texts(self):
        for i, text in enumerate(self.texts): print(f"{i}: {text}")

if __name__ == "__main__":
    calculator = SimilarityCalculator('/home/dbx/BI/recommender/data/raw/review_text_embeddings.pkl')
    
    # calculator.list_texts()
    calculator.load_embeddings()
    # print(np.array(calculator.embeddings))
    array = np.array(calculator.embeddings)
    np.save('review_vectors.npy', array, allow_pickle=True)
    
    '''
    print("示例1: 通过索引计算相似度")
    similarity1 = calculator.calculate_similarity(0, 1)
    print(f"文本 0 和文本 1 的相似度: {similarity1:.4f}")
    
    
    print("示例2: 通过文本内容计算相似度")
    text_a = "Excellent customer service and wide variety of healthy options to choose from."
    text_b = "Great food, terrible service. Double check your bag before you leave!"
    similarity2 = calculator.calculate_similarity(text_a, text_b)
    print(f"相似度: {similarity2:.4f}")
    '''
    