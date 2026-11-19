import torch
import numpy as np
import CLIP_.clip as clip
from torchvision import transforms
from collections import deque


class AdaptiveController:
    def __init__(self, args, target_image):
        self.args = args
        self.device = args.device
        
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device, jit=False)
        
        self.normalize_transform = transforms.Compose([
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        self.target_image = target_image
        self.target_features = self._encode_image(target_image)
        
        self.complexity_baseline = self._compute_complexity_baseline(target_image)
        
        self.semantic_history = deque(maxlen=20)
        self.complexity_history = deque(maxlen=20)
        self.convergence_history = deque(maxlen=10)
        
        self.best_semantic_score = float('-inf')
        self.best_complexity_score = float('inf')
        self.best_sketch = None
        
        self.semantic_threshold = args.semantic_threshold if hasattr(args, 'semantic_threshold') else 0.92
        self.complexity_threshold = args.complexity_threshold if hasattr(args, 'complexity_threshold') else 0.15
        self.convergence_threshold = args.convergence_threshold if hasattr(args, 'convergence_threshold') else 0.001
        
        self.min_strokes = args.min_strokes if hasattr(args, 'min_strokes') else 4
        self.max_strokes = args.max_strokes if hasattr(args, 'max_strokes') else 64
        self.current_strokes = 0
        
        self.stroke_increment = args.stroke_increment if hasattr(args, 'stroke_increment') else 4
        
        self.adaptive_factor = 1.0
        self.patience = args.patience if hasattr(args, 'patience') else 5
        self.no_improvement_count = 0
        
    def _encode_image(self, image):
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = True
        
        with torch.no_grad():
            image_norm = self.normalize_transform(image).to(self.device)
            features = self.clip_model.encode_image(image_norm)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        return features.detach()
    
    def _compute_complexity_baseline(self, image):
        gray = transforms.Grayscale()(image)
        if gray.dim() == 4:
            laplacian_filter = torch.tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], 
                                           dtype=torch.float32, device=self.device).unsqueeze(0)
            laplacian = torch.nn.functional.conv2d(gray, laplacian_filter, padding=1)
        else:
            laplacian_filter = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                           dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            laplacian = torch.nn.functional.conv2d(gray.unsqueeze(0).unsqueeze(0), laplacian_filter, padding=1)
        complexity = torch.mean(torch.abs(laplacian)).item()
        return complexity if complexity > 0 else 0.01
    
    def compute_semantic_similarity(self, sketch):
        sketch_features = self._encode_image(sketch)
        similarity = torch.cosine_similarity(sketch_features, self.target_features).item()
        return similarity
    
    def compute_sketch_complexity(self, sketch):
        gray = transforms.Grayscale()(sketch)
        if gray.dim() == 4:
            laplacian_filter = torch.tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], 
                                           dtype=torch.float32, device=self.device).unsqueeze(0)
            laplacian = torch.nn.functional.conv2d(gray, laplacian_filter, padding=1)
        else:
            laplacian_filter = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                           dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            laplacian = torch.nn.functional.conv2d(gray.unsqueeze(0).unsqueeze(0), laplacian_filter, padding=1)
        complexity = torch.mean(torch.abs(laplacian)).item()
        normalized_complexity = complexity / self.complexity_baseline
        return normalized_complexity
    
    def compute_convergence_rate(self):
        if len(self.semantic_history) < 5:
            return float('inf')
        
        recent = list(self.semantic_history)[-5:]
        if len(recent) < 2:
            return float('inf')
        
        diffs = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        avg_diff = sum(diffs) / len(diffs)
        return avg_diff
    
    def should_stop(self, sketch, num_strokes):
        semantic_score = self.compute_semantic_similarity(sketch)
        complexity_score = self.compute_sketch_complexity(sketch)
        convergence_rate = self.compute_convergence_rate()
        
        self.semantic_history.append(semantic_score)
        self.complexity_history.append(complexity_score)
        self.convergence_history.append(convergence_rate)
        
        self.current_strokes = num_strokes
        
        if semantic_score > self.best_semantic_score:
            self.best_semantic_score = semantic_score
            self.best_complexity_score = complexity_score
            self.best_sketch = sketch.clone().detach()
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        adjusted_semantic_thresh = self.semantic_threshold * self.adaptive_factor
        adjusted_complexity_thresh = self.complexity_threshold / self.adaptive_factor
        
        if num_strokes >= self.min_strokes:
            if semantic_score >= adjusted_semantic_thresh and complexity_score <= adjusted_complexity_thresh:
                if convergence_rate < self.convergence_threshold:
                    return True
        
        if num_strokes >= self.max_strokes:
            return True
        
        if self.no_improvement_count >= self.patience * 2:
            self.adaptive_factor *= 0.95
        
        return False
    
    def should_add_strokes(self):
        if self.current_strokes >= self.max_strokes:
            return False
        
        if len(self.semantic_history) < 3:
            return True
        
        recent_scores = list(self.semantic_history)[-3:]
        avg_improvement = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        if avg_improvement < 0.001 and self.current_strokes < self.max_strokes:
            return True
        
        if self.best_semantic_score < self.semantic_threshold * 0.9:
            return True
        
        return False
    
    def get_stroke_increment(self):
        base_increment = self.stroke_increment
        
        if self.best_semantic_score < self.semantic_threshold * 0.7:
            return base_increment * 2
        elif self.best_semantic_score < self.semantic_threshold * 0.85:
            return base_increment
        else:
            return max(1, base_increment // 2)
    
    def get_state(self):
        return {
            'current_strokes': self.current_strokes,
            'best_semantic_score': self.best_semantic_score,
            'best_complexity_score': self.best_complexity_score,
            'semantic_history': list(self.semantic_history),
            'complexity_history': list(self.complexity_history),
            'convergence_rate': self.compute_convergence_rate(),
            'adaptive_factor': self.adaptive_factor,
            'no_improvement_count': self.no_improvement_count
        }