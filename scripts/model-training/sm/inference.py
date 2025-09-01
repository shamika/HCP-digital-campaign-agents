#!/usr/bin/env python3
"""
SageMaker inference script for HCP campaign model
"""
import os
import json
import torch
import torch.nn.functional as F
import numpy as np

# Simplified inference - we only need the embeddings, not the full GNN model

def model_fn(model_dir):
    """
    Load the model for inference - simplified to only use embeddings
    """
    # Load the saved embeddings
    model_path = os.path.join(model_dir, "node_emb.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the embeddings
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create embedding layers (dimensions should match training)
    emb_dim = 64
    
    # Get the actual dimensions from the saved embeddings
    hcp_weight = checkpoint['hcp_emb']['weight']
    tactic_weight = checkpoint['tactic_emb']['weight']
    
    hcp_emb = torch.nn.Embedding(hcp_weight.shape[0], hcp_weight.shape[1])
    tactic_emb = torch.nn.Embedding(tactic_weight.shape[0], tactic_weight.shape[1])
    
    # Load the trained weights
    hcp_emb.load_state_dict(checkpoint['hcp_emb'])
    tactic_emb.load_state_dict(checkpoint['tactic_emb'])
    
    # Return only embeddings - no need for the full GNN model for inference
    return {
        'hcp_emb': hcp_emb,
        'tactic_emb': tactic_emb
    }

def input_fn(request_body, request_content_type):
    """
    Parse input data for inference
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """
    Make predictions using the loaded embeddings
    """
    hcp_emb = model_dict['hcp_emb']
    tactic_emb = model_dict['tactic_emb']
    
    # Extract HCP and tactic IDs from input
    hcp_ids = input_data.get('hcp_ids', [])
    tactic_ids = input_data.get('tactic_ids', [])
    
    if not hcp_ids or not tactic_ids:
        return {"error": "Please provide both hcp_ids and tactic_ids"}
    
    # Validate IDs are within bounds
    max_hcp_id = hcp_emb.num_embeddings - 1
    max_tactic_id = tactic_emb.num_embeddings - 1
    
    if any(hcp_id > max_hcp_id for hcp_id in hcp_ids):
        return {"error": f"HCP ID out of range. Max ID: {max_hcp_id}"}
    
    if any(tactic_id > max_tactic_id for tactic_id in tactic_ids):
        return {"error": f"Tactic ID out of range. Max ID: {max_tactic_id}"}
    
    # Convert to tensors
    hcp_tensor = torch.tensor(hcp_ids, dtype=torch.long)
    tactic_tensor = torch.tensor(tactic_ids, dtype=torch.long)
    
    # Get embeddings and calculate similarity
    with torch.no_grad():
        hcp_embeddings = hcp_emb(hcp_tensor)
        tactic_embeddings = tactic_emb(tactic_tensor)
        
        # Calculate similarity scores (dot product)
        scores = torch.mm(hcp_embeddings, tactic_embeddings.t())
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(scores)
    
    return {
        "hcp_ids": hcp_ids,
        "tactic_ids": tactic_ids,
        "engagement_probabilities": probabilities.tolist(),
        "top_recommendations": get_top_recommendations(probabilities, hcp_ids, tactic_ids),
        "model_info": {
            "num_hcps": hcp_emb.num_embeddings,
            "num_tactics": tactic_emb.num_embeddings,
            "embedding_dim": hcp_emb.embedding_dim
        }
    }

def get_top_recommendations(probabilities, hcp_ids, tactic_ids, top_k=3):
    """
    Get top-k tactic recommendations for each HCP
    """
    recommendations = []
    
    for i, hcp_id in enumerate(hcp_ids):
        hcp_scores = probabilities[i]
        
        # Get top-k tactics for this HCP
        top_indices = torch.topk(hcp_scores, min(top_k, len(tactic_ids))).indices
        
        hcp_recommendations = []
        for idx in top_indices:
            tactic_id = tactic_ids[idx.item()]
            score = hcp_scores[idx].item()
            hcp_recommendations.append({
                "tactic_id": tactic_id,
                "engagement_probability": score
            })
        
        recommendations.append({
            "hcp_id": hcp_id,
            "recommended_tactics": hcp_recommendations
        })
    
    return recommendations

def output_fn(prediction, accept):
    """
    Format the prediction output
    """
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")