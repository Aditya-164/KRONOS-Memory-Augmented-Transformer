"""
Persistent Memory Module for Neural Networks

This module implements a persistent memory system that allows neural networks to store and retrieve
information based on semantic similarity. The memory is organized in concept buckets, forming a
hierarchical structure that enables efficient information storage and retrieval.

Key components:
- ConceptBucket: Stores embeddings and associated data items
- RetrievalStrategy: Abstract class for implementing different retrieval algorithms
- PersistentMemory: Main module for memory operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union, Any, Protocol, Callable
from dataclasses import dataclass, field
import uuid
import logging
import warnings
from src.models.utils import cosine_similarity
from enum import Enum, auto
from abc import ABC, abstractmethod
import time

# Configure logging
logger = logging.getLogger(__name__)

class BucketStorageStrategy(Enum):
    """Enumeration of strategies for storing items in buckets."""
    FIFO = auto()  # First-in, first-out
    LIFO = auto()  # Last-in, first-out
    LRU = auto()   # Least recently used
    UTILITY = auto()  # Based on utility score


class BucketCreationPolicy(Enum):
    """Enumeration of policies for creating new buckets."""
    ALWAYS = auto()     # Always create new buckets
    THRESHOLD = auto()  # Create based on similarity threshold
    ADAPTIVE = auto()   # Adaptively create based on memory usage


@dataclass
class MemoryItem:
    """Class representing an item stored in memory."""
    input_embedding: torch.Tensor  # Original input embedding (stored on CPU)
    stored_data: torch.Tensor      # Processed data to store (stored on CPU)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    
    def update_access(self):
        """Update access statistics for this item."""
        self.access_count += 1
        self.timestamp = time.time()


@dataclass
class ConceptBucket:
    """
    Class representing a concept bucket in persistent memory.
    
    Each bucket contains:
    - A concept embedding representing the semantic meaning
    - A list of items associated with this concept
    - Statistics for memory management
    - Hierarchical structure information
    """
    embedding: torch.Tensor  # Concept embedding (device-agnostic, moved as needed)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    items: List[MemoryItem] = field(default_factory=list)
    
    # Statistics for memory management
    access_frequency: int = 0
    last_access_time: float = field(default_factory=time.time)
    information_gain: float = 1.0
    
    # Hierarchical structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    def add_item(self, input_embedding: torch.Tensor, data_tensor: torch.Tensor) -> None:
        """
        Add a new item to this bucket.
        
        Args:
            input_embedding: Original input embedding reference
            data_tensor: Processed data to store
        """
        # Store embeddings on CPU to save GPU memory
        item = MemoryItem(
            input_embedding=input_embedding.detach().cpu(),
            stored_data=data_tensor.detach().cpu()
        )
        self.items.append(item)
        self.access_frequency += 1
        self.last_access_time = time.time()
    
    def remove_item(self, index: int) -> None:
        """
        Remove an item from this bucket.
        
        Args:
            index: Index of the item to remove
        """
        if 0 <= index < len(self.items):
            del self.items[index]
    
    def get_utility(self) -> float:
        """
        Calculate the utility score for this bucket.
        
        Returns:
            Utility score based on access frequency and information gain
        """
        # Age decay - reduce utility as time passes
        time_decay = 1.0 / (1.0 + 0.01 * (time.time() - self.last_access_time))
        return self.access_frequency * self.information_gain * time_decay
    
    def update_access(self) -> None:
        """Update access statistics for this bucket."""
        self.access_frequency += 1
        self.last_access_time = time.time()


class RetrievalStrategy(ABC):
    """
    Abstract base class for retrieval strategies.
    
    Subclasses implement different algorithms for retrieving items from memory.
    """
    
    @abstractmethod
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        buckets: Dict[str, ConceptBucket],
        target_device: torch.device,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Retrieve items from memory based on query embedding.
        
        Args:
            query_embedding: Query embedding to match against concepts
            buckets: Dictionary of available concept buckets
            target_device: Device to place retrieved tensors on
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of retrieved data tensors
        """
        pass


class TopKBucketsRetrieval(RetrievalStrategy):
    """
    Retrieval strategy that selects the top-k most similar buckets.
    """
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        buckets: Dict[str, ConceptBucket],
        target_device: torch.device,
        top_k_buckets: int = 1,
        top_k_items_per_bucket: int = 1,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Retrieve items from top-k similar buckets.
        
        Args:
            query_embedding: Query embedding to match against concepts
            buckets: Dictionary of available concept buckets
            target_device: Device to place retrieved tensors on
            top_k_buckets: Number of most similar buckets to use
            top_k_items_per_bucket: Number of items to retrieve from each bucket
            
        Returns:
            List of retrieved data tensors
        """
        retrieved_items = []
        
        if not buckets:
            return retrieved_items
        
        # Get all bucket embeddings
        bucket_ids = list(buckets.keys())
        if not bucket_ids:
            return retrieved_items
            
        bucket_embeddings = torch.stack([buckets[bid].embedding for bid in bucket_ids])
        bucket_embeddings = bucket_embeddings.to(query_embedding.device)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding.unsqueeze(0), bucket_embeddings).squeeze(0)
        
        # Get top-k buckets
        k = min(top_k_buckets, len(bucket_ids))
        if k == 0:
            return retrieved_items
            
        top_similarities, top_indices = torch.topk(similarities, k)
        
        # Retrieve items from each top bucket
        for idx in top_indices:
            bucket_id = bucket_ids[idx.item()]
            bucket = buckets[bucket_id]
            bucket.update_access()
            
            # Get top items from this bucket
            items_to_retrieve = min(top_k_items_per_bucket, len(bucket.items))
            if items_to_retrieve > 0:
                # Get most recently added items
                for item in bucket.items[-items_to_retrieve:]:
                    item.update_access()
                    retrieved_items.append(item.stored_data.to(target_device))
        
        return retrieved_items


class ThresholdRetrieval(RetrievalStrategy):
    """
    Retrieval strategy that selects buckets above a similarity threshold.
    """
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        buckets: Dict[str, ConceptBucket],
        target_device: torch.device,
        similarity_threshold: float = 0.7,
        max_items: int = 10,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Retrieve items from buckets with similarity above threshold.
        
        Args:
            query_embedding: Query embedding to match against concepts
            buckets: Dictionary of available concept buckets
            target_device: Device to place retrieved tensors on
            similarity_threshold: Minimum similarity to consider a bucket
            max_items: Maximum number of items to retrieve in total
            
        Returns:
            List of retrieved data tensors
        """
        retrieved_items = []
        
        if not buckets:
            return retrieved_items
        
        # Get all bucket embeddings
        bucket_ids = list(buckets.keys())
        if not bucket_ids:
            return retrieved_items
            
        bucket_embeddings = torch.stack([buckets[bid].embedding for bid in bucket_ids])
        bucket_embeddings = bucket_embeddings.to(query_embedding.device)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding.unsqueeze(0), bucket_embeddings).squeeze(0)
        
        # Sort buckets by similarity
        similarity_bucket_pairs = [(similarities[i].item(), bucket_ids[i]) 
                                   for i in range(len(bucket_ids))]
        similarity_bucket_pairs.sort(reverse=True)
        
        # Retrieve items from buckets above threshold
        items_retrieved = 0
        for sim, bucket_id in similarity_bucket_pairs:
            if sim < similarity_threshold or items_retrieved >= max_items:
                break
                
            bucket = buckets[bucket_id]
            bucket.update_access()
            
            # Get items from this bucket
            remaining_items = max_items - items_retrieved
            items_to_retrieve = min(remaining_items, len(bucket.items))
            
            if items_to_retrieve > 0:
                for item in bucket.items[-items_to_retrieve:]:
                    item.update_access()
                    retrieved_items.append(item.stored_data.to(target_device))
                    items_retrieved += 1
        
        return retrieved_items


@dataclass
class PersistentMemoryConfig:
    """Configuration for PersistentMemory module."""
    input_dim: int
    concept_embedding_dim: int
    value_dim: Optional[int] = None
    
    # Bucket creation and management
    bucket_creation_policy: BucketCreationPolicy = BucketCreationPolicy.THRESHOLD
    new_concept_similarity_threshold: float = 0.85
    max_items_per_bucket: int = 64
    storage_strategy: BucketStorageStrategy = BucketStorageStrategy.FIFO
    
    # Memory management
    utility_pruning_threshold: Optional[float] = None
    max_buckets: Optional[int] = None
    maintenance_interval: int = 100
    
    # Retrieval settings
    default_top_k_buckets: int = 1
    default_top_k_items_per_bucket: int = 1
    
    # Embedding projections
    use_layernorm: bool = True
    dropout_rate: float = 0.1
    
    # Memory access
    enable_write: bool = True
    enable_read: bool = True


@dataclass
class PersistentMemoryState:
    """State information for PersistentMemory module."""
    # Statistics
    total_items_stored: int = 0
    total_items_retrieved: int = 0
    total_buckets_created: int = 0
    total_buckets_pruned: int = 0
    
    # Metrics for monitoring
    avg_bucket_size: float = 0.0
    avg_retrieval_similarity: float = 0.0
    
    # Cache for repeated operations
    latest_query_embedding: Optional[torch.Tensor] = None
    latest_retrieved_items: List[torch.Tensor] = field(default_factory=list)


class PersistentMemory(nn.Module):
    """
    Persistent Memory module for neural networks.
    
    This module allows neural networks to maintain a long-term memory by storing
    and retrieving information based on semantic similarity. The memory is organized
    in concept buckets that form a hierarchical structure.
    
    Key features:
    - Semantic storage and retrieval based on embeddings
    - Hierarchical organization of concepts
    - Configurable memory management policies
    - Multiple retrieval strategies
    """
    
    def __init__(
        self,
        config: PersistentMemoryConfig,
        initial_core_concept_embeddings: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize the PersistentMemory module.
        
        Args:
            config: Configuration parameters
            initial_core_concept_embeddings: Pre-defined core concept embeddings
            device: Device to use for computation
        """
        super().__init__()
        
        self.config = config
        self.value_dim = config.value_dim if config.value_dim is not None else config.input_dim
        
        # Initialize concept buckets
        self.concept_buckets: Dict[str, ConceptBucket] = {}
        
        # Initialize state
        self.state = PersistentMemoryState()
        
        # Device handling
        self.module_device = device
        
        # Initialize retrieval strategies
        self.retrieval_strategies = {
            "top_k": TopKBucketsRetrieval(),
            "threshold": ThresholdRetrieval()
        }
        self.default_retrieval_strategy = "top_k"
        
        # Initialize projections with improved architecture
        # Input projection with optional layernorm and dropout
        input_projection = []
        input_projection.append(nn.Linear(config.input_dim, config.concept_embedding_dim))
        if config.use_layernorm:
            input_projection.append(nn.LayerNorm(config.concept_embedding_dim))
        if config.dropout_rate > 0:
            input_projection.append(nn.Dropout(config.dropout_rate))
        self.to_concept_embedding = nn.Sequential(*input_projection)
        
        # Value projection
        value_projection = []
        value_projection.append(nn.Linear(config.input_dim, self.value_dim))
        if config.use_layernorm:
            value_projection.append(nn.LayerNorm(self.value_dim))
        if config.dropout_rate > 0:
            value_projection.append(nn.Dropout(config.dropout_rate))
        self.to_value_embedding = nn.Sequential(*value_projection)
        
        # Output projection
        output_projection = []
        output_projection.append(nn.Linear(self.value_dim, config.input_dim))
        if config.use_layernorm:
            output_projection.append(nn.LayerNorm(config.input_dim))
        if config.dropout_rate > 0:
            output_projection.append(nn.Dropout(config.dropout_rate))
        self.to_output_embedding = nn.Sequential(*output_projection)
        
        # Gating mechanism to control memory influence
        self.memory_gate = nn.Sequential(
            nn.Linear(config.input_dim + config.input_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize with core concepts if provided
        if initial_core_concept_embeddings is not None:
            self._initialize_core_concepts(initial_core_concept_embeddings)
        
        # Counter for maintenance
        self.forward_calls_count = 0
    
    def _initialize_core_concepts(self, core_embeddings: torch.Tensor) -> None:
        """
        Initialize the memory with core concept embeddings.
        
        Args:
            core_embeddings: Tensor of core concept embeddings
        """
        if core_embeddings.shape[-1] != self.config.concept_embedding_dim:
            raise ValueError(
                f"Core concept embeddings dimension ({core_embeddings.shape[-1]}) "
                f"does not match concept_embedding_dim ({self.config.concept_embedding_dim})"
            )
        
        device = self.module_device or (
            next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        )
        
        for emb in core_embeddings:
            self.create_new_bucket(emb.to(device))
    
    def to(self, *args, **kwargs):
        """
        Move module to specified device and update module_device.
        
        Args:
            *args, **kwargs: Arguments to pass to nn.Module.to()
            
        Returns:
            Self for chaining
        """
        super().to(*args, **kwargs)
        
        # Update module_device based on parameter device
        if hasattr(self, 'to_concept_embedding') and list(self.to_concept_embedding.parameters()):
            self.module_device = next(self.to_concept_embedding.parameters()).device
        
        # Move concept bucket embeddings to new device
        for bucket_id in self.concept_buckets:
            self.concept_buckets[bucket_id].embedding = self.concept_buckets[bucket_id].embedding.to(self.module_device)
        
        return self
    
    def get_all_bucket_embeddings(self) -> Optional[torch.Tensor]:
        """
        Get all bucket embeddings stacked in a tensor.
        
        Returns:
            Tensor of all bucket embeddings or None if no buckets exist
        """
        if not self.concept_buckets:
            return None
        
        return torch.stack([bucket.embedding for bucket in self.concept_buckets.values()])
    
    def create_new_bucket(
        self,
        concept_embedding: torch.Tensor,
        parent_id: Optional[str] = None
    ) -> ConceptBucket:
        """
        Create a new concept bucket.
        
        Args:
            concept_embedding: Embedding representing the concept
            parent_id: ID of parent bucket for hierarchical organization
            
        Returns:
            Newly created ConceptBucket
        """
        # Create new bucket
        new_bucket = ConceptBucket(
            embedding=concept_embedding.detach().to(self.module_device),
            parent_id=parent_id
        )
        
        # Add to bucket dictionary
        self.concept_buckets[new_bucket.id] = new_bucket
        
        # Update parent-child relationship
        if parent_id and parent_id in self.concept_buckets:
            self.concept_buckets[parent_id].children_ids.append(new_bucket.id)
        
        # Update state
        self.state.total_buckets_created += 1
        
        # Enforce max buckets limit if specified
        if (self.config.max_buckets is not None and 
            len(self.concept_buckets) > self.config.max_buckets):
            self._prune_least_useful_bucket()
        
        return new_bucket
    
    def _prune_least_useful_bucket(self) -> None:
        """
        Remove the least useful bucket based on utility scores.
        """
        if not self.concept_buckets:
            return
        
        # Find the bucket with minimum utility that has no children
        min_utility = float('inf')
        min_bucket_id = None
        
        for bucket_id, bucket in self.concept_buckets.items():
            if not bucket.children_ids:  # Only consider childless buckets
                utility = bucket.get_utility()
                if utility < min_utility:
                    min_utility = utility
                    min_bucket_id = bucket_id
        
        # Remove the bucket if found
        if min_bucket_id:
            # Remove from parent's children list
            parent_id = self.concept_buckets[min_bucket_id].parent_id
            if parent_id and parent_id in self.concept_buckets:
                if min_bucket_id in self.concept_buckets[parent_id].children_ids:
                    self.concept_buckets[parent_id].children_ids.remove(min_bucket_id)
            
            # Delete the bucket
            del self.concept_buckets[min_bucket_id]
            self.state.total_buckets_pruned += 1
    
    def find_closest_buckets(
        self,
        query_concept_embedding: torch.Tensor,
        top_k: int = 1,
        similarity_threshold: Optional[float] = None
    ) -> List[List[Tuple[str, float]]]:
        """
        Find the closest buckets to the query embeddings.
        
        Args:
            query_concept_embedding: Query embeddings of shape (B, D_concept)
            top_k: Number of closest buckets to return
            similarity_threshold: Optional threshold for similarity filtering
            
        Returns:
            List of lists of (bucket_id, similarity) tuples, one list per batch item
        """
        query_concept_embedding = query_concept_embedding.to(self.module_device)
        all_bucket_embeddings = self.get_all_bucket_embeddings()
        
        if all_bucket_embeddings is None or all_bucket_embeddings.numel() == 0:
            return [[] for _ in range(query_concept_embedding.shape[0])]
        
        # Calculate similarities between query and all buckets
        similarities = cosine_similarity(query_concept_embedding, all_bucket_embeddings)
        
        # Get top-k most similar buckets for each batch item
        actual_top_k = min(top_k, similarities.shape[-1])
        if actual_top_k == 0:
            return [[] for _ in range(query_concept_embedding.shape[0])]
        
        top_k_sim, top_k_indices = torch.topk(similarities, actual_top_k, dim=-1)
        
        # Convert to (bucket_id, similarity) tuples
        bucket_ids = list(self.concept_buckets.keys())
        batch_results = []
        
        for i in range(query_concept_embedding.shape[0]):
            results_for_item = []
            for k in range(actual_top_k):
                bucket_idx = top_k_indices[i, k].item()
                sim_score = top_k_sim[i, k].item()
                
                # Apply similarity threshold if specified
                if similarity_threshold is None or sim_score >= similarity_threshold:
                    results_for_item.append((bucket_ids[bucket_idx], sim_score))
            
            batch_results.append(results_for_item)
        
        return batch_results
    
    def _store_item(
        self,
        input_embedding: torch.Tensor,
        concept_embedding: torch.Tensor,
        value_embedding: torch.Tensor
    ) -> None:
        """
        Store a single item in the appropriate bucket.
        
        Args:
            input_embedding: Original input embedding
            concept_embedding: Concept embedding for bucket matching
            value_embedding: Value embedding to store
        """
        if not self.config.enable_write:
            return
        
        # Find or create appropriate bucket based on policy
        bucket_to_store_in = None
        all_bucket_embeddings = self.get_all_bucket_embeddings()
        
        if self.config.bucket_creation_policy == BucketCreationPolicy.ALWAYS:
            # Always create a new bucket
            bucket_to_store_in = self.create_new_bucket(concept_embedding)
        
        elif all_bucket_embeddings is None or all_bucket_embeddings.numel() == 0:
            # No existing buckets, create first one
            bucket_to_store_in = self.create_new_bucket(concept_embedding)
        
        else:
            # Find closest bucket
            similarities = cosine_similarity(
                concept_embedding.unsqueeze(0),
                all_bucket_embeddings
            ).squeeze(0)
            
            max_sim, max_idx = torch.max(similarities, dim=0)
            
            if self.config.bucket_creation_policy == BucketCreationPolicy.THRESHOLD:
                # Create new bucket if similarity below threshold
                if max_sim.item() >= self.config.new_concept_similarity_threshold:
                    bucket_id = list(self.concept_buckets.keys())[max_idx.item()]
                    bucket_to_store_in = self.concept_buckets[bucket_id]
                else:
                    bucket_to_store_in = self.create_new_bucket(concept_embedding)
            
            elif self.config.bucket_creation_policy == BucketCreationPolicy.ADAPTIVE:
                # Create new bucket based on adaptive criteria
                threshold = self.config.new_concept_similarity_threshold
                # Adjust threshold based on number of buckets
                if self.config.max_buckets:
                    threshold *= 1.0 + 0.5 * (len(self.concept_buckets) / self.config.max_buckets)
                
                if max_sim.item() >= threshold:
                    bucket_id = list(self.concept_buckets.keys())[max_idx.item()]
                    bucket_to_store_in = self.concept_buckets[bucket_id]
                else:
                    bucket_to_store_in = self.create_new_bucket(concept_embedding)
        
        # Store the item in the selected bucket
        if bucket_to_store_in:
            if len(bucket_to_store_in.items) >= self.config.max_items_per_bucket:
                # Apply storage strategy
                if self.config.storage_strategy == BucketStorageStrategy.FIFO:
                    bucket_to_store_in.remove_item(0)  # Remove oldest
                elif self.config.storage_strategy == BucketStorageStrategy.LIFO:
                    bucket_to_store_in.remove_item(-1)  # Remove newest
                elif self.config.storage_strategy == BucketStorageStrategy.LRU:
                    # Remove least recently accessed
                    min_access_time = float('inf')
                    min_idx = 0
                    for i, item in enumerate(bucket_to_store_in.items):
                        if item.timestamp < min_access_time:
                            min_access_time = item.timestamp
                            min_idx = i
                    bucket_to_store_in.remove_item(min_idx)
                elif self.config.storage_strategy == BucketStorageStrategy.UTILITY:
                    # Remove item with lowest access count
                    min_access = float('inf')
                    min_idx = 0
                    for i, item in enumerate(bucket_to_store_in.items):
                        if item.access_count < min_access:
                            min_access = item.access_count
                            min_idx = i
                    bucket_to_store_in.remove_item(min_idx)
            
            # Add the new item
            bucket_to_store_in.add_item(input_embedding, value_embedding)
            self.state.total_items_stored += 1
    
    def _retrieve_items(
        self,
        query_concept_embedding: torch.Tensor,
        strategy_name: Optional[str] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Retrieve items from memory based on query embedding.
        
        Args:
            query_concept_embedding: Query embedding for matching
            strategy_name: Name of retrieval strategy to use
            **kwargs: Additional parameters for the retrieval strategy
            
        Returns:
            List of retrieved data tensors
        """
        if not self.config.enable_read:
            return []
        
        # Use default strategy if none specified
        if strategy_name is None:
            strategy_name = self.default_retrieval_strategy
        
        if strategy_name not in self.retrieval_strategies:
            warnings.warn(f"Unknown retrieval strategy: {strategy_name}. Using default.")
            strategy_name = self.default_retrieval_strategy
        
        # Get target device for retrieved items
        target_device = query_concept_embedding.device
        
        # Use the selected strategy to retrieve items
        strategy = self.retrieval_strategies[strategy_name]
        retrieved_items = strategy.retrieve(
            query_embedding=query_concept_embedding,
            buckets=self.concept_buckets,
            target_device=target_device,
            **kwargs
        )
        
        # Update state
        self.state.total_items_retrieved += len(retrieved_items)
        self.state.latest_query_embedding = query_concept_embedding.detach().cpu()
        self.state.latest_retrieved_items = [item.detach().cpu() for item in retrieved_items]
        
        return retrieved_items
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[PersistentMemoryState] = None,
        retrieval_strategy: Optional[str] = None,
        retrieval_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, PersistentMemoryState]:
        """
        Forward pass through the persistent memory module.
        
        Args:
            x: Input tensor of shape (B, N, D_input)
            state: Optional state information
            retrieval_strategy: Strategy to use for retrieval
            retrieval_params: Parameters for the retrieval strategy
            
        Returns:
            Tuple of (memory_output, updated_state)
        """
        batch_size, seq_len, _ = x.shape
        current_x_device = x.device
        self.to(current_x_device)
        
        # Update call counter
        self.forward_calls_count += 1
        
        # Prepare retrieval parameters
        if retrieval_params is None:
            retrieval_params = {}
        
        # Set default parameters if not specified
        if 'top_k_buckets' not in retrieval_params:
            retrieval_params['top_k_buckets'] = self.config.default_top_k_buckets
        if 'top_k_items_per_bucket' not in retrieval_params:
            retrieval_params['top_k_items_per_bucket'] = self.config.default_top_k_items_per_bucket
        
        # 1. Project input to concept and value embeddings
        # Average input over sequence dimension to get concept embedding
        x_avg_batch = x.mean(dim=1)  # (B, D_input)
        concept_keys_batch = self.to_concept_embedding(x_avg_batch)  # (B, D_concept)
        values_to_store_batch = self.to_value_embedding(x)  # (B, N, D_value)
        
        # 2. Store items: process each batch item
        if self.config.enable_write:
            for i in range(batch_size):
                current_concept_key = concept_keys_batch[i]  # (D_concept)
                for j in range(seq_len):
                    self._store_item(
                        input_embedding=x[i, j],
                        concept_embedding=current_concept_key,
                        value_embedding=values_to_store_batch[i, j]
                    )
        
        # 3. Retrieve items: use concept_keys_batch for querying
        memory_outputs = []
        for i in range(batch_size):
            # Retrieve items for this concept key
            retrieved_items = self._retrieve_items(
                query_concept_embedding=concept_keys_batch[i],
                strategy_name=retrieval_strategy,
                **retrieval_params
            )
            
            # Process retrieved items to match sequence length
            processed_items = []
            
            if not retrieved_items:
                # No items retrieved, use zero tensors
                processed_items = [torch.zeros(self.value_dim, device=current_x_device) 
                                  for _ in range(seq_len)]
            else:
                # Use retrieved items, padding or truncating as needed
                for item_tensor in retrieved_items:
                    processed_items.append(item_tensor)
                
                # Pad with zeros if necessary
                while len(processed_items) < seq_len:
                    processed_items.append(torch.zeros(self.value_dim, device=current_x_device))
                
                # Truncate if too many items
                if len(processed_items) > seq_len:
                    processed_items = processed_items[:seq_len]
            
            # Stack processed items for this batch item
            memory_outputs.append(torch.stack(processed_items, dim=0))  # (N, D_value)
        
        # Stack all batch outputs
        if memory_outputs:
            retrieved_values_batch = torch.stack(memory_outputs, dim=0)  # (B, N, D_value)
        else:
            # Fallback if batch_size was 0
            retrieved_values_batch = torch.zeros(batch_size, seq_len, self.value_dim, device=current_x_device)
        
        # 4. Project retrieved values back to input dimension
        memory_output_raw = self.to_output_embedding(retrieved_values_batch)  # (B, N, D_input)
        
        # 5. Apply gating mechanism to control memory influence
        # Concatenate original input and memory output
        gate_input = torch.cat([x, memory_output_raw], dim=-1)  # (B, N, 2*D_input)
        gate_value = self.memory_gate(gate_input)  # (B, N, 1)
        
        # Apply gate to memory output
        memory_output = gate_value * memory_output_raw
        
        # 6. Run maintenance if needed
        if (self.config.maintenance_interval > 0 and 
            self.forward_calls_count % self.config.maintenance_interval == 0):
            self.run_maintenance()
        
        # 7. Update state metrics
        if self.concept_buckets:
            self.state.avg_bucket_size = sum(len(b.items) for b in self.concept_buckets.values()) / len(self.concept_buckets)
        
        return memory_output, self.state
    
    def run_maintenance(self) -> None:
        """
        Perform maintenance operations on memory.
        
        This includes pruning low-utility buckets, consolidating similar buckets,
        and updating information gain metrics.
        """
        # 1. Prune low-utility buckets
        if self.config.utility_pruning_threshold is not None and len(self.concept_buckets) > 1:
            to_prune = []
            for cid, bucket in self.concept_buckets.items():
                # Only prune buckets without children
                if (not bucket.children_ids and 
                    bucket.get_utility() < self.config.utility_pruning_threshold):
                    to_prune.append(cid)
            
            # Remove pruned buckets
            for cid in to_prune:
                if cid in self.concept_buckets:
                    # Remove from parent's children list
                    parent_id = self.concept_buckets[cid].parent_id
                    if parent_id and parent_id in self.concept_buckets:
                        if cid in self.concept_buckets[parent_id].children_ids:
                            self.concept_buckets[parent_id].children_ids.remove(cid)
                    
                    # Delete the bucket
                    del self.concept_buckets[cid]
                    self.state.total_buckets_pruned += 1
        
        # 2. Update information gain metrics
        for bucket in self.concept_buckets.values():
            # Update information gain based on uniqueness
            if len(self.concept_buckets) > 1:
                # Calculate similarity to other buckets
                embedding = bucket.embedding.to(self.module_device)
                other_embeddings = [
                    b.embedding.to(self.module_device) 
                    for bid, b in self.concept_buckets.items() 
                    if bid != bucket.id
                ]
                
                if other_embeddings:
                    other_embeddings_tensor = torch.stack(other_embeddings)
                    similarities = cosine_similarity(
                        embedding.unsqueeze(0), other_embeddings_tensor
                    ).squeeze(0)
                    
                    # Higher avg similarity means lower information gain
                    avg_similarity = similarities.mean().item()
                    bucket.information_gain = max(0.1, 1.0 - avg_similarity)
    
    def get_bucket_by_id(self, bucket_id: str) -> Optional[ConceptBucket]:
        """
        Get a concept bucket by its ID.
        
        Args:
            bucket_id: ID of the bucket to retrieve
            
        Returns:
            ConceptBucket if found, None otherwise
        """
        return self.concept_buckets.get(bucket_id)
    
    def get_bucket_hierarchy(self) -> Dict[str, List[str]]:
        """
        Get the hierarchical structure of concept buckets.
        
        Returns:
            Dictionary mapping parent IDs to lists of child IDs
        """
        hierarchy = {}
        for bucket_id, bucket in self.concept_buckets.items():
            if bucket.parent_id:
                if bucket.parent_id not in hierarchy:
                    hierarchy[bucket.parent_id] = []
                hierarchy[bucket.parent_id].append(bucket_id)
        return hierarchy
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory usage.
        
        Returns:
            Dictionary of memory statistics
        """
        stats = {
            "num_buckets": len(self.concept_buckets),
            "total_items": sum(len(bucket.items) for bucket in self.concept_buckets.values()),
            "avg_bucket_size": self.state.avg_bucket_size,
            "total_buckets_created": self.state.total_buckets_created,
            "total_buckets_pruned": self.state.total_buckets_pruned,
            "total_items_stored": self.state.total_items_stored,
            "total_items_retrieved": self.state.total_items_retrieved,
        }
        return stats
    
    def add_retrieval_strategy(self, name: str, strategy: RetrievalStrategy) -> None:
        """
        Add a new retrieval strategy.
        
        Args:
            name: Name of the strategy
            strategy: Strategy implementation
        """
        self.retrieval_strategies[name] = strategy
    
    def set_default_retrieval_strategy(self, name: str) -> None:
        """
        Set the default retrieval strategy.
        
        Args:
            name: Name of the strategy to use as default
        """
        if name not in self.retrieval_strategies:
            raise ValueError(f"Unknown retrieval strategy: {name}")
        self.default_retrieval_strategy = name
    
    def clear_memory(self) -> None:
        """
        Clear all concept buckets from memory.
        """
        self.concept_buckets.clear()
        self.state.total_buckets_pruned += len(self.concept_buckets)
    
    def save_memory_state(self, path: str) -> None:
        """
        Save the memory state to disk.
        
        Args:
            path: Path to save the state
        """
        # Create state dictionary
        state_dict = {
            "config": self.config,
            "buckets": self.concept_buckets,
            "state": self.state,
            "forward_calls_count": self.forward_calls_count
        }
        
        # Save to disk
        torch.save(state_dict, path)
    
    def load_memory_state(self, path: str) -> None:
        """
        Load memory state from disk.
        
        Args:
            path: Path to load the state from
        """
        # Load state dictionary
        state_dict = torch.load(path)
        
        # Restore state
        self.config = state_dict["config"]
        self.concept_buckets = state_dict["buckets"]
        self.state = state_dict["state"]
        self.forward_calls_count = state_dict["forward_calls_count"]
        
        # Move buckets to current device
        for bucket_id in self.concept_buckets:
            self.concept_buckets[bucket_id].embedding = self.concept_buckets[bucket_id].embedding.to(self.module_device)


# Additional helper classes for advanced functionality

class AdaptiveThresholdRetrieval(RetrievalStrategy):
    """
    Retrieval strategy with adaptive similarity threshold based on memory usage.
    """
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        buckets: Dict[str, ConceptBucket],
        target_device: torch.device,
        base_threshold: float = 0.7,
        max_items: int = 10,
        memory_load_factor: float = 0.5,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Retrieve items with adaptive threshold based on memory load.
        
        Args:
            query_embedding: Query embedding to match against concepts
            buckets: Dictionary of available concept buckets
            target_device: Device to place retrieved tensors on
            base_threshold: Base similarity threshold
            max_items: Maximum number of items to retrieve
            memory_load_factor: Factor to adjust threshold based on memory load
            
        Returns:
            List of retrieved data tensors
        """
        # Adjust threshold based on number of buckets (proxy for memory load)
        num_buckets = len(buckets)
        adjusted_threshold = base_threshold
        if num_buckets > 10:  # Arbitrary threshold for "many buckets"
            # As memory load increases, increase threshold to be more selective
            memory_load = min(1.0, num_buckets / 100)  # Normalize to [0, 1]
            adjusted_threshold += memory_load * memory_load_factor
        
        # Use the adjusted threshold for retrieval
        retrieved_items = []
        
        if not buckets:
            return retrieved_items
        
        # Get all bucket embeddings
        bucket_ids = list(buckets.keys())
        if not bucket_ids:
            return retrieved_items
            
        bucket_embeddings = torch.stack([buckets[bid].embedding for bid in bucket_ids])
        bucket_embeddings = bucket_embeddings.to(query_embedding.device)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding.unsqueeze(0), bucket_embeddings).squeeze(0)
        
        # Sort buckets by similarity
        similarity_bucket_pairs = [(similarities[i].item(), bucket_ids[i]) 
                                   for i in range(len(bucket_ids))]
        similarity_bucket_pairs.sort(reverse=True)
        
        # Retrieve items from buckets above threshold
        items_retrieved = 0
        for sim, bucket_id in similarity_bucket_pairs:
            if sim < adjusted_threshold or items_retrieved >= max_items:
                break
                
            bucket = buckets[bucket_id]
            bucket.update_access()
            
            # Get items from this bucket
            remaining_items = max_items - items_retrieved
            items_to_retrieve = min(remaining_items, len(bucket.items))
            
            if items_to_retrieve > 0:
                for item in bucket.items[-items_to_retrieve:]:
                    item.update_access()
                    retrieved_items.append(item.stored_data.to(target_device))
                    items_retrieved += 1
        
        return retrieved_items


class MemoryOptimizer:
    """
    Helper class for optimizing memory organization.
    """
    
    def __init__(self, memory: PersistentMemory):
        """
        Initialize the memory optimizer.
        
        Args:
            memory: PersistentMemory instance to optimize
        """
        self.memory = memory
    
    def consolidate_similar_buckets(self, similarity_threshold: float = 0.95) -> int:
        """
        Consolidate highly similar buckets.
        
        Args:
            similarity_threshold: Threshold for considering buckets similar enough to merge
            
        Returns:
            Number of buckets consolidated
        """
        if len(self.memory.concept_buckets) <= 1:
            return 0
        
        consolidated_count = 0
        bucket_ids = list(self.memory.concept_buckets.keys())
        bucket_embeddings = [self.memory.concept_buckets[bid].embedding for bid in bucket_ids]
        bucket_embeddings_tensor = torch.stack(bucket_embeddings).to(self.memory.module_device)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(bucket_embeddings_tensor, bucket_embeddings_tensor)
        
        # Set diagonal to 0 to avoid self-similarity
        similarities.fill_diagonal_(0)
        
        # Find pairs above threshold
        highly_similar_pairs = []
        for i in range(len(bucket_ids)):
            for j in range(i+1, len(bucket_ids)):
                if similarities[i, j] > similarity_threshold:
                    highly_similar_pairs.append((bucket_ids[i], bucket_ids[j], similarities[i, j].item()))
        
        # Sort by similarity (highest first)
        highly_similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Process pairs
        merged_buckets = set()
        for id1, id2, sim in highly_similar_pairs:
            if id1 in merged_buckets or id2 in merged_buckets:
                continue
                
            if id1 in self.memory.concept_buckets and id2 in self.memory.concept_buckets:
                # Merge bucket2 into bucket1
                bucket1 = self.memory.concept_buckets[id1]
                bucket2 = self.memory.concept_buckets[id2]
                
                # Move items from bucket2 to bucket1
                for item in bucket2.items:
                    if len(bucket1.items) < self.memory.config.max_items_per_bucket:
                        bucket1.items.append(item)
                
                # Update access statistics
                bucket1.access_frequency += bucket2.access_frequency
                
                # Update embedding (weighted average)
                total_access = bucket1.access_frequency + bucket2.access_frequency
                if total_access > 0:
                    weight1 = bucket1.access_frequency / total_access
                    weight2 = bucket2.access_frequency / total_access
                    bucket1.embedding = (weight1 * bucket1.embedding + weight2 * bucket2.embedding).detach()
                
                # Update parent-child relationships
                if bucket2.parent_id and bucket2.parent_id in self.memory.concept_buckets:
                    parent = self.memory.concept_buckets[bucket2.parent_id]
                    if id2 in parent.children_ids:
                        parent.children_ids.remove(id2)
                
                # Transfer children to bucket1
                for child_id in bucket2.children_ids:
                    if child_id in self.memory.concept_buckets:
                        child = self.memory.concept_buckets[child_id]
                        child.parent_id = id1
                        if child_id not in bucket1.children_ids:
                            bucket1.children_ids.append(child_id)
                
                # Delete bucket2
                del self.memory.concept_buckets[id2]
                merged_buckets.add(id2)
                consolidated_count += 1
        
        return consolidated_count
    
    def rebalance_hierarchy(self) -> int:
        """
        Rebalance the bucket hierarchy based on similarities.
        
        Returns:
            Number of relationships changed
        """
        if len(self.memory.concept_buckets) <= 1:
            return 0
        
        changes_count = 0
        bucket_ids = list(self.memory.concept_buckets.keys())
        
        # Get embeddings for all buckets
        bucket_embeddings = [self.memory.concept_buckets[bid].embedding for bid in bucket_ids]
        bucket_embeddings_tensor = torch.stack(bucket_embeddings).to(self.memory.module_device)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(bucket_embeddings_tensor, bucket_embeddings_tensor)
        
        # For each bucket, find the most similar other bucket that could be its parent
        for i, bucket_id in enumerate(bucket_ids):
            bucket = self.memory.concept_buckets[bucket_id]
            
            # Skip root buckets (those without parents)
            if bucket.parent_id is None:
                continue
                
            # Find current parent similarity
            current_parent_idx = bucket_ids.index(bucket.parent_id) if bucket.parent_id in bucket_ids else -1
            current_parent_sim = similarities[i, current_parent_idx].item() if current_parent_idx >= 0 else -1
            
            # Find most similar potential parent
            max_sim = -1
            best_parent_idx = -1
            
            for j, other_id in enumerate(bucket_ids):
                if other_id == bucket_id:  # Skip self
                    continue
                if other_id in bucket.children_ids:  # Skip children
                    continue
                    
                sim = similarities[i, j].item()
                if sim > max_sim:
                    max_sim = sim
                    best_parent_idx = j
            
            # If we found a better parent with significantly higher similarity
            if best_parent_idx >= 0 and max_sim > current_parent_sim + 0.1:
                new_parent_id = bucket_ids[best_parent_idx]
                
                # Update relationships
                if bucket.parent_id in self.memory.concept_buckets:
                    self.memory.concept_buckets[bucket.parent_id].children_ids.remove(bucket_id)
                
                bucket.parent_id = new_parent_id
                self.memory.concept_buckets[new_parent_id].children_ids.append(bucket_id)
                changes_count += 1
        
        return changes_count