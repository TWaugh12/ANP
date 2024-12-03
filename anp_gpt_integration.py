from AhpAnpLib import structs_AHPLib as structs 
from AhpAnpLib import inputs_AHPLib as inputs
from AhpAnpLib import calcs_AHPLib as calcs
from openai import OpenAI
from typing import List, Dict, Optional, Tuple, Any
import logging
import json
import numpy as np
import tempfile
import os
from dataclasses import dataclass
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LogprobComparison:
    """Enhanced comparison class that includes confidence metrics."""
    value: float
    logprob: float
    top_logprobs: List[Dict[str, float]]
    node_from: str
    node_a: str
    node_b: str
    context: str

    def confidence_score(self) -> float:
        primary_prob = np.exp(self.logprob)
        second_best = max((np.exp(lp["logprob"]) for lp in self.top_logprobs[1:]), default=0)
        return primary_prob - second_best

    def weighted_value(self) -> float:
        conf = self.confidence_score()
        if conf > 0.9:
            return self.value
        weighted_sum = self.value * conf
        remaining_weight = 1.0 - conf
        for alt in self.top_logprobs[1:]:
            alt_prob = np.exp(alt["logprob"])
            weighted_sum += float(alt["value"]) * (alt_prob * remaining_weight)
        return weighted_sum

    @property
    def is_highly_confident(self) -> bool:
        return self.confidence_score() > 0.9

@dataclass
class ModelAnalysis:
    """Stores analysis results for ANP model."""
    inconsistency: float
    confidence_metrics: Dict[str, float]
    matrix_properties: Dict[str, Any]
    priority_vectors: Dict[str, np.ndarray]

# 2. Main Manager Class
class ANPModelManager:
    """Manages ANP model creation and analysis with GPT integration."""
    
    def __init__(self, api_key: str):
        """Initialize manager with OpenAI API key."""
        self.client = OpenAI(api_key=api_key)
        self.current_model: Optional[structs.Model] = None
        self.logger = logging.getLogger(__name__)
        self.model_uuid = str(uuid.uuid4())[:8]
        self.comparison_history: List[LogprobComparison] = []
        
    def generate_model_structure(self, description: str) -> Dict[str, Any]:
        """Generate ANP model structure using GPT-4o."""
        try:
            schema = {
                "name": "anp_model_schema",  # Added required name parameter
                "schema": {
                    "type": "object",
                    "properties": {
                        "clusters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "nodes": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "inner_dependencies": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "from_node": {"type": "string"},
                                                "to_nodes": {
                                                    "type": "array",
                                                    "items": {"type": "string"}
                                                }
                                            },
                                            "required": ["from_node", "to_nodes"]
                                        }
                                    }
                                },
                                "required": ["name", "nodes", "inner_dependencies"]
                            }
                        },
                        "feedback_relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from_cluster": {"type": "string"},
                                    "to_cluster": {"type": "string"}
                                },
                                "required": ["from_cluster", "to_cluster"]
                            }
                        }
                    },
                    "required": ["clusters", "feedback_relationships"]
                }
            }

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "Create an ANP model structure with precise clusters, nodes, and relationships."
                    },
                    {
                        "role": "user",
                        "content": f"Create a detailed ANP model for: {description}"
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": schema
                }
            )
            
            structure = json.loads(response.choices[0].message.content)
            self.logger.info(f"Received model structure: {structure}")
            return structure

        except Exception as e:
            self.logger.error(f"Error creating model structure: {str(e)}")
            raise
    def generate_all_judgments(self) -> None:
        """Generate all required pairwise comparisons for the current model.
        
        This is the core function that:
        1. Identifies all required comparisons from model structure
        2. Generates comparison matrices with consistency checks
        3. Calculates priority vectors
        4. Handles both cluster and node level comparisons
        5. Validates the complete process
        """
        if not self.current_model:
            raise ValueError("No model currently loaded")

        try:
            self.logger.info("Starting comprehensive pairwise comparison generation")
            self.logger.info(f"Model UUID: {self.model_uuid}")
            
            # Reset model matrices and vectors
            self.current_model.all_pc_matrices = []
            self.current_model.all_pr_vectors = []
            total_comparisons_needed = 0
            total_matrices_needed = 0
            
            # Pre-calculate required comparisons
            for cluster in self.current_model.clusters:
                for node in cluster.nodes:
                    connected_clusters = self.current_model.retAllClusterConnectionsFromNode(node.name)
                    if connected_clusters:
                        for connected_cluster in connected_clusters:
                            connected_nodes = [n for n in connected_cluster.nodes 
                                            if n in self.current_model.retAllNodeConnectionsFrom(node.name)]
                            if len(connected_nodes) > 1:
                                n = len(connected_nodes)
                                total_comparisons_needed += (n * (n-1)) // 2
                                total_matrices_needed += 1

            self.logger.info(f"Required comparisons: {total_comparisons_needed}")
            self.logger.info(f"Required matrices: {total_matrices_needed}")

            # Process each cluster's nodes
            completed_comparisons = 0
            completed_matrices = 0
            
            for cluster in self.current_model.clusters:
                self.logger.info(f"\nProcessing cluster: {cluster.name}")
                
                # Track inner dependencies
                inner_dependencies = {}
                for node in cluster.nodes:
                    connected_within = [n for n in self.current_model.retAllNodeConnectionsFrom(node.name)
                                    if n.parCluster == cluster]
                    if connected_within:
                        inner_dependencies[node.name] = [n.name for n in connected_within]

                # Process each node's connections
                for node in cluster.nodes:
                    self.logger.info(f"\nProcessing node: {node.name}")
                    
                    # Get all clusters this node connects to
                    connected_clusters = self.current_model.retAllClusterConnectionsFromNode(node.name)
                    if not connected_clusters:
                        self.logger.info(f"No connections found for {node.name}")
                        continue

                    # Process each connected cluster
                    for connected_cluster in connected_clusters:
                        # Get actual connected nodes
                        connected_nodes = [n for n in connected_cluster.nodes 
                                        if n in self.current_model.retAllNodeConnectionsFrom(node.name)]
                        
                        if len(connected_nodes) <= 1:
                            self.logger.info(
                                f"Skipping {connected_cluster.name} - insufficient nodes for comparison"
                            )
                            continue

                        self.logger.info(
                            f"Generating comparisons for {node.name} with respect to {connected_cluster.name}"
                        )
                        
                        # Generate comparison matrix
                        matrix, analysis = self.generate_comparison_matrix(
                            node,
                            connected_nodes,
                            f"{node.name} with respect to {connected_cluster.name}"
                        )
                        
                        # Calculate and store priority vector
                        priority_vector = calcs.priorityVector(matrix)
                        
                        # Add to model
                        self.current_model.all_pc_matrices.append(matrix)
                        self.current_model.all_pr_vectors.append(priority_vector)
                        
                        # Update progress
                        completed_matrices += 1
                        completed_comparisons += (len(connected_nodes) * (len(connected_nodes)-1)) // 2
                        
                        # Log progress and analysis
                        progress = (completed_matrices / total_matrices_needed) * 100
                        self.logger.info(f"Progress: {progress:.1f}%")
                        self.logger.info(f"Matrix inconsistency: {analysis['inconsistency']:.3f}")
                        self.logger.info(
                            f"Average confidence: {analysis['confidence_analysis']['average_confidence']:.3f}"
                        )

            # Validate matrices and vectors match
            if len(self.current_model.all_pc_matrices) != len(self.current_model.all_pr_vectors):
                raise ValueError("Mismatch between number of matrices and priority vectors")
                
            return

        except Exception as e:
            self.logger.error(f"Error generating judgments: {str(e)}")
            raise

    def create_model_from_description(self, description: str) -> structs.Model:
        """Create and initialize ANP model from description."""
        try:
            structure = self.generate_model_structure(description)
            
            # Create model with shortened name for filesystem
            self.current_model = structs.Model(f"ANP_Model_{self.model_uuid}")
            self.current_model.description = description  # Store full description separately
            
            # Create clusters and nodes
            for cluster_data in structure["clusters"]:
                cluster = structs.Cluster(cluster_data["name"], len(self.current_model.clusters))
                self.current_model.addCluster2Model(cluster)
                
                for node_name in cluster_data["nodes"]:
                    node = structs.Node(node_name, len(cluster.nodes))
                    cluster.addNode2Cluster(node)

            # Add connections after all nodes exist
            self._add_model_connections(structure)
            
            return self.current_model
            
        except Exception as e:
            self.logger.error(f"Error creating model: {str(e)}")
            raise

    def _add_model_connections(self, structure: Dict[str, Any]) -> None:
        """Add all connections to the model."""
        # Add inner dependencies
        for cluster_data in structure["clusters"]:
            for dependency in cluster_data.get("inner_dependencies", []):
                from_node = dependency["from_node"]
                for to_node in dependency["to_nodes"]:
                    if self.current_model.getNodeObjByName(from_node) and \
                       self.current_model.getNodeObjByName(to_node):
                        self.current_model.addNodeConnectionFromTo(from_node, to_node)

        # Add feedback relationships
        for relationship in structure["feedback_relationships"]:
            from_cluster = self.current_model.getClusterObjByName(relationship["from_cluster"])
            to_cluster = self.current_model.getClusterObjByName(relationship["to_cluster"])
            if from_cluster and to_cluster:
                self.current_model.addNodeConnectionFromAllNodesToAllNodesOfCluster(
                    from_cluster.name, 
                    to_cluster.name
                )

    def generate_single_comparison(self, comparison: Dict[str, str]) -> LogprobComparison:
        """Generate single pairwise comparison with logprobs."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Make precise pairwise comparisons for ANP models."},
                    {"role": "user", "content": f"""
                    Compare {comparison['node_a']} vs {comparison['node_b']} 
                    with respect to {comparison['context']}.
                    Use Saaty's 1-9 scale:
                    1=Equal, 3=Moderate, 5=Strong, 7=Very Strong, 9=Extreme
                    (2,4,6,8 are intermediate values)
                    Return value and direction.
                    """}
                ],
                temperature=0.2,
                logprobs=True,
                top_logprobs=5
            )

            result = response.choices[0]
            value = int(result.message.content)
            logprob_data = result.logprobs.content[0]

            comparison_obj = LogprobComparison(
                value=value,
                logprob=logprob_data.logprob,
                top_logprobs=[{
                    "value": lp.token,
                    "logprob": lp.logprob
                } for lp in logprob_data.top_logprobs],
                **comparison
            )
            
            self.comparison_history.append(comparison_obj)
            return comparison_obj

        except Exception as e:
            self.logger.error(f"Error in comparison: {str(e)}")
            raise

    def generate_comparison_matrix(
        self,
        node_from: structs.Node,
        comparison_nodes: List[structs.Node],
        criterion: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate comparison matrix with confidence analysis and consistency checking.
        
        This method:
        1. Generates pairwise comparisons using logprobs
        2. Creates a comparison matrix
        3. Validates consistency
        4. Attempts to improve inconsistent matrices
        5. Provides detailed analysis
        """
        try:
            n = len(comparison_nodes)
            matrix = np.ones((n, n))
            comparisons = []
            confidence_metrics = []
            
            # Generate all comparisons with logprobs
            for i in range(n):
                for j in range(i+1, n):
                    comparison = {
                        "node_from": node_from.name,
                        "node_a": comparison_nodes[i].name,
                        "node_b": comparison_nodes[j].name,
                        "context": criterion
                    }
                    
                    # Get comparison with confidence
                    comparison_result = self.generate_single_comparison(comparison)
                    comparisons.append(comparison_result)
                    confidence_metrics.append(comparison_result.confidence_score())
                    
                    # Use weighted value based on confidence
                    weighted_value = comparison_result.weighted_value()
                    matrix[i,j] = weighted_value
                    matrix[j,i] = 1.0 / weighted_value

            # Calculate initial consistency
            initial_inconsistency = calcs.calcInconsistency(matrix)
            
            # If inconsistent, try to improve using confidence information
            if initial_inconsistency > 0.1:  # Saaty's threshold
                self.logger.warning(
                    f"Initial matrix inconsistency {initial_inconsistency:.3f} > 0.1. "
                    "Attempting to improve..."
                )
                
                # Sort comparisons by confidence
                sorted_comparisons = sorted(
                    zip(comparisons, range(len(comparisons))), 
                    key=lambda x: x[0].confidence_score()
                )
                
                # Try alternative values for least confident comparisons
                attempts = 0
                max_attempts = 5
                while initial_inconsistency > 0.1 and attempts < max_attempts:
                    least_confident = sorted_comparisons[0][0]
                    comp_idx = sorted_comparisons[0][1]
                    i = comparison_nodes.index(
                        self.current_model.getNodeObjByName(least_confident.node_a)
                    )
                    j = comparison_nodes.index(
                        self.current_model.getNodeObjByName(least_confident.node_b)
                    )
                    
                    # Try alternative values from logprobs
                    for alt in least_confident.top_logprobs[1:]:
                        test_matrix = matrix.copy()
                        test_value = float(alt["value"])
                        if test_value > 9:  # Stay within Saaty's scale
                            continue
                        
                        test_matrix[i,j] = test_value
                        test_matrix[j,i] = 1.0 / test_value
                        
                        test_inconsistency = calcs.calcInconsistency(test_matrix)
                        if test_inconsistency < initial_inconsistency:
                            matrix = test_matrix
                            initial_inconsistency = test_inconsistency
                            self.logger.info(
                                f"Improved inconsistency to {initial_inconsistency:.3f} "
                                f"using alternative value {test_value}"
                            )
                            break
                    
                    attempts += 1

            # Compile detailed analysis
            analysis = {
                "inconsistency": initial_inconsistency,
                "confidence_analysis": {
                    "average_confidence": np.mean(confidence_metrics),
                    "min_confidence": np.min(confidence_metrics),
                    "max_confidence": np.max(confidence_metrics),
                    "std_confidence": np.std(confidence_metrics)
                },
                "matrix_properties": {
                    "size": n,
                    "diagonal_consistency": np.allclose(np.diagonal(matrix), 1.0),
                    "reciprocal_consistency": np.allclose(
                        matrix * matrix.T, 
                        np.ones_like(matrix)
                    )
                },
                "comparisons": [
                    {
                        "nodes": (c.node_a, c.node_b),
                        "value": c.value,
                        "confidence": c.confidence_score(),
                        "alternatives": c.top_logprobs
                    }
                    for c in comparisons
                ]
            }

            # Log detailed matrix information
            self.logger.info(f"Generated {n}x{n} comparison matrix:")
            self.logger.info(f"\n{matrix}")
            self.logger.info(f"Final inconsistency: {initial_inconsistency:.3f}")
            
            return matrix, analysis

        except Exception as e:
            self.logger.error(
                f"Error generating comparison matrix for {node_from.name}: {str(e)}"
                )
            raise

    def analyze_matrix_confidence(self, comparisons: List[LogprobComparison]) -> Dict[str, Any]:
        """Analyze confidence levels in comparison matrix."""
        total = len(comparisons)
        high_conf = sum(1 for c in comparisons if c.is_highly_confident)
        
        return {
            "total_comparisons": total,
            "high_confidence_count": high_conf,
            "confidence_ratio": high_conf / total if total > 0 else 0,
            "average_confidence": np.mean([c.confidence_score() for c in comparisons]),
            "comparison_details": [
                {
                    "pair": (c.node_a, c.node_b),
                    "value": c.value,
                    "confidence": c.confidence_score(),
                    "alternatives": c.top_logprobs
                }
                for c in comparisons
            ]
        }

    def calculate_priorities(self) -> Dict[str, Any]:
        """Calculate final priorities with validation."""
        if not self.current_model:
            raise ValueError("No model currently loaded")
                
        try:
            # Validate we have all required matrices
            self.logger.info("Validating model data...")
            if not self.current_model.all_pc_matrices or not self.current_model.all_pr_vectors:
                raise ValueError("Missing pairwise comparison matrices or priority vectors")
                
            if len(self.current_model.all_pc_matrices) != len(self.current_model.all_pr_vectors):
                raise ValueError(f"Matrix/vector count mismatch: {len(self.current_model.all_pc_matrices)} matrices vs {len(self.current_model.all_pr_vectors)} vectors")
                
            # Calculate matrices
            self.logger.info("Calculating supermatrix...")
            super_matrix = calcs.calcUnweightedSuperMatrix(self.current_model)
            
            self.logger.info("Calculating weighted supermatrix...")
            weighted_matrix = calcs.calcWeightedSupermatrix(self.current_model)
            
            self.logger.info("Calculating limit matrix...")
            limit_matrix = calcs.calcLimitANP(weighted_matrix, self.current_model)
            
            # Extract priorities by cluster
            priorities = {}
            for cluster in self.current_model.clusters:
                try:
                    cluster_priorities = calcs.calcPrioritiesOfCluster(
                        weighted_matrix,
                        cluster.name,
                        self.current_model
                    )
                    priorities[cluster.name] = {
                        node.name: float(priority)
                        for node, priority in zip(cluster.nodes, cluster_priorities)
                    }
                except Exception as e:
                    self.logger.warning(f"Could not calculate priorities for cluster {cluster.name}: {str(e)}")

            # Compile results
            results = {
                "priorities": priorities,
                "matrices": {
                    "supermatrix": super_matrix.tolist(),
                    "weighted_matrix": weighted_matrix.tolist(),
                    "limit_matrix": limit_matrix.tolist()
                },
                "validation": {
                    "matrix_count": len(self.current_model.all_pc_matrices),
                    "vector_count": len(self.current_model.all_pr_vectors),
                    "model_size": sum(len(cluster.nodes) for cluster in self.current_model.clusters)
                }
            }
            
            self.logger.info("Priority calculation completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Error calculating priorities: {str(e)}")
            raise

    def analyze_all_comparisons(self) -> Dict[str, Any]:
        """Analyze all comparisons made in the model."""
        return {
            "total_comparisons": len(self.comparison_history),
            "high_confidence_comparisons": sum(
                1 for c in self.comparison_history if c.is_highly_confident
            ),
            "average_confidence": np.mean([
                c.confidence_score() for c in self.comparison_history
            ]),
            "lowest_confidence_comparisons": sorted(
                [(c.node_a, c.node_b, c.confidence_score()) 
                 for c in self.comparison_history],
                key=lambda x: x[2]
            )[:5]
        }

    def calculate_overall_confidence(self) -> float:
        """Calculate overall model confidence score."""
        if not self.comparison_history:
            return 0.0
        
        weights = [c.confidence_score() for c in self.comparison_history]
        return float(np.average(weights))