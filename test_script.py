# tests/test_anp_integration.py

from typing import Any
import pytest
import logging
import os
from anp_gpt_integration import ANPModelManager, LogprobComparison
import numpy as np
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'anp_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
class TestANPIntegration:
    """Test suite for ANP-GPT integration."""
    
    @pytest.fixture
    def manager(self):
        """Create ANPModelManager instance for testing."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OpenAI API key not found in environment variables")
        return ANPModelManager(api_key)
        
    @pytest.fixture
    def test_problem(self):
        """Define test problem description."""
        return """
        Create an ANP model for selecting the best renewable energy project.
        Consider:
        1. Alternatives: Solar, Wind, and Hydroelectric
        2. Main criteria: Environmental Impact, Cost, Technical Feasibility, Social Acceptance
        Additional considerations:
        - Environmental regulations and impact assessments
        - Long-term sustainability
        - Initial and maintenance costs
        - Technical complexity and reliability
        - Community acceptance and social impact
        """

    def test_model_creation(self, manager, test_problem):
        """Test model structure creation."""
        model = manager.create_model_from_description(test_problem)
        assert model is not None
        assert len(model.clusters) > 0
        
        # Validate cluster structure
        assert any(c.name == "Alternatives" for c in model.clusters)
        assert any(c.name == "Criteria" for c in model.clusters)
        
        # Validate node creation
        alt_cluster = next(c for c in model.clusters if c.name == "Alternatives")
        assert len(alt_cluster.nodes) == 3  # Solar, Wind, Hydroelectric
        
    def test_judgment_generation(self, manager, test_problem):
        """Test pairwise comparison generation."""
        model = manager.create_model_from_description(test_problem)
        manager.generate_all_judgments()
        
        # Validate matrices and vectors
        assert len(manager.current_model.all_pc_matrices) > 0
        assert len(manager.current_model.all_pc_matrices) == len(manager.current_model.all_pr_vectors)
        
        # Check matrix properties
        for matrix in manager.current_model.all_pc_matrices:
            assert matrix.shape[0] == matrix.shape[1]  # Square matrix
            assert np.allclose(np.diagonal(matrix), 1.0)  # Unity diagonal
            assert np.all(matrix > 0)  # Positive values
            
    def test_priority_calculation(self, manager, test_problem):
        """Test priority calculation and analysis."""
        model = manager.create_model_from_description(test_problem)
        manager.generate_all_judgments()
        results = manager.calculate_priorities()
        
        # Validate results structure
        assert "priorities" in results
        assert "matrices" in results
        assert "validation" in results
        
        # Check priority properties
        priorities = results["priorities"]
        for cluster_priorities in priorities.values():
            assert sum(cluster_priorities.values()) == pytest.approx(1.0, rel=1e-5)
            
    def test_confidence_analysis(self, manager, test_problem):
        """Test confidence metrics and analysis."""
        model = manager.create_model_from_description(test_problem)
        manager.generate_all_judgments()
        
        # Check comparison history
        assert len(manager.comparison_history) > 0
        
        # Validate confidence scores
        for comparison in manager.comparison_history:
            assert 0 <= comparison.confidence_score() <= 1
            assert comparison.logprob <= 0  # Log probabilities are negative
            
        # Test overall confidence
        overall_confidence = manager.calculate_overall_confidence()
        assert 0 <= overall_confidence <= 1
def print_section(title: str, content: Any, indent: int = 0) -> None:
    """Print section with formatting."""
    prefix = " " * indent
    print(f"\n{prefix}{'-' * 40}")
    print(f"{prefix}{title}")
    print(f"{prefix}{'-' * 40}")
    
    if isinstance(content, dict):
        for key, value in content.items():
            if isinstance(value, (dict, list)):
                print_section(key, value, indent + 2)
            else:
                print(f"{prefix}  {key}: {value}")
    elif isinstance(content, list):
        for item in content:
            print(f"{prefix}  - {item}")
    else:
        print(f"{prefix}  {content}")

def main():
    """Main test execution with detailed output."""
    try:
        # Initialize components
        api_key = "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"
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        manager = ANPModelManager(api_key)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting ANP analysis")

        # Run test problem
        problem = """
        Create an ANP model for selecting the best renewable energy project.
        Consider:
        1. Alternatives: Solar, Wind, and Hydroelectric
        2. Main criteria: Environmental Impact, Cost, Technical Feasibility, Social Acceptance
        Additional considerations:
        - Environmental regulations and impact assessments
        - Long-term sustainability
        - Initial and maintenance costs
        - Technical complexity and reliability
        - Community acceptance and social impact
        """

        # Create and analyze model
        logger.info("Creating model structure...")
        model = manager.create_model_from_description(problem)
        
        # Generate judgments
        logger.info("Generating pairwise comparisons...")
        manager.generate_all_judgments()
        
        # Calculate priorities
        logger.info("Calculating priorities...")
        results = manager.calculate_priorities()
        
        # Print detailed results
        print_section("Model Structure", {
            "UUID": manager.model_uuid,
            "Clusters": [
                {
                    "name": cluster.name,
                    "nodes": [node.name for node in cluster.nodes],
                    "connections": [
                        {
                            "from": node.name,
                            "to": [conn.name for conn in manager.current_model.retAllNodeConnectionsFrom(node.name)]
                        }
                        for node in cluster.nodes
                    ]
                }
                for cluster in model.clusters
            ]
        })
        
        print_section("Analysis Results", results)
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
