"""
Script to evaluate distilled models on common benchmarks using lm-evaluation-harness.
Supports evaluation on: arc_easy, arc_challenge, boolq, piqa, siqa, hellaswag, openbookqa, winogrande
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import time

import torch
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
import safetensors.torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from attention_approximation.utils import LOGGER


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    model_path: str
    model_config_path: str
    tokenizer_path: Optional[str] = None
    benchmarks: List[str] = None
    batch_size: int = 8
    device: str = "cuda"
    num_fewshot: int = 0
    seed: int = 42
    output_dir: str = "evaluation_results"
    limit: Optional[int] = None  # Limit number of examples per task (for debugging)
    use_cache: bool = True
    trust_remote_code: bool = False
    
    def __post_init__(self):
        if self.benchmarks is None:
            self.benchmarks = [
                "arc_easy",
                "arc_challenge", 
                "boolq",
                "piqa",
                "siqa",
                "hellaswag",
                "openbookqa",
                "winogrande"
            ]


class BenchmarkEvaluator:
    """Handles evaluation of models on standard benchmarks."""
    
    # Map of benchmark names to lm-eval task names
    TASK_MAPPING = {
        "arc_easy": "arc_easy",
        "arc_challenge": "arc_challenge",
        "boolq": "boolq",
        "piqa": "piqa",
        "siqa": "siqa",
        "hellaswag": "hellaswag",
        "openbookqa": "openbookqa",
        "obqa": "openbookqa",  # Alias
        "winogrande": "winogrande"
    }
    
    # Default number of few-shot examples for each task
    DEFAULT_FEWSHOT = {
        "arc_easy": 25,
        "arc_challenge": 25,
        "boolq": 0,
        "piqa": 0,
        "siqa": 0,
        "hellaswag": 10,
        "openbookqa": 0,
        "winogrande": 5
    }
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results = {}
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
        self.load_model()
        
    def load_model(self):
        """Load the model and tokenizer."""
        LOGGER.info(f"Loading model from {self.config.model_path}")
        
        # Load model config
        model_config = LlamaConfig.from_pretrained(self.config.model_config_path)
        
        # Initialize model
        self.model = LlamaForCausalLM(model_config)
        
        # Load model weights
        if self.config.model_path.endswith('.safetensors'):
            state_dict = safetensors.torch.load_file(self.config.model_path)
        else:
            state_dict = torch.load(self.config.model_path, map_location='cpu')
            # Handle checkpoint format
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
                
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        if self.config.tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_path,
                trust_remote_code=self.config.trust_remote_code
            )
        else:
            # Try to load tokenizer from model config path
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_config_path,
                    trust_remote_code=self.config.trust_remote_code
                )
            except:
                # Fallback to a default tokenizer
                LOGGER.warning("Could not load tokenizer, using default LLaMA tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "huggyllama/llama-7b",
                    trust_remote_code=False
                )
                
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        LOGGER.info(f"Model loaded successfully. Device: {self.device}")
        
    def create_lm_eval_model(self):
        """Create an lm-eval compatible model wrapper."""
        # Create HFLM wrapper for lm-eval
        lm = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            device=str(self.device),
            batch_size=self.config.batch_size,
            trust_remote_code=self.config.trust_remote_code
        )
        return lm
        
    def evaluate_single_benchmark(self, task_name: str, num_fewshot: Optional[int] = None) -> Dict:
        """
        Evaluate model on a single benchmark.
        
        Args:
            task_name: Name of the benchmark task
            num_fewshot: Number of few-shot examples (None to use default or config value)
            
        Returns:
            Dictionary containing evaluation results
        """
        # Map task name if needed
        actual_task = self.TASK_MAPPING.get(task_name, task_name)
        
        # Determine number of few-shot examples
        if num_fewshot is None:
            if self.config.num_fewshot >= 0:
                num_fewshot = self.config.num_fewshot
            else:
                num_fewshot = self.DEFAULT_FEWSHOT.get(actual_task, 0)
                
        LOGGER.info(f"Evaluating on {actual_task} with {num_fewshot}-shot...")
        
        # Create lm-eval model wrapper
        lm = self.create_lm_eval_model()
        
        # Run evaluation
        try:
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=[actual_task],
                num_fewshot=num_fewshot,
                batch_size=self.config.batch_size,
                device=str(self.device),
                no_cache=not self.config.use_cache,
                limit=self.config.limit,
                seed=self.config.seed
            )
            
            # Extract metrics
            task_results = results['results'][actual_task]
            
            # Get the primary metric (usually accuracy or acc_norm)
            if 'acc' in task_results:
                score = task_results['acc']
            elif 'acc_norm' in task_results:
                score = task_results['acc_norm']
            else:
                # Fallback to first metric
                metrics = {k: v for k, v in task_results.items() if not k.endswith('_stderr')}
                score = list(metrics.values())[0] if metrics else 0.0
                
            result = {
                'task': actual_task,
                'num_fewshot': num_fewshot,
                'score': score,
                'metrics': task_results,
                'samples': results['samples'][actual_task] if 'samples' in results else None
            }
            
            LOGGER.info(f"  {actual_task}: {score:.4f}")
            
            return result
            
        except Exception as e:
            LOGGER.error(f"Error evaluating {actual_task}: {e}")
            return {
                'task': actual_task,
                'num_fewshot': num_fewshot,
                'score': 0.0,
                'error': str(e)
            }
            
    def evaluate_all_benchmarks(self) -> Dict[str, Any]:
        """
        Evaluate model on all specified benchmarks.
        
        Returns:
            Dictionary containing all evaluation results
        """
        LOGGER.info(f"Starting evaluation on {len(self.config.benchmarks)} benchmarks")
        start_time = time.time()
        
        all_results = {}
        scores = []
        
        for benchmark in self.config.benchmarks:
            result = self.evaluate_single_benchmark(benchmark)
            all_results[benchmark] = result
            scores.append(result['score'])
            
            # Save intermediate results
            self.save_results(all_results, intermediate=True)
            
        # Calculate aggregate metrics
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        total_time = time.time() - start_time
        
        summary = {
            'model_path': self.config.model_path,
            'benchmarks': self.config.benchmarks,
            'average_score': avg_score,
            'individual_scores': {b: all_results[b]['score'] for b in self.config.benchmarks},
            'total_evaluation_time': total_time,
            'config': asdict(self.config)
        }
        
        self.results = {
            'summary': summary,
            'detailed_results': all_results
        }
        
        return self.results
        
    def save_results(self, results: Dict, intermediate: bool = False):
        """Save evaluation results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if intermediate:
            filename = self.output_dir / f"intermediate_results_{timestamp}.json"
        else:
            filename = self.output_dir / f"evaluation_results_{timestamp}.json"
            
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        LOGGER.info(f"Results saved to {filename}")
        
    def print_summary(self):
        """Print a formatted summary of results."""
        if not self.results:
            LOGGER.warning("No results to display")
            return
            
        summary = self.results.get('summary', {})
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        print(f"Model: {summary.get('model_path', 'Unknown')}")
        print(f"Average Score: {summary.get('average_score', 0):.4f}")
        print("-"*80)
        print(f"{'Benchmark':<20} {'Score':<10} {'Metric':<20}")
        print("-"*80)
        
        for benchmark in self.config.benchmarks:
            if benchmark in self.results.get('detailed_results', {}):
                result = self.results['detailed_results'][benchmark]
                score = result.get('score', 0)
                print(f"{benchmark:<20} {score:<10.4f} {'accuracy':<20}")
                
        print("="*80)
        print(f"Total evaluation time: {summary.get('total_evaluation_time', 0):.2f} seconds")
        print("="*80 + "\n")
        
    def compare_with_baseline(self, baseline_results_path: str):
        """
        Compare current results with baseline results.
        
        Args:
            baseline_results_path: Path to baseline results JSON file
        """
        with open(baseline_results_path, 'r') as f:
            baseline = json.load(f)
            
        if 'summary' not in baseline:
            LOGGER.error("Invalid baseline results format")
            return
            
        baseline_scores = baseline['summary'].get('individual_scores', {})
        current_scores = self.results['summary'].get('individual_scores', {})
        
        print("\n" + "="*80)
        print("COMPARISON WITH BASELINE")
        print("="*80)
        print(f"{'Benchmark':<20} {'Current':<10} {'Baseline':<10} {'Diff':<10} {'Rel %':<10}")
        print("-"*80)
        
        for benchmark in self.config.benchmarks:
            current = current_scores.get(benchmark, 0)
            baseline_val = baseline_scores.get(benchmark, 0)
            diff = current - baseline_val
            rel_percent = (diff / baseline_val * 100) if baseline_val != 0 else 0
            
            print(f"{benchmark:<20} {current:<10.4f} {baseline_val:<10.4f} "
                  f"{diff:+<10.4f} {rel_percent:+<10.2f}%")
                  
        # Overall comparison
        current_avg = self.results['summary'].get('average_score', 0)
        baseline_avg = baseline['summary'].get('average_score', 0)
        avg_diff = current_avg - baseline_avg
        avg_rel = (avg_diff / baseline_avg * 100) if baseline_avg != 0 else 0
        
        print("-"*80)
        print(f"{'AVERAGE':<20} {current_avg:<10.4f} {baseline_avg:<10.4f} "
              f"{avg_diff:+<10.4f} {avg_rel:+<10.2f}%")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on common benchmarks")
    
    # Model configuration
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to model weights (.safetensors or .pt)")
    parser.add_argument('--model_config', type=str, required=True,
                       help="Path to model config")
    parser.add_argument('--tokenizer_path', type=str, default=None,
                       help="Path to tokenizer (uses model_config if not specified)")
    
    # Benchmark selection
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['arc_easy', 'arc_challenge', 'boolq', 'piqa', 
                               'siqa', 'hellaswag', 'openbookqa', 'winogrande'],
                       help="Benchmarks to evaluate on")
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument('--num_fewshot', type=int, default=-1,
                       help="Number of few-shot examples (-1 for benchmark defaults)")
    parser.add_argument('--device', type=str, default='cuda',
                       help="Device to use for evaluation")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument('--limit', type=int, default=None,
                       help="Limit number of examples per task (for debugging)")
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help="Directory to save results")
    parser.add_argument('--compare_baseline', type=str, default=None,
                       help="Path to baseline results for comparison")
    
    # Other options
    parser.add_argument('--no_cache', action='store_true',
                       help="Disable caching in lm-eval")
    parser.add_argument('--trust_remote_code', action='store_true',
                       help="Trust remote code when loading models")
    
    args = parser.parse_args()
    
    # Create configuration
    config = BenchmarkConfig(
        model_path=args.model_path,
        model_config_path=args.model_config,
        tokenizer_path=args.tokenizer_path,
        benchmarks=args.benchmarks,
        batch_size=args.batch_size,
        device=args.device,
        num_fewshot=args.num_fewshot,
        seed=args.seed,
        output_dir=args.output_dir,
        limit=args.limit,
        use_cache=not args.no_cache,
        trust_remote_code=args.trust_remote_code
    )
    
    # Run evaluation
    evaluator = BenchmarkEvaluator(config)
    results = evaluator.evaluate_all_benchmarks()
    
    # Save final results
    evaluator.save_results(results)
    
    # Print summary
    evaluator.print_summary()
    
    # Compare with baseline if provided
    if args.compare_baseline:
        evaluator.compare_with_baseline(args.compare_baseline)
        
    LOGGER.info("Evaluation complete!")


if __name__ == "__main__":
    main()