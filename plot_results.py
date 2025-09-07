import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_results(results):
    # Initialize data structure for combinations
    combination_scores = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        method = result['retrieval_method']
        collection = result['collection']
        combination = f"{method}_{collection}"
        
        if result['cosine_score'] is not None:
            combination_scores[combination]['cosine'].append(result['cosine_score'])
        
        if result['bleurt_score'] is not None:
            combination_scores[combination]['bleurt'].append(result['bleurt_score'])
    
    return combination_scores

def plot_combination_scores(combination_scores):
    combinations = sorted(combination_scores.keys())
    metrics = ['cosine', 'bleurt']
    
    # Prepare data for plotting
    x = np.arange(len(combinations))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Calculate means for each combination and metric
    cosine_means = [np.mean(combination_scores[comb]['cosine']) for comb in combinations]
    bleurt_means = [np.mean(combination_scores[comb]['bleurt']) for comb in combinations]
    
    # Create bars
    rects1 = ax.bar(x - width/2, cosine_means, width, label='Cosine Score', color='skyblue')
    rects2 = ax.bar(x + width/2, bleurt_means, width, label='BLEURT Score', color='salmon')
    
    # Add labels and title
    ax.set_xlabel('Method-Collection Combinations')
    ax.set_ylabel('Scores')
    ax.set_title('Average Scores by Method-Collection Combination')
    ax.set_xticks(x)
    ax.set_xticklabels(combinations, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('combination_scores_bar.png')
    plt.close()

def plot_score_correlation(results):
    cosine_scores = []
    bleurt_scores = []
    methods = []
    
    for result in results:
        if result['cosine_score'] is not None and result['bleurt_score'] is not None:
            cosine_scores.append(result['cosine_score'])
            bleurt_scores.append(result['bleurt_score'])
            methods.append(result['retrieval_method'])
    
    plt.figure(figsize=(10, 6))
    
    # Create a color map for methods
    method_colors = {'cosine': 'blue', 'keyword': 'red', 'reranking': 'green'}
    
    for method in set(methods):
        mask = [m == method for m in methods]
        plt.scatter([cosine_scores[i] for i in range(len(cosine_scores)) if mask[i]],
                   [bleurt_scores[i] for i in range(len(bleurt_scores)) if mask[i]],
                   c=method_colors[method],
                   label=method,
                   alpha=0.6)
    
    plt.xlabel('Cosine Score')
    plt.ylabel('BLEURT Score')
    plt.title('Correlation between Cosine and BLEURT Scores')
    plt.grid(True)
    plt.legend()
    
    # Add correlation coefficient
    correlation = np.corrcoef(cosine_scores, bleurt_scores)[0, 1]
    plt.annotate(f'Correlation: {correlation:.2f}', 
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('score_correlation_colored.png')
    plt.close()

def main():
    # Load results
    results = load_results('evaluation_results_20250409_203934.json')
    
    # Analyze results
    combination_scores = analyze_results(results)
    
    # Create visualizations
    plot_combination_scores(combination_scores)
    plot_score_correlation(results)
    
    # Print summary statistics
    print("\nCombination-wise Average Scores:")
    for combination in sorted(combination_scores.keys()):
        print(f"\n{combination}:")
        for metric in combination_scores[combination]:
            scores = combination_scores[combination][metric]
            print(f"  {metric}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

if __name__ == "__main__":
    main() 