import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime

def load_results(results_dir):
    """Load all results from a results directory"""
    
    # Find all result JSON files
    result_files = glob.glob(f"{results_dir}/*_results.json")
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return None
    
    # Load all results
    all_results = []
    
    for file in result_files:
        try:
            with open(file, 'r') as f:
                result = json.load(f)
                print(f"Loaded result from {file}: {result}")  # Add this line to inspect
                all_results.append(result)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    print(f"Loaded {len(all_results)} result files")
    
    return all_results

def create_detailed_report(results, output_dir):
    """Create a detailed report of all models"""
    
    if not results:
        print("No results to analyze")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract key metrics for comparison
    comparison = []
    
    for result in results:
        # If result is a list, you might need to access elements inside the list
        if isinstance(result, list):
            for item in result:
                comparison.append({
                    'model_name': item['model_name'],
                    'character_accuracy': item['metrics']['character_accuracy'],
                    'word_accuracy': item['metrics']['word_accuracy'],
                    'diacritic_error_rate': item['metrics']['diacritic_error_rate'],
                    'model_size': item['model_size'],
                    'training_time': item['training_time'],
                    'inference_time': item['inference_time'],
                    **item['hyperparameters']
                })
        else:
            comparison.append({
                'model_name': result['model_name'],
                'character_accuracy': result['metrics']['character_accuracy'],
                'word_accuracy': result['metrics']['word_accuracy'],
                'diacritic_error_rate': result['metrics']['diacritic_error_rate'],
                'model_size': result['model_size'],
                'training_time': result['training_time'],
                'inference_time': result['inference_time'],
                **result['hyperparameters']
            })

    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(comparison)
    
    # Calculate efficiency metrics
    df['accuracy_per_param'] = df['character_accuracy'] / (df['model_size'] / 1_000_000)  # Accuracy per million params
    df['accuracy_per_time'] = df['character_accuracy'] / df['training_time'] * 3600  # Accuracy per hour of training
    df['inference_efficiency'] = df['character_accuracy'] / df['inference_time']  # Higher is better
    
    # Sort by character accuracy (descending)
    df_sorted = df.sort_values('character_accuracy', ascending=False)
    
    # Save comparison to CSV
    df_sorted.to_csv(f"{output_dir}/detailed_comparison.csv", index=False)
    
    # Create HTML report
    create_html_report(df_sorted, output_dir)
    
    # Create advanced visualizations
    create_advanced_visualizations(df, output_dir)
    
    # Print top models by different metrics
    print("\nTop 3 models by character accuracy:")
    for i, row in df_sorted.head(3).iterrows():
        print(f"{i+1}. {row['model_name']}: {row['character_accuracy']:.4f}")
    
    print("\nTop 3 models by efficiency (accuracy/params):")
    for i, row in df.sort_values('accuracy_per_param', ascending=False).head(3).iterrows():
        print(f"{i+1}. {row['model_name']}: {row['accuracy_per_param']:.6f}")
    
    print("\nTop 3 models by inference efficiency:")
    for i, row in df.sort_values('inference_efficiency', ascending=False).head(3).iterrows():
        print(f"{i+1}. {row['model_name']}: {row['inference_efficiency']:.4f}")
    
    return df_sorted

def create_html_report(df, output_dir):
    """Create an HTML report with interactive tables"""
    
    # Create HTML file
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Arabic Diacritization Model Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .highlight {{ background-color: #e6f7ff; }}
            .container {{ margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <h1>Arabic Diacritization Model Comparison</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="container">
            <h2>Models Sorted by Character Accuracy</h2>
            {df.sort_values('character_accuracy', ascending=False).to_html(classes='dataframe', index=False)}
        </div>
        
        <div class="container">
            <h2>Models Sorted by Efficiency (Accuracy/Parameters)</h2>
            {df.sort_values('accuracy_per_param', ascending=False).to_html(classes='dataframe', index=False)}
        </div>
        
        <div class="container">
            <h2>Models Sorted by Inference Efficiency</h2>
            {df.sort_values('inference_efficiency', ascending=False).to_html(classes='dataframe', index=False)}
        </div>
        
        <div class="container">
            <h2>Models Sorted by Training Time</h2>
            {df.sort_values('training_time').to_html(classes='dataframe', index=False)}
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(f"{output_dir}/model_comparison.html", 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {output_dir}/model_comparison.html")

def create_advanced_visualizations(df, output_dir):
    """Create advanced visualizations for model comparison"""
    
    # Create directory for visualizations
    viz_dir = f"{output_dir}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set style
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    
    # 1. Parallel Coordinates Plot
    plt.figure(figsize=(14, 8))
    
    # Select columns for parallel coordinates
    cols = ['max_sequence_length', 'embedding_dim', 'hidden_dim', 'dropout_rate', 
            'learning_rate', 'character_accuracy', 'model_size', 'inference_time']
    
    # Normalize the data for better visualization
    df_norm = df[cols].copy()
    for col in cols:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    
    # Plot parallel coordinates
    pd.plotting.parallel_coordinates(
        df_norm.assign(model=df['model_name']), 'model', 
        colormap=plt.cm.viridis, alpha=0.7
    )
    
    plt.title('Parallel Coordinates Plot of Model Parameters and Performance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/parallel_coordinates.png", dpi=300)
    plt.close()
    
    # 2. Pairplot of key parameters
    plt.figure(figsize=(20, 20))
    key_cols = ['embedding_dim', 'hidden_dim', 'dropout_rate', 'character_accuracy', 'model_size']
    sns.pairplot(df[key_cols], height=3, aspect=1.2, plot_kws={'alpha': 0.6})
    plt.suptitle('Pairwise Relationships Between Key Parameters', y=1.02, fontsize=16)
    plt.savefig(f"{viz_dir}/pairplot.png", dpi=300)
    plt.close()
    
    # 3. 3D Scatter plot
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        df['model_size'] / 1_000_000,  # Convert to millions
        df['training_time'] / 60,  # Convert to minutes
        df['character_accuracy'],
        c=df['character_accuracy'],
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    
    ax.set_xlabel('Model Size (million parameters)')
    ax.set_ylabel('Training Time (minutes)')
    ax.set_zlabel('Character Accuracy')
    ax.set_title('3D Relationship: Model Size, Training Time, and Accuracy')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Character Accuracy')
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/3d_scatter.png", dpi=300)
    plt.close()
    
    # 4. Radar chart for top 5 models
    top_models = df.sort_values('character_accuracy', ascending=False).head(5)
    
    # Select metrics for radar chart
    metrics = ['character_accuracy', 'word_accuracy', 'accuracy_per_param', 
               'accuracy_per_time', 'inference_efficiency']
    
    # Normalize metrics for radar chart
    top_models_radar = top_models[metrics].copy()
    for metric in metrics:
        top_models_radar[metric] = (top_models_radar[metric] - top_models_radar[metric].min()) / \
                                  (top_models_radar[metric].max() - top_models_radar[metric].min())
    
    # Number of variables
    N = len(metrics)
    
    # Create angles for radar chart
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add lines and points for each model
    for i, (idx, row) in enumerate(top_models_radar.iterrows()):
        values = row.values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot line and points
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=top_models.iloc[i]['model_name'])
        ax.scatter(angles, values, s=50)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Radar Chart of Top 5 Models', size=15, y=1.1)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/radar_chart.png", dpi=300)
    plt.close()

def main():
    """Main function to analyze results"""
    print("Arabic Diacritization Model Analysis")
    print("=" * 80)
    
    # Get the latest results directory
    results_dirs = sorted(glob.glob("results/optimization_*"))
    
    if not results_dirs:
        print("No optimization results found")
        return
    
    latest_dir = results_dirs[-1]
    print(f"Analyzing results from {latest_dir}")
    
    # Load results
    results = load_results(latest_dir)
    
    if not results:
        return
    
    # Create analysis directory
    analysis_dir = f"{latest_dir}/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create detailed report
    df = create_detailed_report(results, analysis_dir)
    
    print(f"\nAnalysis complete! Results saved to {analysis_dir}")


if __name__ == "__main__":
    main()