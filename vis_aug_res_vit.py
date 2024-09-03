import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('augmentation_results_yolo.csv')

# Combine Augmentation Type and Value
df['Technique'] = df['Augmentation Type'] + ' (' + df['Value'].astype(str) + ')'

# Sort by accuracy
df_sorted = df.sort_values('Top1 Accuracy', ascending=False)

# Create a categorical color palette
n_colors = df['Augmentation Type'].nunique()
palette = sns.color_palette("husl", n_colors)
color_dict = dict(zip(df['Augmentation Type'].unique(), palette))

# Set up the plot
plt.figure(figsize=(20, 16))  # Increased figure size
sns.set(style="whitegrid", font_scale=1.2)  # Increased overall font scale

# Create the bar plot
ax = sns.barplot(x='Top1 Accuracy', y='Technique', data=df_sorted, 
                 palette=[color_dict[t] for t in df_sorted['Augmentation Type']])

# Customize the plot
plt.title('Top1 Accuracy for Different Augmentation Techniques', fontsize=26)
plt.xlabel('Top1 Accuracy', fontsize=20)
plt.ylabel('Augmentation Technique', fontsize=20)

# Set x-axis range
plt.xlim(0.6, 1.0)

# Add value labels on the bars
for i, v in enumerate(df_sorted['Top1 Accuracy']):
    ax.text(v, i, f'{v:.3f}', va='center', ha='left', fontweight='bold', fontsize=16)

# Create a custom legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in color_dict.values()]
plt.legend(handles, color_dict.keys(), title='Augmentation Type', 
           bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15, title_fontsize=17)

# Increase tick label font size
plt.tick_params(axis='both', which='major', labelsize=16)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('augmentation_results_plot_yolo.jpg', dpi=500, bbox_inches='tight')

print("Plot has been saved as 'augmentation_results_plot.png'")

# Display summary statistics
print("\nSummary Statistics:")
summary = df.groupby('Augmentation Type')['Top1 Accuracy'].agg(['mean', 'max', 'min'])
summary = summary.sort_values('mean', ascending=False)
print(summary)