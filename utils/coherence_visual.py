import matplotlib.pyplot as plt

# Step 1: Read the .txt file and count words in each line
def count_words_per_line(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        word_counts = [len(line.strip().split()) for line in lines]
    return word_counts

# Step 2: Plot and save the line diagram
def plot_word_counts(word_counts, output_img):
    plt.figure(figsize=(10, 5))
    
    # Generate X values in seconds (10, 20, 30, ...)
    x_values = [10 * i for i in range(1, len(word_counts) + 1)]
    
    # Plot line chart
    plt.plot(x_values, word_counts, marker='o', linestyle='-', color='blue')
    
    # Add value annotations above each point
    for x, y in zip(x_values, word_counts):
        plt.text(x, y + 0.5, str(y), ha='center', va='bottom', fontsize=9)

    # Titles and labels
    plt.title('Word Count Per 10 Seconds')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Word Count')
    plt.grid(True)
    
    # Save and close
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()
    print(f"Diagram saved to {output_img}")

def speed_visulize(transc_path, output_img='output.png'):
    word_counts = count_words_per_line(transc_path)
    plot_word_counts(word_counts, output_img)