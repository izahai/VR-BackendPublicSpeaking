# 14 lines

max_char_a_line = 46

def split_text(
    text: str, 
    max_chars: int = max_char_a_line,
    num_entries_per_cluster : int = 5,
    lines_per_entry: int = 2
) -> str:
    words = text.split()
    lines = []
    current_line = []

    # --- Splitting line by max char per line ---
    for word in words:
        # Check if adding the next word exceeds the max_char limit
        if sum(len(w) for w in current_line) + len(current_line) + len(word) <= max_chars:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    # Add the last line if it exists
    if current_line:
        lines.append(" ".join(current_line))

    # --- Pair up the lines into 2-line entries ---
    paired_lines = []
    for i in range(0, len(lines)):
        # Join 2 lines with a newline character
        pair = " ".join(lines[i:i + lines_per_entry])
        paired_lines.append(pair)
    
    # --- Create clusters using the paired lines ---
    ls_cluster = []
    for i in range(0, len(paired_lines), num_entries_per_cluster):
        cluster = paired_lines[i:i + num_entries_per_cluster]
        ls_cluster.append(cluster)

    # Write to file
    with open("ls_cluster.txt", "w") as f:
        for item in ls_cluster:
            f.write(f"{item}\n")

    return "\n".join(lines), ls_cluster, len(lines)