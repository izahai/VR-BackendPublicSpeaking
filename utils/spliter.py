# 14 lines

max_char_a_line = 46

def split_text(
    text: str, 
    max_chars: int = max_char_a_line,
    num_line_per_cluster : int = 7,
) -> str:
    words = text.split()
    lines = []
    current_line = []
    ls_cluster = []

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

    # Create spliting chunks (3 lines per chunk) into cluster
    i = num_line_per_cluster
    cluster = []
    ls_cluster.append([]) # First empty index
    if len(lines) > num_line_per_cluster:
        while i < len(lines):
            if ((i-num_line_per_cluster) % num_line_per_cluster) == 0 and i != num_line_per_cluster:
                ls_cluster.append(cluster)
                cluster = []
            cluster.append(" ".join([lines[i-2], lines[i-1], lines[i]]))
            i += 1

    # The last cluster
    if cluster:
        ls_cluster.append(cluster)
        
    with open("ls_cluster.txt", "w") as f:
        for item in ls_cluster:
            f.write(f"{item}\n")

    return "\n".join(lines), ls_cluster, len(lines)