# 14 lines

max_char_a_line = 49

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

    i = 7
    cluster = []
    ls_cluster.append([])
    if len(lines) > 7:
        while i < len(lines):
            if ((i-7) % num_line_per_cluster) == 0 and i != 7:
                ls_cluster.append(cluster)
                cluster = []
            cluster.append(" ".join([lines[i-1], lines[i]]))
            i += 1

    # The last cluster
    if cluster:
        ls_cluster.append(cluster)
        
    return "\n".join(lines), ls_cluster, len(lines)