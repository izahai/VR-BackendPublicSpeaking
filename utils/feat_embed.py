from sentence_transformers import SentenceTransformer, util

def bert_feat_embed(
    model_bert : SentenceTransformer, 
    ls_cluster : list,
) -> list:
    ls_embed_cluster = []
    for cluster in ls_cluster:
        embed_cluster = []
        for line in cluster:
            embed_line = model_bert.encode(line, convert_to_tensor=True)
            embed_cluster.append(embed_line)
        ls_embed_cluster.append(embed_cluster)
    return ls_embed_cluster
    
def maximun_similarity(
    target: str,
    cluster: list,
):
    max_sim = -1
    max_index = -1

    for idx, line in enumerate(cluster):
        sim = util.cos_sim(target, line).item()
        if sim > max_sim:
            max_sim = sim
            max_index = idx

    return max_sim, max_index