from fastcoref import LingMessCoref

# Example text
texto_ata = (
    "O Presidente da Câmara Municipal, João Manuel Ferreira Costa, declarou aberta a reunião. "
    "O Sr. Presidente informou que João Ferreira Costa tinha aprovado a proposta n.º 2341/2023. "
    "A proposta foi apresentada pelo Presidente da Câmara na sessão anterior."
)


def main():
    print("--- BASELINE: LINGMESS (biu-nlp/lingmess-coref) ---")

    # Load the LingMess coreference model (downloads automatically on first run)
    model = LingMessCoref(model_name_or_path="biu-nlp/lingmess-coref", device="cpu")

    # Run coreference resolution on the input text
    predictions = model.predict(texts=[texto_ata])

    # Extract coreference clusters from the predictions
    clusters = predictions[0].get_clusters(as_strings=True)

    print("Detected coreference clusters:")
    for i, cluster in enumerate(clusters):
        # Each cluster is a list of mentions that refer to the same entity
        print(f"  Cluster {i + 1}: {cluster}")

    # Extract clusters as character offsets for pseudonymization
    clusters_offsets = predictions[0].get_clusters(as_strings=False)

    print("\nPseudonymized Text:")
    texto_pseudonimizado = texto_ata

    # Assign a consistent pseudonym ID to each mention in the same cluster
    for i, cluster in enumerate(clusters_offsets):
        # Iterate in reverse to avoid index shifting when replacing substrings
        for start, end in sorted(cluster, reverse=True):
            tag = f"<MENTION-{i + 1}>"
            texto_pseudonimizado = texto_pseudonimizado[:start] + tag + texto_pseudonimizado[end:]

    print(texto_pseudonimizado)


if __name__ == "__main__":
    main()