import pandas as pd
import ast
import csv
import json
import random

# Input and output file paths
input_file = "NuminaMath_LLMoutputs_bm25_repllama_added.csv"
output_file = "NuminaMath_LLMoutputs_lexicalqueries_bm25_repllama_negatives_added_format.csv"

df = pd.read_csv(input_file)

# Open output CSV file and write header
with open(output_file, mode='w', newline='', encoding='utf-8') as out_csv:
    writer = csv.DictWriter(out_csv, fieldnames=df.columns.tolist() + ["query", "positive_passages", "negative_passages"])
    writer.writeheader()

    # Stream through the input file in chunks
    chunk_iter = pd.read_csv(input_file, chunksize=5000)

    for chunk in chunk_iter:
        # Keep only rows where flag is True
        chunk = chunk[chunk["flag_in_top20"] == True]

        for idx, row in chunk.iterrows():
            try:
                # Parse stringified lists
                bm25_ids = ast.literal_eval(row["BM25_top100_docids"])
                bm25_docs = ast.literal_eval(row["BM25_top100_docs"])
                repllama_ids = ast.literal_eval(row.get("Repllama_top10_docids", "[]"))
                repllama_docs = ast.literal_eval(row.get("Repllama_top10_docs", "[]"))

                # Convert gold id to string for safe comparisons
                gold_id = str(row["theorem_id"])

                # Construct query
                query = row["generated_LLM"]

                # Positive passage
                positive_passages = [{
                    "doc_id": int(gold_id),
                    "title": "",
                    "text": row["BRIGHT_theorem"]
                }]

                # Filter and sample 10 random BM25 negatives (excluding gold)
                bm25_subset = [
                    (str(docid), text) for docid, text in zip(bm25_ids[30:50], bm25_docs[30:50])
                    if str(docid) != gold_id
                ]
                bm25_negatives = random.sample(bm25_subset, min(10, len(bm25_subset)))

                negative_passages = [
                    {"doc_id": int(docid), "title": "", "text": text}
                    for docid, text in bm25_negatives
                ]


                
                # Add Repllama negatives (excluding gold)
                for docid, text in zip(repllama_ids, repllama_docs):
                    if str(docid) != gold_id:
                        negative_passages.append({
                            "doc_id": int(docid),
                            "title": "",
                            "text": text
                        })

                row_dict = row.to_dict()
                row_dict["query"] = query
                row_dict["positive_passages"] = json.dumps(positive_passages)
                row_dict["negative_passages"] = json.dumps(negative_passages)

                writer.writerow(row_dict)


            except Exception as e:
                print(f"Skipping row {idx} due to error: {e}")


