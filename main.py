import os
import streamlit as st
import pickle
import time
import langchain
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import *
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


# file_path = "faiss_store_openai.pkl"

from langchain.chains import RetrievalQAWithSourcesChain

from reapdat_langchain_document_loaders import CSVVectorStoreHandler
from reapdat_vectorstore_query_handler import VectorStoreQueryHandler

# Load environment variables
load_dotenv()

llm = OpenAI(temperature=0.1, max_tokens=500)
print("Data Loading...Started...✅✅✅")

if __name__ == "__main__":
    csv_path = "C:/Users/Mani_Moon/reapdat/input_files/madrasrest/madrasrest_reviews_sample.csv"
    vectorstore_dir = "C:/Users/Mani_Moon/reapdat/vectorestore/output"  # Replace with your output directory
    vectorstore_filename = "faiss_store_openai.pkl"
    # Replace with your CSV file path
    output_file = "C:/Users/Mani_Moon/reapdat/vectorestore/output/output.jpg"

    handler = CSVVectorStoreHandler(csv_path,vectorstore_dir,vectorstore_filename)
    query_handler = VectorStoreQueryHandler(vectorstore_dir,vectorstore_filename)

    # handler = CSVVectorStoreHandler(csv_path, vectorstore_dir, vectorstore_filename)
    # query_handler = VectorStoreQueryHandler(vectorstore_dir, vectorstore_filename)

    if handler.load_csv():
        print("CSV loaded successfully.")

        if handler.clean_documents():
            print("Documents cleaned successfully.")

        if handler.split_documents():
            print("Documents split successfully.")

        if handler.create_vectorstore_from_main():
            print("Vector store created successfully.")

        try:
            vectorstore = query_handler.load_vectorstore()

            ratings = handler.extract_ratings()
            # print(ratings)
            print("Ratings from documents loaded successfully")

            ratings_percentage = handler.calculate_ratings_percentages(ratings)
            print("Ratings percentage loaded successfully")

            neg_topics = query_handler.fetch_top_topics("negative", top_n=10)
            print("Negative topics loaded successfully")
            pos_topics = query_handler.fetch_top_topics("positive", top_n=10)
            print("positive topics loaded successfully")

            # Summary of the reviews
            reviews = query_handler.fetch_reviews_from_vectorstore(k=100)
            summary = query_handler.summarize_reviews(reviews)
            print("summary loaded successfully")

            ratings_percentage_dict = query_handler.convert_ratings_to_dict(ratings_percentage)

            query_handler.generate_plots_and_summary(ratings_percentage_dict, pos_topics, neg_topics, summary)
            print("Plot and image loaded successfully")
        except Exception as e:
            print(f"Error: {e}")












            # chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            # print("Re-loaded vector store successfully.")
            # query = "What is the sentiment analysis breakdown (positive, negative, neutral) of the reviews?"
            # response = chain({"question": query}, return_only_outputs=True)
            # print(response)

    #         if not data:
    #             st.error("No data could be loaded from the provided URLs. Please check the URLs and try again.")
    #         else:
    #             # Split data
    #             text_splitter = RecursiveCharacterTextSplitter(
    #                 separators=['\n\n', '\n', '.', ','],
    #                 chunk_size=1000
    #             )
    #             main_placeholder.text("Text Splitter...Started...✅✅✅")
    #             docs = text_splitter.split_documents(data)
    #
    #             if not docs:
    #                 st.error("No documents were created after splitting the text. Please check the input data.")
    #             else:
    #                 # Create embeddings and save to FAISS index
    #                 embeddings = OpenAIEmbeddings()
    #                 vectorstore_openai = FAISS.from_documents(docs, embeddings)
    #                 main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    #                 time.sleep(2)
    #
    #                 # Save the FAISS index to a pickle file
    #                 with open(file_path, "wb") as f:
    #                     pickle.dump(vectorstore_openai, f)
    #
    #                 st.success("FAISS index created and saved successfully! 🎉")
    #     except Exception as e:
    #         st.error(f"An error occurred: {e}")
