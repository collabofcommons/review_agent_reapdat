from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import re
import numpy as np
from langchain.chains import RetrievalQAWithSourcesChain
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from typing import List, Dict
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches


class VectorStoreQueryHandler:
    def __init__(self, vectorstore_dir, vectorstore_filename):

        self.vectorstore_dir = vectorstore_dir
        self.vectorstore = None
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore_filename = vectorstore_filename
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained sentence transformer model
        self.sentiment_analyzer = pipeline("sentiment-analysis")

        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            print("Sentiment analyzer initialized successfully.")
        except Exception as e:
            print(f"Error initializing sentiment analyzer: {e}")
            self.sentiment_analyzer = None

    def load_vectorstore(self):
        try:
            self.vectorstore = FAISS.load_local("faiss_index_dir", OpenAIEmbeddings(),
                                                allow_dangerous_deserialization=True)
            print(f"Vector store loaded successfully from {self.vectorstore_filename}.")
            # return True
            return self.vectorstore
        except Exception as e:
            print(f"An error occurred while loading the vector store: {e}")
            return False

    # chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())

    def query(self, query_text, top_k=5):
        if not self.vectorstore:
            print("Error: Vector store not loaded. Please load the vector store before querying.")
            return []

        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query_text, k=top_k)
            return [(doc.page_content, score) for doc, score in docs_with_scores]
        except Exception as e:
            print(f"An error occurred while querying the vector store: {e}")
            return []

    def fetch_reviews_from_vectorstore(self, k=100):
        """
        Fetch the top k reviews from the vector store.
        """
        if not self.vectorstore:
            print("Vector store is not loaded.")
            return []

        try:
            # Fetch documents (reviews) from the vector store
            docs_with_scores = self.vectorstore.similarity_search_with_score("", k=k)
            reviews = [doc.page_content for doc, _ in docs_with_scores]
            return reviews
        except Exception as e:
            print(f"Error fetching reviews from vector store: {e}")
            return []

    def fetch_top_topics(self, sentiment, top_n=10):
        """
        Fetch the top N dominant topics based on sentiment (positive/negative).
        """
        try:
            # Fetch reviews from the vector store
            reviews = self.fetch_reviews_from_vectorstore(k=100)  # Adjust k to get more reviews
            if not reviews:
                print("No reviews found in the vector store.")
                return "Error"
            llm = OpenAI(temperature=0.1, max_tokens=500)
            # Step 1: Use LLM (OpenAI) to extract the top N topics based on sentiment
            llm = OpenAI()  # OpenAI model for extracting topics
            prompt = PromptTemplate(
                input_variables=["reviews", "sentiment", "top_n"],
                template="From these reviews: {reviews}, extract the top {top_n} {sentiment} topics, each 5-6 words long."
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            topics = chain.run({"reviews": reviews, "sentiment": sentiment, "top_n": top_n})

            # Ensure topics are split into a list
            if isinstance(topics, str):
                topics = [topic.strip() for topic in topics.split('\n') if topic.strip()]

            # Step 2: Use Sentence-Transformers to find the similarity scores between topics and reviews
            topic_embeddings = self.model.encode(topics, convert_to_tensor=True)
            review_embeddings = self.model.encode(reviews, convert_to_tensor=True)

            # Calculate similarity scores for each topic in reviews
            topic_scores = {}
            for topic, topic_embedding in zip(topics, topic_embeddings):
                scores = util.cos_sim(topic_embedding, review_embeddings)
                topic_scores[topic] = scores.mean().item() * 100  # Convert to percentage

            # Step 3: Post-process the topics
            # Filter out irrelevant topics by checking the semantic similarity with reviews
            relevant_topics = []
            threshold = 0.2  # Adjust this value as needed to set a threshold for relevance
            relevant_topics = {topic: score for topic, score in topic_scores.items() if score > threshold}

            # for topic, score in topic_scores.items():
            #     # Check if the topic has enough semantic relevance to the reviews
            #     if score > threshold:
            #         relevant_topics.append(f"{topic} - {score:.2f}%")

            # # Return the formatted output with only relevant topics
            # if relevant_topics:
            #     formatted_output = "\n".join(relevant_topics)
            # else:
            #     formatted_output = "No relevant topics found."

            return relevant_topics

        except Exception as e:
            print(f"An error occurred while fetching topics: {e}")
            return "Error"

    def summarize_reviews(self, reviews):
        try:
            # Initialize the LLM
            llm = OpenAI()

            # Define the prompt template
            prompt = PromptTemplate(
                input_variables=["reviews"],
                template=(
                    "Summarize the following customer reviews into 3-4 concise sentences:\n\n"
                    "{reviews}\n\n"
                    "Provide a clear and comprehensive summary."
                ),
            )

            # Create the chain
            chain = LLMChain(llm=llm, prompt=prompt)

            # Combine reviews into a single text
            combined_reviews = " ".join(reviews)

            # Run the chain to get the summary
            summary = chain.run({"reviews": combined_reviews})

            return summary.strip()
        except Exception as e:
            return f"An error occurred while summarizing reviews: {e}"

    def convert_topics_to_dict(self, topics_str):
        """Convert the positive topics string into a dictionary."""
        topic_dict = {}
        items = topics_str.split('\n')

        for item in items:
            item = item.strip()  # Remove extra spaces

            # Skip empty lines or malformed lines
            if not item or '-' not in item:
                continue

            # Ensure each topic line contains only one "-" to separate label and percentage
            parts = item.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid format for topic item: {item}")

            label, value = parts

            # Clean and convert the value to a float
            value = value.strip().replace('%', '')
            if not value.replace('.', '', 1).isdigit():  # Allow for decimal percentages
                raise ValueError(f"Invalid percentage value for {label.strip()}: {value}")

            topic_dict[label.strip()] = float(value)

        return topic_dict

    def convert_ratings_to_dict(self, ratings_str):
        """Convert a ratings percentage string into a dictionary."""
        ratings_dict = {}
        items = ratings_str.split(',')

        for item in items:
            # Check if there is a "-" in the item
            if '-' not in item:
                raise ValueError(f"Invalid format for ratings item: {item}")

            label, value = item.split('-')

            # Check if we have a value and it is in percentage format
            value = value.strip().replace('%', '')
            if not value.isnumeric():
                raise ValueError(f"Invalid percentage value for {label.strip()}: {value}")

            ratings_dict[label.strip()] = float(value)

        return ratings_dict

    # def generate_plots_and_summary(self,ratings_percentage, positive_topics, negative_topics, summary_text):

    #     # Create a figure with a grid layout (2 rows, 2 columns)
    #     fig, axs = plt.subplots(2, 2, figsize=(30, 20))
    #     axs = axs.ravel()  # Flatten the 2D array of axes to 1D for easy access

    #     # Pie chart for ratings percentage
    #     labels = list(ratings_percentage.keys())
    #     sizes = list(ratings_percentage.values())
    #     colors = ['#66b3ff', '#99ff99', '#ff6666']
    #     axs[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    #     axs[0].set_title('Ratings Percentage')

    #     # Bar chart for positive topics
    #     topics = list(positive_topics.keys())
    #     positive_percentages = list(positive_topics.values())
    #     bars = axs[1].barh(topics, positive_percentages, color='#4CAF50')
    #     axs[1].set_title('Positive Topics Percentage')
    #     axs[1].set_xlabel('Percentage (%)')

    #     # Display the labels on top of the bars for positive topics
    #     for bar in bars:
    #         axs[1].text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.1f}%', va='center', ha='left', color='black')

    #     # Bar chart for negative topics
    #     topics = list(negative_topics.keys())
    #     negative_percentages = list(negative_topics.values())
    #     bars = axs[2].barh(topics, negative_percentages, color='#ff6666')
    #     axs[2].set_title('Negative Topics Percentage')
    #     axs[2].set_xlabel('Percentage (%)')

    #     # Display the labels on top of the bars for negative topics
    #     for bar in bars:
    #         axs[2].text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.1f}%', va='center', ha='left', color='black')

    #     # Text box for the summary
    #     axs[3].axis('off')
    #     axs[3].set_title('Summary')# Hide the 4th subplot axis
    #     axs[3].text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center', wrap=True, bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=1'))

    #     # Adjust the layout to avoid overlap
    #     plt.tight_layout()

    #     # Save the final figure as a .jpg file
    #     plt.savefig('G:/Ashwini/reapdat/290125/output/report_with_charts.jpg', format='jpg')

    #     # Display the plot
    #     plt.show()

    def generate_plots_and_summary(self, ratings_percentage, positive_topics, negative_topics, summary_text):
        # Create a figure with a grid layout (2 rows, 2 columns)
        fig, axs = plt.subplots(2, 2, figsize=(35, 25))  # Increased figure size for better clarity
        axs = axs.ravel()  # Flatten the 2D array of axes to 1D for easy access

        # Pie chart for ratings percentage
        labels = list(ratings_percentage.keys())
        sizes = list(ratings_percentage.values())
        colors = ['#66b3ff', '#99ff99', '#ff6666']
        axs[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 18})
        axs[0].set_title('Ratings Percentage', fontsize=22)

        # Bar chart for positive topics
        topics = list(positive_topics.keys())
        positive_percentages = list(positive_topics.values())
        bars = axs[1].barh(topics, positive_percentages, color='#4CAF50')
        axs[1].set_title('Positive Topics Percentage', fontsize=22)
        axs[1].set_xlabel('Percentage (%)', fontsize=18)
        axs[1].tick_params(axis='y', labelsize=16)  # Increase font size of y-axis labels

        # Display the labels on top of the bars for positive topics
        for bar in bars:
            axs[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.1f}%',
                        va='center', ha='left', color='black', fontsize=16, fontweight='bold')

        # Bar chart for negative topics
        topics = list(negative_topics.keys())
        negative_percentages = list(negative_topics.values())
        bars = axs[2].barh(topics, negative_percentages, color='#ff6666')
        axs[2].set_title('Negative Topics Percentage', fontsize=22)
        axs[2].set_xlabel('Percentage (%)', fontsize=18)
        axs[2].tick_params(axis='y', labelsize=16)  # Increase font size of y-axis labels

        # Display the labels on top of the bars for negative topics
        for bar in bars:
            axs[2].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.1f}%',
                        va='center', ha='left', color='black', fontsize=16, fontweight='bold')

        # Text box for the summary
        axs[3].axis('off')  # Hide axis for the summary box
        axs[3].set_title('Summary', fontsize=22, fontweight='bold')  # Title for the summary box
        axs[3].text(0.5, 0.5, summary_text, fontsize=40, ha='center', va='center', wrap=True,
                    bbox=dict(facecolor='lightgray', alpha=0.8, edgecolor='black', linewidth=2))

        # Adjust the layout to avoid overlap
        plt.tight_layout()

        # Save the final figure as a .jpg file
        plt.savefig('C:/Users/Mani_Moon/reapdat/vectorestore/output/report_with_charts.jpg', format='jpg', dpi=300)

        # Display the plot
        plt.show()








