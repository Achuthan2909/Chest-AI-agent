from smolagents import OpenAIServerModel, CodeAgent
from datetime import datetime
from typing import Dict, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

class ChestXRayAgent:
    def __init__(self):
        # Initialize the reasoning model (using Qwen for better reasoning)
        self.reasoning_model = OpenAIServerModel(
            model_id="qwen2.5:14b-instruct-8k",
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.reasoner = CodeAgent(tools=[], model=self.reasoning_model, add_base_tools=False, max_steps=2)
        
        # Initialize vector store and embeddings for RAG
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
        self.vectordb = Chroma(persist_directory=db_dir, embedding_function=self.embeddings)

    def _get_medical_context(self, diagnosis: str) -> str:
        """Retrieve relevant medical literature context from the vector database."""
        query = f"What does medical literature say about {diagnosis} findings in chest X-rays?"
        docs = self.vectordb.similarity_search(query, k=3)
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_report(self, 
                       diagnosis: str,
                       activations: Dict[str, float],
                       confidence: float) -> str:
        """
        Generate a detailed radiology report using LLM reasoning and RAG.
        
        Args:
            diagnosis (str): The predicted diagnosis
            activations (dict): Dictionary of region activations
            confidence (float): Confidence score for the diagnosis
            
        Returns:
            str: Formatted radiology report
        """
        # Get medical context from vector database
        medical_context = self._get_medical_context(diagnosis)
        
        # Format activation data for the prompt
        activation_text = "\n".join([
            f"{region}: {score:.3f}" if not isinstance(score, float) or score == score
            else f"{region}: Not applicable"
            for region, score in activations.items()
        ])

        # Create the prompt for the LLM
        prompt = f"""You are an expert radiologist analyzing a chest X-ray using AI-generated Grad-CAM activation maps.
        Please provide a detailed radiology report based on the following information:

        Diagnosis: {diagnosis}
        Confidence Score: {confidence:.2%}

        Grad-CAM Activation Map:
        {activation_text}

        Relevant Medical Literature:
        {medical_context}

        Please provide a detailed report that includes:
        1. Primary findings and diagnosis
        2. Analysis of the activation patterns and their clinical significance
        3. Interpretation of which regions are most important for the diagnosis
        4. Correlation with medical literature findings
        5. Clinical recommendations
        6. Any limitations or considerations

        Format the report professionally and include relevant medical terminology.
        Make sure to reference and incorporate insights from the provided medical literature.
        """

        # Get the report from the LLM
        report = self.reasoner.run(prompt, reset=False)
        
        # Add metadata to the report
        final_report = f"""
        **CHEST X-RAY ANALYSIS REPORT**  📄
        -------------------------------------
        **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        {report}
        
        *This report was generated using AI-based analysis and should be reviewed by a medical professional.*
        """
        
        return final_report