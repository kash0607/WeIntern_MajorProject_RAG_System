from langchain_community.document_loaders import PyPDFLoader
import os


class PDFLoader:
    def __init__(self, pdf_directory):
        self.pdf_directory = pdf_directory

    def load_documents(self):
        documents = []

        for file in os.listdir(self.pdf_directory):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_directory, file)

                loader = PyPDFLoader(pdf_path)
                docs = loader.load()

                print(f"Loaded {file} ({len(docs)} pages)")

                documents.extend(docs)

        return documents