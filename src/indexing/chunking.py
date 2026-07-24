from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


class Chunking:

    def fixed_chunking(self, documents):
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = splitter.split_documents(documents)

        print(f"\nFixed Chunking: {len(chunks)} chunks created")

        return chunks

    def recursive_chunking(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = splitter.split_documents(documents)

        print(f"\nRecursive Chunking: {len(chunks)} chunks created")

        return chunks