from utils import llm_invoke


class PageSummaries():
    def __init__(self, src_language, tgt_language, llm_client=None, src_summary_prompt: str=None, tgt_summary_prompt: str=None, src_summaries: list=None, tgt_summaries: list=None, encoder=None, temperature=0.7):
        self.client = llm_client
        self.src_language = src_language
        self.tgt_language = tgt_language
        if src_summary_prompt:
            self.src_summary_prompt = src_summary_prompt
        else:
            with open(f"./prompts/write_page_summary_short_{src_language}.txt", "r") as f1:
                self.src_summary_prompt = f1.read()
        if tgt_summary_prompt:
            self.tgt_summary_prompt = tgt_summary_prompt
        else:
            with open(f"./prompts/write_page_summary_short_{tgt_language}.txt", "r") as f2:
                self.tgt_summary_prompt = f2.read()

        self.src_summaries = src_summaries if src_summaries is not None else []
        self.tgt_summaries = tgt_summaries if tgt_summaries is not None else []
        self.encoder = encoder
        if self.encoder is not None:
            self.summary_embeddings = [self.encoder.encode(page_summary) for page_summary in self.src_summaries]
        else:
            self.summary_embeddings = []
        self.temperature = temperature
    
    def to_dict(self):
        return {
            "src_language": self.src_language,
            "tgt_language": self.tgt_language,
            "src_summary_prompt": self.src_summary_prompt,
            "tgt_summary_prompt": self.tgt_summary_prompt,
            "src_summaries": self.src_summaries,
            "tgt_summaries": self.tgt_summaries,
            "temperature": self.temperature,
        }
    
    @classmethod
    def from_dict(cls, data, client=None, encoder=None):
        page_summaries = cls(
            src_language=data["src_language"],
            tgt_language=data["tgt_language"],
            llm_client=client,
            src_summary_prompt=data["src_summary_prompt"],
            tgt_summary_prompt=data["tgt_summary_prompt"],
            src_summaries=data["src_summaries"],
            tgt_summaries=data['tgt_summaries'],
            encoder=encoder,
            temperature=data["temperature"]
        )
        return page_summaries

    def update_summary(self, src_page_text: str, tgt_page_text: str, page_id: int) -> str:
        src_prompt = self.src_summary_prompt.format(text=src_page_text)
        generated_src_summary = llm_invoke(client=self.client, messages=src_prompt, temperature=self.temperature)['content']
        assert len(self.src_summaries) == len(self.summary_embeddings) == page_id
        self.src_summaries.append(generated_src_summary)
        self.summary_embeddings.append(self.encoder.encode(generated_src_summary))
        
        return generated_src_summary
    
    
    def retrieve_summaries(self, query) -> list[str]:
        query_embedding = self.encoder.encode(query)
        similarities = self.encoder.similarity(query_embedding, self.summary_embeddings)
        return similarities
