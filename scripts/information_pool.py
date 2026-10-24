from entity_records import EntityRecords
from page_summaries import PageSummaries
from sentence_pairs import SentencePairs
from copy import deepcopy
from langcodes import Language


class InformationPool():
    def __init__(self,
        src_lang,
        tgt_lang,
        llm_client=None,
        source_pages = None,
        target_pages = None,
        comet_scores = None,
        reference_pages = None, 
        current_page_id=0,
        encoder=None,
        temperature=0.7
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_language = Language.make(language=src_lang).display_name()
        self.tgt_language = Language.make(language=tgt_lang).display_name()
        self.client = llm_client
        self.page_summaries = PageSummaries(src_language=self.src_language, tgt_language=self.tgt_language, llm_client=llm_client, encoder=encoder)
        self.entity_records = EntityRecords(src_language=self.src_language, tgt_language=self.tgt_language, llm_client=llm_client)
        self.sentence_pairs = SentencePairs(encoder=encoder, src_lang=src_lang)
        self.source_pages = source_pages if source_pages is not None else []
        self.target_pages = target_pages if target_pages is not None else []
        self.comet_scores = comet_scores if comet_scores is not None else []
        self.reference_pages = reference_pages if reference_pages is not None else []
        self.current_page_id = current_page_id
        self.encoder = encoder
        self.temperature = temperature
    
    def to_dict(self):
        return {
            "page_summaries": self.page_summaries.to_dict(),
            "entity_records": self.entity_records.to_dict(),
            "sentence_pairs": self.sentence_pairs.to_dict(),
            "source_pages": self.source_pages,
            "target_pages": self.target_pages,
            "comet_scores": self.comet_scores,
            "reference_pages": self.reference_pages,
            "current_page_id": self.current_page_id,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "src_language": self.src_language,
            "tgt_langugae": self.tgt_language,
            "temperature": self.temperature,
        }
    
    @classmethod
    def from_dict(cls, data, client=None, encoder=None):
        info_pool = cls(
            src_lang = data["src_lang"],
            tgt_lang = data["tgt_lang"],
            llm_client=client,
            source_pages = data["source_pages"],
            target_pages = data["target_pages"],
            comet_scores = data["comet_scores"],
            reference_pages = data["reference_pages"],
            current_page_id = data["current_page_id"],
            encoder=encoder,
            temperature=data["temperature"]
        )
        # info_pool.client = client
        info_pool.page_summaries = PageSummaries.from_dict(data=data["page_summaries"], client=client, encoder=encoder)
        info_pool.entity_records = EntityRecords.from_dict(data=data["entity_records"])
        info_pool.sentence_pairs = SentencePairs.from_dict(data=data["sentence_pairs"], encoder=encoder)
        return info_pool

    def __deepcopy__(self, memo):
        new_info_pool = InformationPool(
            self.src_lang, 
            self.tgt_lang, 
            self.client, 
            source_pages=deepcopy(self.source_pages), 
            target_pages=deepcopy(self.target_pages), 
            reference_pages=deepcopy(self.reference_pages),
            comet_scores=deepcopy(self.comet_scores),
            current_page_id=deepcopy(self.current_page_id),
            encoder=self.page_summaries.encoder,
        )
        new_info_pool.page_summaries = self.page_summaries
        new_info_pool.page_summaries.src_summaries = deepcopy(self.page_summaries.src_summaries)
        new_info_pool.page_summaries.tgt_summaries = deepcopy(self.page_summaries.tgt_summaries)
        new_info_pool.entity_records = self.entity_records
        new_info_pool.entity_records.records = deepcopy(self.entity_records.records)
        return new_info_pool
    