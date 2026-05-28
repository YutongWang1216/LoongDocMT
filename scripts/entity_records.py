from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json
from copy import deepcopy
import re
from utils import llm_invoke
from logger import get_logger
import os


logger = get_logger(__name__)

classify_prompt_template = '''Given a text passage and a specified entity, classify the entity into one of these categories: Character, Organization, Location, Event, Object, or Other. Only output the category name.

<Text passage>
{text}

<Entity>
{entity}
'''

write_prompt_template = '''Given a text passage and a specified entity, summarize the relevant information about the entity including the following items:
{info_items}

<Text passage>
{text}

<Entity>
{entity}


The output should be a Markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```", and without any comments:

{format_instruction}
'''

update_prompt_template = '''Given a text passage and a specified entity, update the existing information about this entity including the following items:
{info_items}

<Text passage>
{text}

<Entity>
{entity}

<Existing Information>
{exist_info}

{format_instruction}
'''

info_key_map = {
    "character": ["Role", "Description", "Relationships", "Motivation/Goals", "Development"],
    "organization": ["Type", "Purpose", "Members", "Location", "Significance"],
    "location": ["Type", "Description", "Inhabitants", "Events", "Symbolism"],
    "event": ["Title", "Description", "Participants", "Location", "Consequences", "Timeline"],
    "object": ["Type", "Appearance", "Purpose", "Owner/Creator", "Significance"],
    "other": ["Label", "Type", "Description", "Significance", "Interaction", "Impact"]
}

info_item_map = {
    "character": '''
Role: Their role in the story (e.g., protagonist, antagonist, supporting).
Description: Physical and personality traits.
Relationships: Connections with other characters.
Motivation/Goals: What drives the character’s actions.
Development: Key changes or growth throughout the story.''',
    "organization": '''
Type: Nature of the organization (e.g., company, guild, secret society).
Purpose: Its mission, goals, or primary function.
Members: Key individuals or groups associated with it.
Location: Headquarters or base of operations.
Significance: Its role and impact on the story.''',
    "location": '''
Type: The nature of the location (e.g., city, forest, building).
Description: Key physical and atmospheric details.
Inhabitants: Who or what lives there.
Events: Significant events that happen in this location.
Symbolism: Any symbolic meaning or importance.''',
    "event": '''
Description: What happens during the event.
Participants: Characters or groups involved.
Location: Where the event occurs.
Consequences: Effects or outcomes of the event.
Timeline: When it takes place within the story.''',
    "object": '''
Type: Its category (e.g., weapon, artifact, document).
Appearance: Distinctive physical features.
Purpose: Its function or intended use.
Owner/Creator: Who owns or made it.
Significance: Why it is important in the story.''',
    "other": '''
Type: The nature or category of the entity (e.g., concept, symbol, supernatural phenomenon, custom).
Description: A brief explanation or definition of what it is.
Significance: Why it matters or how it influences the story (e.g., central to a theme, plot device, or character development).
Interaction: How it interacts with characters, locations, or events.
Impact: The role or consequences it has on the plot or characters.''',
}


class DelayRecord():
    def __init__(self, src_name: str, tgt_name: str, entity_type: str = None, cur_data: dict = None, last_update: int = 0):
        self.src_name = src_name
        self.tgt_name = tgt_name
        self.entity_type = entity_type
        self.cur_data = cur_data
        # self.cur_description = cur_description
        self.last_update = last_update
        
    def to_dict(self):
        return {
            "src_name": self.src_name,
            "tgt_name": self.tgt_name,
            "entity_type": self.entity_type,
            "cur_data": self.cur_data,
            "last_update": self.last_update
        }
        
    @classmethod
    def from_dict(cls, data):
        return cls(
            src_name=data["src_name"],
            tgt_name=data["tgt_name"],
            entity_type=data["entity_type"],
            cur_data=data["cur_data"],
            last_update=data["last_update"]
        )


class EntityRecords():
    def __init__(self, src_language, tgt_language, src_lines=None, llm_client=None, update_prompt: str=None, records: dict[str, DelayRecord]=None, temperature=0.7):
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.src_lines = src_lines if src_lines is not None else []
        if update_prompt:
            self.update_prompt= update_prompt
        else:
            with open("./prompts/update_entity_records.txt", "r") as f:
                self.update_prompt = f.read()
        self.records: dict[str, DelayRecord] = records if records else {}
        self.client = llm_client
        self.temperature = temperature

    def to_dict(self):
        return {
            "src_language": self.src_language,
            "tgt_language": self.tgt_language,
            "src_lines": self.src_lines,
            "update_prompt": self.update_prompt,
            "records": {entity: self.records[entity].to_dict() for entity in self.records},
            "temperature": self.temperature
        }
    
    @classmethod
    def from_dict(cls, data, client=None):
        entity_records = cls(
            src_language=data["src_language"],
            tgt_language=data["tgt_language"],
            src_lines=data.get("src_lines", []),
            llm_client=client,
            update_prompt=data["update_prompt"],
            records=data["records"],
            temperature=data["temperature"]
        )
        return entity_records

    def update_record(self, src_lines: list[str], src_page_text: str, tgt_page_text: str) -> str:
        logger.debug("EntityRecords update_record called.")
        
        self.src_lines.extend(src_lines)
        
        prompt = f'Given the following text, identify all the key proper noun entities mentioned in this text. Proper noun entities consist of six categories: character, organization, location, event, object, and other. Only select the most important entities and output their of the entities, separated by commas. Do not include any additional information or explanations.\n\n<text> {src_page_text}'

        response = llm_invoke(client=self.client, messages=[{"role": "user", "content": prompt}], temperature=self.temperature)['content']
        entity_list = [name.strip() for name in re.split(r'[,，]', response) if name.strip()]

        entity_map = {}
        
        for entity in entity_list:
            prompt = f'Given an {self.src_language} text with its corresponding {self.tgt_language} translation and a proper noun entity writen in {self.src_language}, find the {self.tgt_language} name of this entity in the {self.tgt_language} translation. Note that your response should include only the {self.tgt_language} name without any additional information. If no corresponding name is found, just return "N/A".\n\n<{self.src_language} text> {src_page_text}\n\n<{self.tgt_language} text> {tgt_page_text}\n\n<{self.src_language} entity> {entity}'

            recover_messages = [
                {'role': 'user', 'content': prompt}
            ]

            response = llm_invoke(client=self.client, messages=recover_messages, temperature=self.temperature, call_by_entity=True)['content'].strip()  # 增加 call_by_entity 参数，便于在 llm_invoke 中设置 frequency_penalty
            entity_map[entity] = response

        print("#" * 10, "Entity extraction", "#" * 10)
        print(', '.join([f'{name} - {entity_map[name]}' for name in entity_map]))
        
        for entity_name in entity_map:
            if entity_name not in self.records:
                new_record = DelayRecord(src_name=entity_name, tgt_name=entity_map[entity_name])
                self.records[entity_name] = new_record
                logger.debug(f"New entity added: {new_record.src_name} -> {new_record.tgt_name}: {new_record.last_update} / {len(self.src_lines)}")
            else:
                logger.debug(f"Entity already exists: {self.records[entity_name].src_name} -> {self.records[entity_name].tgt_name}: {self.records[entity_name].last_update} / {len(self.src_lines)}")
        
    def get_records(self, src_page_text: str) -> list:
        logger.debug("EntityRecords get_records called.")
        
        existed_records: list[DelayRecord] = []
        for entity in self.records:
            if entity not in src_page_text:
                continue
            
            record = self.records[entity]
            logger.debug(f"Processing entity: {record.src_name} -> {record.tgt_name}: {record.last_update} / {len(self.src_lines)}")
            
            if record.last_update == len(self.src_lines) and record.cur_data is not None and record.entity_type is not None:
                logger.debug(f"Entity {entity} is already up-to-date.")
                existed_records.append(record)
                continue
            classify_prompt = classify_prompt_template.format(text=src_page_text, entity=entity)
            attempt_cnt = 0
            while True:
                if attempt_cnt > 5:
                    entity_type = 'other'
                    break
                try:
                    attempt_cnt += 1
                    messages = [
                        {"role": "user", "content": classify_prompt}
                    ]
                    entity_type = llm_invoke(client=self.client, messages=messages, temperature=self.temperature)["content"].strip().lower()
                    if entity_type in info_key_map:
                        break
                    else:
                        found_flag = False
                        for key in info_key_map:
                            logger.debug(f"Checking key: {key}")
                            if key in entity_type:
                                entity_type = key
                                found_flag = True
                                break
                        if found_flag:
                            break
                except Exception as e:
                    print(e)
            
            ans_schema = [ResponseSchema(name=key, type="string", description="a string") for key in info_key_map[entity_type]]
            json_parser = StructuredOutputParser.from_response_schemas(ans_schema)
            json_output_instructions = json_parser.get_format_instructions(only_json=True) + "\nIf an entry has no corresponding content, just fill in \"N/A\"."

            if record.cur_data is None:
                logger.debug(f"Entity {entity} is new, creating new record.")
                prompt = write_prompt_template.format(
                    info_items=info_item_map[entity_type],
                    text=self.src_lines,
                    entity=entity,
                    format_instruction=json_output_instructions
                )
            else:
                logger.debug(f"Entity {entity} exists, updating record.")
                prompt = update_prompt_template.format(
                    info_items=info_item_map[entity_type],
                    text=self.src_lines[record.last_update:],
                    entity=entity,
                    exist_info=json.dumps(record.cur_data, ensure_ascii=False),
                    format_instruction=json_output_instructions
                )

            max_attempts = 10
            attempt_cnt = 0
            while True:
                if attempt_cnt >= max_attempts:
                    print(f"Failed to update entity {entity} after {max_attempts} attempts.")
                    new_data = None
                    break
                try:
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    llm_response = llm_invoke(client=self.client, messages=messages, temperature=self.temperature)["content"]
                    new_data = json_parser.parse(llm_response)
                    break
                except Exception as e:
                    print('### In exception')
                    print(e)
                    print(llm_response)
                    attempt_cnt += 1
                    print(f"Retrying update records with attempt {attempt_cnt+1}/{max_attempts}...")
            if new_data is None:
                continue
            record.entity_type = entity_type
            record.cur_data = new_data
            record.last_update = len(self.src_lines)
            existed_records.append(self.records[entity])
            logger.debug(f"Entity {entity} updated: {record.src_name} -> {record.tgt_name}: {record.last_update} / {len(self.src_lines)}")
            logger.debug(f"Entity {entity} type: {record.entity_type}")
            logger.debug(f"Entity {entity} data:\n{json.dumps(record.cur_data, ensure_ascii=False, indent=2)}")
        
        info_to_return = []
        for record in existed_records:
            tldr = self.gen_tldr(record.src_name, src_page_text)
            info_to_return.append((record.src_name, record.tgt_name, tldr))
        return info_to_return


    def get_record(self, entity_name: str) -> str:
        obtained_item = self.records.get(entity_name, None)
        return obtained_item
    
    def gen_tldr(self, entity_name, page_text):
        logger.debug(f"EntityRecords gen_tldr called for entity: {entity_name}")

        ent_record = self.records[entity_name]
        data_str = json.dumps(ent_record.cur_data, ensure_ascii=False)

        prompt = f'You are translating an {self.src_language} text. Now, given a JSON-formatted record containing information about an entity mentioned in the text, summarize the details you find useful for the translation task in one concise sentence. Note that your response should include only the summary sentence without any additional information.\n\n<{self.src_language} text> {page_text}\n\n<Entity information> {data_str}'

        messages = [
            {'role': 'user', 'content': prompt}
        ]

        entity_tldr = llm_invoke(client=self.client, messages=messages, temperature=self.temperature)['content']

        return entity_tldr
