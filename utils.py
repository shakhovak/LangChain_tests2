from langchain_fireworks import Fireworks
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
import re

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U0001F1F2"
    "\U0001F1F4"
    "\U0001F620"
    "\u200d"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"  # dingbats
    "\u3030"
    "\U00002500-\U00002BEF"  # Chinese char
    "\U00010000-\U0010ffff"
    "]+",
    flags=re.UNICODE,
)


def dialog_rework(text):
    """function to rework dialogues under assumptions that 
    thy will have similar format after transcription"""
    dialogue_reworked = []
    text = re.sub(r"\[.*?\]", "", text)
    text = emoji_pattern.subn(r"", text)[0]
    text = text.replace("Менеджер:", "\nМенеджер:").replace("Клиент:", "\nКлиент:")
    text_items = text.split("\n")
    for utterance in text_items:
        if len(utterance) > 1:
            temp = utterance.split(":")
            if len(temp) > 1:
                text_reworked = re.sub(r"[0-9]+", "", temp[1])
                text_reworked = re.sub(r"[^\w\s]", "", text_reworked)
                fin_string = temp[0] + ": " + text_reworked.strip()
                dialogue_reworked.append(fin_string)
    fin_dialogue = ", ".join(dialogue_reworked)
    return fin_dialogue


def generate_task(query, model_name, fireworks_api_key, temperature, top_k):
    """function to generate task from scripts using models"""

    llm = Fireworks(
        model=model_name,
        fireworks_api_key=fireworks_api_key,
        temperature=temperature,
        max_tokens=50,
        top_k=top_k,
    )

    examples = [
        {
            "query": "'Менеджер: Здравствуйте чем могу помочь',\
            'Клиент: Я хотел бы заказать товар Д', \
            'Менеджер: Я рад помочь в этом вопросе'",
            "answer": "Уточнить у клиента параметра заказа, проверить размещение заказа в системе",
        },
        {
            "query": "'Менеджер: Добрый день',\
            'Клиент: Здравствуйте Хотел бы узнать о новой продуктах в линейке В',\
            'Менеджер: Конечно я рад буду отправить вам наше предложение'",
            "answer": "Отправить клиенту предложение с продуктами В, через несколько дней уточнить какая реакция у клиента",
        },
    ]

    example_template = """
        User: {query}
        AI: {answer}
        """

    example_prompt = PromptTemplate(
        input_variables=["query", "answer"], template=example_template
    )

    prefix = """    """

    suffix = """
    User: {query}
    AI: """

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n",
    )

    final_prompt = few_shot_prompt_template.format(query=query)

    answer = llm.invoke(final_prompt)
    answer = answer.split("\n\n")[0].strip().replace('"""', "").replace("\n", "")
    return answer
