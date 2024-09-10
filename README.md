# Использование языковых моделей для постановки задач

**Контекст**

Вы работаете над проектом [etoai.ru](http://etoai.ru/), который анализирует диалоги между клиентами и менеджерами и создает задачи в CRM для поддержания договоренностей с клиентами.

Недавно поступили жалобы от пользователей, что задачи формулируются неправильно. Проблема локализована на уровне LLM-модели: необходимо доработать используемый промт.

## Описание подхода

### Этап 1.
На данном этапе нужно собрать обратную связь от пользователей системы для того, чтобы понять в чем выражается "неправильность" задач, которые сгенерировала модель. Например, неправильность может быть связана с тем, что модель указывает неправильный порядок действий или ставит в задачу те действия, которые менеджер уже выполнил во время диалога с клиентом. Понимание сути ошибок критично в формулировки инструкции для LLM.

### Этап 2.
Воспроизведу базовую обработку информации из даилога, которая будет включать обработку текста диалога и генерацию задания на основе обработанного текста.

Базовая обработка в репозитории выглядит следующим образом:

```bash
│   README.md
│   utils.py - файл со вспомогательными функциями
│   main.py - основной файл для генерации, csv --> csv
|
├───data
│       test_dialogs_dataset - test_dialogs_dataset.csv - исходный файл
|       reworked.csv - файл с генерациями от языковой модели

```
Сделаю предположение, что все транскрипции диалогов будут в формате представленного файла data/test_dialogs_dataset - test_dialogs_dataset.csv. Для обработки данных из файлов такого формата добавлю функцию ```dialog_rework``` в utils, которая:
- удаляет все знаки кроме кириллицы
- удаляет эмодзи
- оставляет диалог в формате str "Менеджер: фраза, Клиент: фраза и т.д"

Полученная после обработки сторока будет использована при генерации моделью.

Для генерации буду пользоваться фреймворком LangChain, так как он очень хорошо приспособлен именно для различных экспериментов с промптами. Для загрузки моделей возьму библиотеку Fireworks, так как у меня нет ключа к OpenAI, а использование моделей с Fireworks аналогично использованию моделей с OpenAI и хотелось протестировать подход. Для перехода на модели OpenAI нужно заменить всего одну строчку с инициализацией модели на приведенную ниже:

```bash
from langchain.llms import OpenAI

# initialize the models
llm = OpenAI(
    model_name="text-davinci-003",
    openai_api_key="YOUR_API_KEY"
)
```
В качестве бейзлайна буду использовать подход few shot learning. 

Для генерации добавлю функцию ```generate_task``` в utils. Упралять результами генерации можно:

1. Меняя параметры генерации, такие как temperature и top_p
2. Меняя искомый ответ в примерах, используемых для Few Shot Learning
3. Меняя prefix для промта

### Этап 3. Проведение экспериментов и оценка качества.
Для оценки качества сгенерированных моделью задач можно воспользоваться 2-мя подходами:
- использовать user feedback, т.е. попросить пользователей проголосовать какие варианты задач лучше соответсвуют диалогу
- если уже есть данные из системы (записанные диалоги и поставленные на их основе задачи), то можно сравнивать сгенерированные моделью ответы и "идеальные", например с помощью косинусной близости между векторами предложений, полученных из трансофрмерной модели.

Так как у меня нет "иедальных" вариантов ответов, то я воспользуюсь собственным feedback для оценки результата. Буду экспериментировать с 3 предложениями из тестового датасета.

1. "Менеджер: Добрый день, Клиент: Здравствуйте Хотел бы узнать о новой акции, Менеджер: Конечно у нас есть отличное предложение"
2. "Менеджер: Здравствуйте чем могу помочь, Клиент: я хотел бы отменить свой заказ, Менеджер: Хорошо я помогу вам с этим"
3. "Клиент: Я ищу информацию о вашем продукте, Менеджер: Конечно Вот ссылка на наш сайт"

В экспериментах поменяю в базовом варианте параметры генерации, варианты ответов во fewshot и prefix как контекст. Остальные параметры будут базовыми.

Базовые параметры:

- temperature = 0.5
- top_p = 50

```bash
    examples = [
        {
            "query": "'Менеджер: Здравствуйте чем могу помочь',\
            'Клиент: я хотел бы заказать товар Д', \
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

```
```bash
    prefix = """The following are exerpts from conversations of company 
    managers and clients. You are an AI assistant whose role is to analyze 
    these dialogues and create tasks for managers based on the result of the 
    dialogues. Tasks are to be formulated in the consise and easy understandable way. 
    If the task consists of multiple steps, formulte it including all steps. 
    Translate the final answer into Russian language.  Do not display not translated informфtion. 
    Here are some examples in Russian language: 
    """
```

| Диалог | Параметры  | Результат   | Оценка   |
| :---:   | :---: | :---: |:---: |
| 1| Базовые  | Отправить клиенту предложение с акцией, через несколько дней уточнить какая реакция у клиента|OK |
| 2| Базовые  | Отменить заказ в системе, уведомить клиента об отмене заказа, уведомить продавца об отмене заказа| OK|
| 3| Базовые  | Отправить клиенту ссылку на сайт, через несколько дней уточнить какая реакция у клиента| OK|
| 1| temperature=1, top_p = 70  | Зафиксировать в журнале учета акции, проверить соответствие акции политике компании| не очень|
| 2| temperature=1, top_p = 70  | Отменить заказ клиента|не очень |
| 3| temperature=1, top_p = 70  | Для проверки актуальности информации на сайте создать отчет в Гугл analytics.|не очень |
| 1| Ответ в fewshot из 2-х слов  | Отправить информацию о новой акции |OK |
| 2| Ответ в fewshot из 2-х слов  | Отменить заказ| OK|
| 3| Ответ в fewshot из 2-х слов  | Добавить информацию о продукте на сайте| OK|
| 1| empty prefix | Отправить клиенту предложение с новыми акциями, через несколько дней уточнить какая реакция у клиента |OK |
| 2| empty prefix | Отменить заказ клиента, если это возможно, и уточнить причины отмены| OK|
| 3| empty prefix| Уточнить у клиента о продукте, с которым он интересуется и предоставить информацию о нём| OK|

Получается, что больше всего на финальный вариант влияет вариант few shot и temperature (ее слишком высокое значение близкое к 1 дает немного безумные идеи!!!). Prefix в виде контекста не сильно повлиял.
