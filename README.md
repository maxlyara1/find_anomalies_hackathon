# Как запустить веб-интерфейс на локальном устройстве?
### Ввести следующие команды в консоль (при первом запуске все, далее - при повторных запусках - выполнить только 4 шаг, соответственно модель сможет работать автономно т.к. все библиотеки ранее уже были загружены):
1. python -m venv venv
2. * venv\Scripts\activate - для Windows
   * source venv/bin/activate - для Макбука
3. pip install -r requirements.txt
4. streamlit run app.py

# Пояснение к файлам:
* app.py - файл, который запускает веб-интерфейс, а также содержит всё решение
* requirements.txt - файл с библиотеками и их версиями, которые необходимо установить
* Интерфейс_26_05_2024.mp4 - видео с визуализацией интерфейса и объяснениями
* презентация_26_05_2024.pdf, презентация_26_05_2024.pptx - презентация нашего решения

# Результат работы модели:
* Как пример указано то, как велась работа с сезонностью - выделение из обычной метрики сезонной компоненты для её дальнейшего удаления
* Таблица с предиктами, где колонка anomaly==True - аномалия, anomaly==False - нормальный объект. Её можно загрузить напрямую из веб интерфейса в .csv формате.
* Визуализации аномалий
