# Проект анализа аномалий временных рядов для RedLab Hack

Этот репозиторий содержит код и материалы проекта, разработанного командой ikanam_chipi_chipi для RedLab Hack. Проект направлен на автоматическое обнаружение аномалий во временных рядах данных с применением статистических методов и машинного обучения. Мы решали кейс №1 от компании RedLab и заняли **3-е место**!

<img width="1630" alt="image" src="https://github.com/user-attachments/assets/428d6470-595a-498d-86f3-b57f0def1a7a" />

_Команда ikanam_chipi_chipi заняла 3-е место на RedLab Hack за проект "Анализ аномалий временных рядов"._

Наше решение использует комбинацию методов для выделения сезонности, прогнозирования временных рядов и обнаружения отклонений от прогноза.  Мы разработали веб-интерфейс для удобной визуализации и анализа результатов.

**Состав команды ikanam_chipi_chipi:**

*   Ляра Максим - Team Lead, ML-инженер
*   Сабирова Аделя - Аналитик, дизайнер

**Результаты**

Наше решение позволяет:

*   **Выделять сезонную компоненту:**  Изолировать сезонные колебания во временных рядах для упрощения анализа трендов и аномалий.
*   **Прогнозировать временные ряды:**  Использовать статистические методы и машинное обучение для предсказания будущих значений временного ряда.
*   **Обнаруживать аномалии:**  Автоматически выявлять точки данных, значительно отклоняющиеся от прогноза и/или исторической нормы.

**Пример работы с сезонностью:**

<img width="908" alt="image" src="https://github.com/user-attachments/assets/3902cb1e-5c51-4fcf-83fb-39cb00e772a0" />

**Таблица с предиктами и аномалиями:**

Модель генерирует таблицу с предсказанными значениями и флагом, указывающим на наличие аномалии (anomaly == True). Таблицу можно скачать в формате `.csv` прямо из веб-интерфейса.

<img width="908" alt="image" src="https://github.com/user-attachments/assets/60187693-9d55-40b4-8c5a-65f470520391" />

**Визуализации аномалий:**

Веб-интерфейс отображает временные ряды с выделенными аномальными точками, что позволяет быстро оценить проблемные участки данных.

**Веб-интерфейс**

Мы разработали веб-интерфейс с использованием Streamlit для удобного взаимодействия с моделью и визуализации результатов.  Интерфейс позволяет загружать данные, настраивать параметры анализа и просматривать отчеты.

<img width="1505" alt="image" src="https://github.com/user-attachments/assets/ba22a460-e34c-4c53-873c-218a86343655" />

**Основные возможности веб-интерфейса:**

*   Загрузка данных в формате CSV.
*   Настройка параметров модели (например, размер окна для обнаружения аномалий).
*   Визуализация временных рядов с выделенными аномалиями.
*   Экспорт результатов анализа в формате CSV.

**Запуск веб-интерфейса на локальном устройстве**

1.  **Клонируйте репозиторий:**
2.  **Создание и активация виртуального окружения:**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS:
    source venv/bin/activate
    ```
3.  **Установка необходимых библиотек:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Запуск веб-интерфейса:**
    ```bash
    streamlit run app.py
    ```

После выполнения этих шагов, веб-интерфейс будет запущен, и вы сможете получить к нему доступ через указанный в консоли адрес (обычно `http://localhost:8501`).

**Структура репозитория**

* `app.py`: Основной файл веб-приложения Streamlit.
* `requirements.txt`: Список зависимостей Python.
* `Интерфейс_26_05_2024.mp4`: Видеодемонстрация работы интерфейса (пример).
* `презентация_26_05_2024.pdf` / `презентация_26_05_2024.pptx`: Презентация проекта (пример).
