# Бот-наставник для технических собеседований (MVP)

## Запуск
1. Создай бота в Telegram через @BotFather и получи токен.
2. Получи бесплатный ключ LLM (Groq, Gemini или OpenRouter), см. .env.example.
3. Скопируй `.env.example` в `.env` и вставь токен и ключ.
4. Установи зависимости и запусти:

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python bot.py

## Команды
/task — задача · /hint — подсказка · /skip — другая задача · /daily — ежедневная рассылка · /stats — прогресс

## Что дальше
- Подписка и задачи по компаниям (отдельный список в tasks.json с полем company + проверка доступа)
- Оплата (ЮKassa / Telegram Payments), System Design-режим, больше задач
- Перенос на PostgreSQL и webhook вместо polling для продакшена
