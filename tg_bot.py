import telebot
from telebot import apihelper
import requests

# tb = telebot.AsyncTeleBot("7086093017:AAHL4mTN12QMbCz6rveXLx_RCy-M7lKn-3w")
bot = telebot.TeleBot("7086093017:AAHL4mTN12QMbCz6rveXLx_RCy-M7lKn-3w")


url = 'http://0.0.0.0:8000/'
#
#
# @bot.message_handler(commands=['start'])
# def start_message(message):
# 	bot.send_message(message.chat.id, 'Пришли мне Emoji, который необходимо сделать размытым.')
#
#
# @bot.message_handler(content_types=['text'])
# def send_text(message):
# 	if message.text.lower() == 'нам первый смайлик':
# 		img = open('Смайлики и люди 1.png', 'rb')
#
# 		bot.send_document(message.chat.id, img)
# 	elif message.text.lower() == 'наш второй смайлик':
# 		img = open('Смайлики и люди 2.png', 'rb')
# 		bot.send_document(message.chat.id, img)
#
# 	else:
# 		bot.send_message(message.chat.id, 'Прости, но пока у меня нет этих Emoji')


# bot.polling()
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Пример персонального ChatBOT в ТГ")


@bot.message_handler(func=lambda m: True)
def echo_all(message):
    messageObj = {'message': message.text}

    print(messageObj)

    resp = requests.post(url, json=messageObj)
    print(resp.text)
    mes = str(resp.json()['message'])
    print(mes)
    print(resp.content)

    if(mes == ''):
        bot.reply_to(message, "Пожалуйста, задайте вопрос!")
        return

    bot.reply_to(message, mes)

bot.polling()
# @tb.message_handler(commands=['start'])
# async def start_message(message):
#
# 	await bot.send_message(message.chat.id, 'Hello!')

# x = requests.post(url, json = myobj)
#
# print(x.text)
