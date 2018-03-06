from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
import logging
import datetime
import snake

updater = Updater(token="559384287:AAGt9y-Cyl7CmW_8nr4a3W0yL6bpbu9NHxM")
dispatcher = updater.dispatcher

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)



def dienste(bot, update):
    dienste_file = open("dienste.txt")
    text = ""

    days = ( datetime.date.today() - init_date ).days
    weeks = days // 7
    
    for person in init_dienste:
        name = person[0]
        dienst = dienste_dict[(person[1] + weeks)%3]
        text += name + " ist dran mit " + dienst + "\n"
    
    bot.send_message(chat_id=update.message.chat_id, text=text)

dienste_handler = CommandHandler("dienste", dienste)
dispatcher.add_handler(dienste_handler)


def repeat_run(bot, update):
    input = update.message.text.split()
    mut_rate  =  input[1]
    mut_dev   =  input[2]
    max_gen   =  input[3]

    snake.start_run(mut_rate, mut_dev, max_gen)

repeat_handler = CommandHandler("repeat_run", repeat_run)
dispatcher.add_handler(repeat_handler)







logging.info("starting bot")

updater.start_polling()
updater.idle()

logging.info("bot has stopped")