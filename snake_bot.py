from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
import logging
import datetime
import threading
import snake

threads = {}
thread_starters = {}
runs = 0

updater = Updater(token="559384287:AAGt9y-Cyl7CmW_8nr4a3W0yL6bpbu9NHxM")
dispatcher = updater.dispatcher

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.ERROR)



def start_run(bot, update):
    input = update.message.text.split()
    mut_rate  =  float(input[1])
    mut_dev   =  float(input[2])
    max_gen   =  int(input[3])

    global runs
    name = "run_%i"%(runs)
    runs += 1
    
    thread = threading.Thread(name=name, target=run_thread, args=[bot, name, mut_rate, mut_dev, max_gen])
    threads[name] = thread
    thread_starters[name] = update.message.chat_id
    thread.run()


start_handler = CommandHandler("start_run", start_run)
dispatcher.add_handler(start_handler)


def run_thread(bot, thread_name, mut_rate, mut_dev, max_gen):

    chat_id = thread_starters[thread_name]

    bot.send_message(chat_id=chat_id, text="Starting thread %s "%(thread_name))

    snake.start_run(mut_rate, mut_dev, max_gen, "./data/%s/"%thread_name)

    bot.send_message(chat_id=chat_id, text="Finished thread %s "%(thread_name))



logging.info("starting bot")

updater.start_polling()
updater.idle()

logging.info("bot has stopped")