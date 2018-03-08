from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
import logging
import datetime
from multiprocessing import Process
import snake
import pickle
from collections import namedtuple
import os
import shutil


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

updater = Updater(token="559384287:AAGt9y-Cyl7CmW_8nr4a3W0yL6bpbu9NHxM")
dispatcher = updater.dispatcher


def save_runs():
    datei = open("./data/runs.obj", "wb")
    pickle.dump(runs,datei)


def start_run(bot, update):

    input = update.message.text.split()
    mut_rate  =  float(input[1])
    mut_dev   =  float(input[2])
    max_gen   =  int(input[3])

    global n_runs
    run_name = "run_%i"%(n_runs)
    n_runs += 1


    thread = Process(name=run_name, target=run_thread, args=(bot, update, run_name, mut_rate, mut_dev, max_gen))

    run = Run(run_name, mut_rate, mut_dev, max_gen)
    runs[run_name] = run
    save_runs()

    threads[thread_name] = thread

    thread.start()

start_handler = CommandHandler("start_run", start_run)
dispatcher.add_handler(start_handler)



def run_thread(bot, update, thread_name, mut_rate, mut_dev, max_gen):


    bot.send_message(chat_id=update.message.chat_id, text="Starting thread %s "%(thread_name))
    logging.info("Starting thread %s "%(thread_name))

    snake.start_run(mut_rate, mut_dev, max_gen, "./data/%s/"%thread_name)

    logging.info("Finished thread %s "%(thread_name))
    bot.send_message(chat_id=update.message.chat_id, text="Finished thread %s "%(thread_name))




def get_active(bot, update):
    chat_id = update.message.chat_id
    text = "Current runs:\n"

    for run_name in runs:
        if(run_name in threads):
            if(threads[run.name].is_alive()):
                text += "%s:   mut_rate = %.2f   mut_dev = %.2f   max_gen = %i\n"%(run_name, run.mut_rate, run.mut_dev, run.max_gen)

    bot.send_message(chat_id=chat_id, text=text)

get_active_handler = CommandHandler("get_active", get_active)
dispatcher.add_handler(get_active_handler)



def get_runs(bot, update):
    chat_id = update.message.chat_id
    text = "All runs:\n"

    for run_name in runs:
        run = runs[run_name]
        text += "%s:   mut_rate = %.2f   mut_dev = %.2f   max_gen = %i\n"%(run.name, run.mut_rate, run.mut_dev, run.max_gen)

    bot.send_message(chat_id=chat_id, text=text)

get_runs_handler = CommandHandler("get_runs", get_runs)
dispatcher.add_handler(get_runs_handler)



def get_picture(bot, update):
    chat_id = update.message.chat_id

    input = update.message.text.split()

    for run_name in input[1:]:

        path = "./data/%s/fitness.png"%(run_name)

        if(os.path.isfile(path)):
            bot.sendPhoto(chat_id=chat_id, caption=run_name, photo=open(path, "rb"))
        else:
            bot.send_message(chat_id=chat_id, text="No file found for %s"%run_name)
        

dispatcher.add_handler(CommandHandler("get_picture", get_picture))


def stop_run(bot, update):
    chat_id = update.message.chat_id

    run_name = update.message.text.split()[1]

    if(run_name not in threads):
        bot.send_message(chat_id=chat_id, text="Run not found or not active")
        return
    else:
        threads[run_name].terminate()
        bot.send_message(chat_id=chat_id, text="%s has been terminated"%run_name)

dispatcher.add_handler(CommandHandler("stop_run", stop_run))


def del_run(bot, update):
    chat_id = update.message.chat_id
    
    input = update.message.text.split()

    for run_name in input[1:]:
        if(run_name not in runs):
            bot.send_message(chat_id=chat_id, text="%s not found"%run_name)
        else:
            if(run_name in threads):
                if(threads[run_name].is_alive()):
                    threads[run_name].terminate()
                threads.pop(run_name)
            runs.pop(run_name)
            shutil.rmtree("./data/%s"%run_name)
            bot.send_message(chat_id=chat_id, text="%s and all corresponding data have been deleted"%run_name)

dispatcher.add_handler(CommandHandler("del_run", del_run))


if __name__ == "__main__": 

    Run = namedtuple("Run", ['name', 'mut_rate', 'mut_dev', 'max_gen'])

    
    if(os.path.isfile("./data/runs.obj")):
        file = open("./data/runs.obj", "rb")
        runs = pickle.load(file)
    else:
        runs = {}

    threads = {}

    n_runs = len(runs)

    logging.info("starting bot")

    updater.start_polling()
    updater.idle()

    logging.info("bot has stopped")