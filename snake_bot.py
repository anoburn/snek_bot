from telegram.ext import Updater, CommandHandler#, MessageHandler, Filters
import logging
from multiprocessing import Process, Queue
import numpy as np
import snake
import pickle
import os
import psutil
import shutil


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

updater = Updater(token="559384287:AAGt9y-Cyl7CmW_8nr4a3W0yL6bpbu9NHxM")
dispatcher = updater.dispatcher


class Run():
    def __init__(self, name, mut_rate, mut_dev, max_gen):
        self.name = name
        self.mut_rate = mut_rate
        self.mut_dev = mut_dev
        self.max_gen = max_gen
        self.finished = False


def save_runs():
    global runs
    if(not finished_runs.empty()):
        for run_name in finished_runs.get(False):
            runs[run_name].finished = True

    path = "./data/"
    if not os.path.exists(path):
            os.makedirs(path)
    datei = open("./data/runs.obj", "wb")
    pickle.dump(runs, datei)


def fill_up_runs():
    global runs
    global n_runs
    keys = list(runs.keys())
    keys.sort()
    runs_new = {}
    for i in range(len(runs)):
        name_new = "run_%i"%i
        name_old = keys[i]
        runs_new[name_new] = runs[name_old]
        runs_new[name_new].name = name_new

        if name_old in queues:
            queues[name_old].put("newname %s"%name_new)
            while(name_old in queues and not queues[name_old].empty()):
                pass
            queues[name_new] = queues.pop(name_old)
        if name_old in threads:
            if(threads[name_old].is_alive()):
                threads[name_new] = threads.pop(name_old)

        os.rename('./data/%s/'%name_old, './data/%s/'%name_new)
    runs = runs_new
    n_runs = len(runs)
    save_runs()

def fill_up_command(bot, update):
    fill_up_runs()
dispatcher.add_handler(CommandHandler("fill_up_runs", fill_up_command))


def start_run(bot, update):

    input = update.message.text.split()
    mut_rate  =  float(input[1])
    mut_dev   =  float(input[2])
    max_gen   =  int(input[3])

    logging.info("{} called start_run with parameters {}".format(update.message.chat_id, input))

    global n_runs
    run_name = "run_%i"%(n_runs)
    n_runs += 1

    q = Queue()

    thread = Process(name=run_name, target=run_new_world, args=(bot, update, run_name, mut_rate, mut_dev, max_gen, q, finished_runs))

    run = Run(run_name, mut_rate, mut_dev, max_gen)
    runs[run_name] = run
    save_runs()

    threads[run_name] = thread
    queues[run_name]  = q

    thread.start()

start_handler = CommandHandler("start_run", start_run)
dispatcher.add_handler(start_handler)


def continue_run(bot, update, run_name):
    text = None
    if(run_name not in runs):
        text = "Can't find %s"%run_name
    elif(runs[run_name].finished):
        text = "%s has already finished"%run_name
    elif(run_name in threads):
        if(threads[run_name].is_alive()):
            text = "%s is already running"%run_name

    if(text is None):
        q = Queue()
        thread = Process(name=run_name, target=run_old_world, args=(bot, update, run_name, q, finished_runs))
        threads[run_name] = thread
        queues[run_name]  = q
        thread.start()
    else:
        bot.send_message(chat_id=update.message.chat_id, text=text)


def continue_command(bot, update):
    input = update.message.text.split()[1:]

    save_runs()

    for run_name in input:
        continue_run(bot, update, run_name)

dispatcher.add_handler(CommandHandler("continue", continue_command))


def continue_all(bot, update):
    save_runs()

    for run_name in runs:
        if(not runs[run_name].finished):
            continue_run(bot, update, run_name)

    #bot.send_message(chat_id=update.message.chat_id, text='All unfinished runs are continuing')

dispatcher.add_handler(CommandHandler('continue_all', continue_all))


def run_new_world(bot, update, run_name, mut_rate, mut_dev, max_gen, q, finished_runs):
    bot.send_message(chat_id=update.message.chat_id, text="Starting thread %s "%(run_name))
    logging.info("Starting thread %s "%(run_name))

    world = snake.GameHandler(np.zeros((30, 30)), mut_rate, mut_dev, max_gen, run_name)
    actual_run(bot, update.message.chat_id, run_name, world, q, finished_runs)


def run_old_world(bot, update, run_name, q, finished_runs):
    chat_id = update.message.chat_id
    bot.send_message(chat_id=chat_id, text="Continuing %s"%run_name)
    world = pickle.load(open('data/%s/world.obj'%run_name, 'rb'))
    bot.send_message(chat_id=chat_id, text="Loaded %s, run restored"%run_name)

    actual_run(bot, chat_id, run_name, world, q, finished_runs)


def actual_run(bot, chat_id, run_name, world, q, finished_runs):
    p = psutil.Process(os.getpid())
    if(os.name == 'nt'):
        #Windows
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    elif(os.name == "posix"):
        #Unix
        p.nice(19)

    result = world.run(q)

    if(result):
        if(finished_runs.empty()):
            content = []
        else:
            content = finished_runs.get()
        content.append(run_name)
        finished_runs.put(content)
        logging.info("Finished %s"%run_name)
        bot.send_message(chat_id=chat_id, text="Finished %s"%run_name)
    else:
        logging.info("Interrupted %s"%run_name)
        bot.send_message(chat_id=chat_id, text="Interrupted %s"%run_name)


def pause_run(bot, update, run_name):
    chat_id = update.message.chat_id
    if(run_name in threads):
        if(threads[run_name].is_alive()):
            bot.send_message(chat_id=chat_id, text="Pausing %s"%run_name)
            logging.info("Pausing %s"%run_name)
            q = queues[run_name]
            q.put('stop')
    else:
        bot.send_message(chat_id=chat_id, text="Can't find %s in active runs"%run_name)


def pause(bot, update):
    input = update.message.text.split()[1:]
    for run_name in input:
        pause_run(bot, update, run_name)

dispatcher.add_handler(CommandHandler('pause', pause))


def pause_all(bot, update):
    chat_id = update.message.chat_id
    for run_name in threads:
        pause_run(bot, update, run_name)
    bot.send_message(chat_id=chat_id, text="All active bots have received pause command. Please wait for them to confirm.")

dispatcher.add_handler(CommandHandler('pause_all', pause_all))


def get_active(bot, update):
    save_runs()
    chat_id = update.message.chat_id
    text = "Current runs:\n"

    for run_name in runs:
        if(run_name in threads):
            run = runs[run_name]
            if(threads[run.name].is_alive()):
                text += "%s:   mut_rate = %.2f   mut_dev = %.2f   max_gen = %i\n"%(run.name, run.mut_rate, run.mut_dev, run.max_gen)

    bot.send_message(chat_id=chat_id, text=text)

get_active_handler = CommandHandler("get_active", get_active)
dispatcher.add_handler(get_active_handler)


def get_runs(bot, update):
    save_runs()
    chat_id = update.message.chat_id
    text = "Specified runs:\n"
    input = update.message.text.split()[1:]
    runs_copy = dict(runs)

    for cond in input:
        cond = cond.split('=')
        cond[0] = cond[0].lower()
        cond[1] = cond[1].lower()
        if '-' in cond[1]:
            min, max = cond[1].split('-')
            min = float(min)
            max = float(max)

            for run_name in runs:
                run = runs[run_name]
                value = getattr(run, cond[0])
                if not (min <= value <= max):
                    runs_copy.pop(run_name, None)
        else:
            for run_name in runs:
                run = runs[run_name]
                value = getattr(run, cond[0])
                if value != float(cond[1]):
                    runs_copy.pop(run_name, None)
    for run_name in sorted(list(runs_copy.keys())):
        run = runs[run_name]
        text += "%s:   mut_rate = %.2f   mut_dev = %.2f   max_gen = %i   done = %r\n"%(run.name, run.mut_rate, run.mut_dev, run.max_gen, run.finished)

    bot.send_message(chat_id=chat_id, text=text)

dispatcher.add_handler(CommandHandler("get_runs", get_runs))


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

dispatcher.add_handler(CommandHandler("stop", stop_run))


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

    fill_up_runs()   #includes save_runs()

dispatcher.add_handler(CommandHandler("del", del_run))


if __name__ == "__main__": 

    #Run = namedtuple("Run", ['name', 'mut_rate', 'mut_dev', 'max_gen', 'finished'])

    threads = {}
    queues  = {}
    finished_runs = Queue()

    if(os.path.isfile("./data/runs.obj")):
        file = open("./data/runs.obj", "rb")
        runs = pickle.load(file)
        fill_up_runs()
    else:
        runs = {}

    n_runs = len(runs)

    logging.info("starting bot")

    updater.start_polling()
    updater.idle()

    logging.info("bot has stopped")
    save_runs()