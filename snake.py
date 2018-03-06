import pygame
import numpy as np
import math
import logging
import sys
import random
import matplotlib.pyplot as plt
import os
import timeit
import functools
 

#def setup():
#    pygame.init()
#    logging.basicConfig(stream=sys.stderr, level=logging.ERROR)  # set to ERROR to omit debugging
 
 
def start_run(mut_rate, mut_dev, max_gen):
    logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
    pygame.init()       # Wurde von setup() hierhin verschoben
    size = width, height = 800, 600
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    rec = Rectangle(10, 10, 1)
    grid = np.zeros((30, 30))
    game = World(grid, mut_rate, mut_dev)
    lastgen = 0
    while game.running and lastgen <= max_gen:
        game.current_snake.calc_dir(game.input)
        #print(timer.timeit(10))
        #timer = timeit.Timer(functools.partial(game.generate_matrix, game.population[0].input_to_hidden1, game.population[1].input_to_hidden1, 0.5))
        #print(timer.timeit(10))
        game.move(game.current_snake.dir)
        game.update_score()
        if lastgen % 100 == 0:
            #game.save_population(game.autosave_file)
            pass
        if not game.show:
            if lastgen != game.generation:
                lastgen = game.generation
                game.update_messages()
                pygame.display.flip()
                screen.fill((0, 0, 0))
                for text, recta in game.text_boxes:
                    screen.blit(text, recta)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    game.show = not game.show
        # for event in pygame.event.get():
        #     if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        #         game.show = not game.show
        else:
            game.update_messages()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        game.paused = not game.paused
                    elif event.key == pygame.K_ESCAPE:
                        game.running = False
                        break
                    elif event.key == pygame.K_LEFT:
                        game.set_mutation_deviation(-0.1)
                    #     game.current_snake.set_dir('LEFT')
                    #     logging.debug('L')
                    elif event.key == pygame.K_RIGHT:
                        game.set_mutation_deviation(0.1)
                    #     game.current_snake.set_dir('RIGHT')
                    #     logging.debug('R')
                    elif event.key == pygame.K_DOWN:
                        game.set_speed(-1)
                    #     game.current_snake.set_dir('DOWN')
                    #     logging.debug('D')
                    elif event.key == pygame.K_UP:
                        game.set_speed(1)
                    elif event.key == pygame.K_w:
                        game.set_mutation_rate(0.01)
                    elif event.key == pygame.K_s:
                        game.set_mutation_rate(-0.01)
                    elif event.key == pygame.K_p:
                        game.show = not game.show
                    #     game.current_snake.set_dir('UP')
                    #     logging.debug('U')
            if game.paused:
                continue
            # game.move(game.current_snake.dir)
            # game.current_snake.calc_dir(game.input)
            screen.fill((0, 0, 0))
            for text, recta in game.text_boxes:
                screen.blit(text, recta)
            for row in range(game.width):
                for column in range(game.height):
                    color = game.get_RGB(row, column)
                    pygame.draw.rect(screen, color, [(rec.s + rec.w) * column + rec.s + 10,
                                                     (rec.s + rec.h) * row + rec.s + 10, rec.w, rec.h])
            clock.tick(game.speed * 10)
    game.save_population(game.save_file)
    pygame.quit()
 
 
class Rectangle:
    def __init__(self, _width, _height, _spacing):
        self.w = _width
        self.h = _height
        self.s = _spacing
 
 
class Snake:
    def __init__(self, m1= None, m2= None, m3= None):
        self.dir = 'RIGHT'
        self.body = []
        self.hidden_layer_1_neurons = 32
        self.hidden_layer_2_neurons = 16
        self.output_neurons = 4
        self.input_neurons = 7

        if(m1 is None):
            self.input_to_hidden1 = np.random.standard_normal(size=(self.hidden_layer_1_neurons, self.input_neurons))
        else:
            self.input_to_hidden1 = m1
        if(m2 is None):
            self.hidden1_to_hidden2 = np.random.standard_normal(size=(self.hidden_layer_2_neurons, self.hidden_layer_1_neurons))
        else:
            self.hidden1_to_hidden2 = m2
        if(m3 is None):
            self.hidden2_to_output = np.random.standard_normal(size=(self.output_neurons, self.hidden_layer_2_neurons))
        else:
            self.hidden2_to_output = m3

        self.fitness = 1
        self.dead = False

    def append(self, x, y):
        self.body.append((x, y))
 
    def set_dir(self, direction):
        #     if direction == 'RIGHT' and self.dir != 'LEFT':
        #         self.dir = direction
        #     elif direction == 'LEFT' and self.dir != 'RIGHT':
        #         self.dir = direction
        #     elif direction == 'UP' and self.dir != 'DOWN':
        #         self.dir = direction
        #     elif direction == 'DOWN' and self.dir != 'UP':
        #         self.dir = direction
        self.dir = direction
 
    def calc_dir(self, _input):
        hidden1 = np.dot(self.input_to_hidden1, _input)
        hidden2 = np.dot(self.hidden1_to_hidden2, hidden1)
        output = np.dot(self.hidden2_to_output, hidden2)
        self.dir = int_to_dir[int(np.argmax(output))]
 
 
class World:  # starting population size, population, mutation rate, mutation deviation,
    def __init__(self, _game_grid, _mutation_rate, _mutation_dev):
        # get the basic game grid and resulting width and height
        self.grid = _game_grid
        self.width, self.height = self.grid.shape
        # instantiate snake and append starting snake
        # TODO: generate runspecific folder
        self.generation = 1
        self.population = [Snake() for _ in range(50)]
        self.spawn_counter = 1
        self.current_snake = self.population[0]
        self.score = 1
        self.last_fruit = 0
        self.mutation_rate = _mutation_rate
        self.mutation_deviation = _mutation_dev
        # generate folder
        self.folder = './data/m_rate_{}_m_dev_{}/'.format(self.mutation_rate, self.mutation_deviation)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            os.makedirs(self.folder + 'autosave/')
 
        self.save_file = self.folder + 'gen{}.txt'.format(self.generation)
        self.autosave_file = self.folder + 'autosave/gen_{}.txt'.format(self.generation)
        self.figure_file = self.folder + 'gen_{}.png'.format(self.generation)
        self.figure_data_file = self.folder + 'figure_data.txt'
        self.parent_fitness = []
        self.death_wall = 0
        self.death_self = 0
        self.death_time = 0
        # self.fruit_loc = [(20, 10), (15, 20), (25, 5), (25, 10)]
        # game speed, running and pause
        self.speed = 8
        self.running = True
        self.show = True
        self.paused = False
        # textbox holder
        self.text_boxes = []  # contains: surface with rendered text, location_rectangle
        self.text_fonts = []  # contains: fonts
        # score font, text box
        score_text, score_rec, score_font = self.textbox(40, 'Score: {}'.format(self.score), 10, 390)
        self.text_boxes.append([score_text, score_rec])
        self.text_fonts.append(score_font)
        # generation font, text box
        generation_text, generation_rec, generation_font = self.textbox(40, 'Gen: {}'.format(self.generation), 10, 350)
        self.text_boxes.append([generation_text, generation_rec])
        self.text_fonts.append(generation_font)
        # snake font, text box
        snake_text, snake_rec, snake_font = self.textbox(40, 'Snake: {}'.format(self.spawn_counter), 170, 350)
        self.text_boxes.append([snake_text, snake_rec])
        self.text_fonts.append(snake_font)
        # mut_dev font, text box
        mut_dev_text, mut_dev_rec, mut_dev_font = self.textbox(40, 'Mutation deviation: {}'.format(self.mutation_deviation), 10, 430)
        self.text_boxes.append([mut_dev_text, mut_dev_rec])
        self.text_fonts.append(mut_dev_font)
        # death_counter font, text box
        death_counter_text, death_counter_rec, death_counter_font = self.textbox(40, 'Deaths: s: {}, w: {}, t: {}'.format(self.death_self, self.death_wall, self.death_time), 10, 510)
        self.text_boxes.append([death_counter_text, death_counter_rec])
        self.text_fonts.append(death_counter_font)
        # mut_rate font, text box
        mut_rate_text, mut_rate_rec, mut_rate_font = self.textbox(40, 'Mutation rate: {}'.format(self.mutation_rate), 10, 470)
        self.text_boxes.append([mut_rate_text, mut_rate_rec])
        self.text_fonts.append(mut_rate_font)
 
        self.start_run()
 
    def textbox(self, _fontsize, _text, pos_l, pos_t):
        font = pygame.font.SysFont(None, _fontsize)
        text = font.render(_text, True, (255, 255, 255))
        rec = text.get_rect()
        rec.left = pos_l
        rec.top = pos_t
        return text, rec, font
 
    def update_messages(self):
        for x, font in enumerate(self.text_fonts):
            if x == 0:
                self.text_boxes[x][0] = font.render('Score: {}'.format(self.score), True, (255, 255, 255))
            elif x == 1:
                self.text_boxes[x][0] = font.render('Gen: {}'.format(self.generation), True, (255, 255, 255))
            elif x == 2:
                self.text_boxes[x][0] = font.render('Snake: {}'.format(self.spawn_counter), True, (255, 255, 255))
            elif x == 3:
                self.text_boxes[x][0] = font.render('Mutation deviation: {}'.format(self.mutation_deviation), True, (255, 255, 255))
            elif x == 4:
                self.text_boxes[x][0] = font.render('Deaths: s: {}, w: {}, t: {}'.format(self.death_self, self.death_wall, self.death_time), True, (255, 255, 255))
            elif x == 5:
                self.text_boxes[x][0] = font.render('Mutation rate: {}'.format(self.mutation_rate), True, (255, 255, 255))
 
    def get_RGB(self, _row, _column):
        return number_to_RGB[self.grid[_row][_column]]
 
    def spawn_fruit(self):
        # if self.fruit_loc:
        #     self.fruit_pos = (self.fruit_loc[0])
        #     self.grid[self.fruit_loc[0]] = 2
        #     self.fruit_loc = self.fruit_loc[1:]
        # else:
        # generate two random numbers and spawn apple if the grid position is empty
        s = np.random.uniform(0, self.width, 2)
        s = (int(math.floor(s[0])), int(math.floor(s[1])))
        if self.grid[s[0]][s[1]] == 0:
            self.grid[s[0]][s[1]] = 2
            self.fruit_pos = (s[0], s[1])
        else:
            self.spawn_fruit()
 
    def move(self, direction):
        # check if snake is dead
        if not self.current_snake.dead:
            logging.debug(str(self.current_snake.dir))
            # check upcoming block
            x, y = directions[direction]
            goal = (self.current_snake.body[-1][0] + x, self.current_snake.body[-1][1] + y)
            if 0 <= goal[0] < self.width and 0 <= goal[1] < self.height:
                b = self.grid[goal[0]][goal[1]]
                if b == 1:
                    logging.info('Snake died: hit body')
                    self.death_self += 1
                    self.current_snake.dead = True
                # remove the last part of the snake if no fruit was found
                elif b == 0:
                    self.grid[self.current_snake.body[0][0]][self.current_snake.body[0][1]] = 0
                    self.current_snake.body = self.current_snake.body[1:]
                # spawn new fruit if one was found
                elif b == 2:
                    self.spawn_fruit()
                    self.score += 500
                    self.last_fruit = 0
                    logging.info('collected fruit')
                # move snake to direction
                self.append_snake(goal[0], goal[1])
                logging.debug('Snake Body: {}'.format(self.current_snake.body))
                self.update_distances()
            else:
                logging.info('Snake died: hit the wall')
                self.death_wall += 1
                self.current_snake.dead = True
        else:
            self.respawn()
 
    def respawn(self):
        self.current_snake.fitness = self.score
        if self.spawn_counter < self.population.__len__():
            self.current_snake = self.population[self.spawn_counter]
            self.spawn_counter += 1
            self.start_run()
        else:
            self.generate_images()
            self.reproduce()
 
    def generate_images(self):
        means, maxs = [], []
        for data in self.parent_fitness:
            data = np.array(data)
            means.append(np.mean(data))
            maxs.append(np.max(data))
 
        len = means.__len__()
        fig1 = plt.figure()
        mean, = plt.plot(range(len), means, color='green', linestyle='solid', linewidth=0.5)
        max, = plt.plot(range(len), maxs, color='red', marker=',', linewidth=0.2)
        plt.title('Mutationrate: {}, Mutationdeviation: {}'.format(self.mutation_rate, self.mutation_deviation))
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend([mean, max], ["mean", "max"])
        plt.savefig(self.figure_file)
        plt.close(fig1)
 
            # print(data)
 
    def reproduce(self):
        logging.info('\n\n\n\n\nGenerating Offspring')
        self.generation += 1
        # get fitness of all snakes, get top x fittest, get indices of top x fittest
        scores = np.array([self.population[i].fitness for i in range(self.population.__len__())], dtype=int)
        top_indices = scores.argsort()[-20:][::-1]
        top_fitness = sorted(scores)[-20:][::-1]
        logging.debug('fitness of parents: {}\ntop 20 fitness: {}'.format(scores, top_fitness))
        self.parent_fitness.append(top_fitness)
        # get list of parents and generate offspring
        parents = [self.population[index] for index in top_indices]
        logging.debug('parent 0: \nmatrix0: {}, \nmatrix1:{}, \nmatrix2:{}'.format(parents[0].input_to_hidden1, parents[0].hidden1_to_hidden2, parents[0].hidden2_to_output))
        np_o_p = [8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2]
        #offspring = [Snake() for _ in range(parents.__len__() * 5)]
        offspring = []
        for i, parent in enumerate(parents):
            partners = [parents[index] for index in random.sample(range(parents.__len__()), np_o_p[i])]
            logging.debug('Parent: {}, partners length: {}, with partners: {} '.format(parents[i], partners.__len__(), partners))
            for j, partner in enumerate(partners):
                quotient = parent.fitness / (parent.fitness + partner.fitness)
                
                #offspring[i * partners.__len__() + j].input_to_hidden1 = self.generate_matrix(parent.input_to_hidden1, partner.input_to_hidden1, quotient)
                #offspring[i * partners.__len__() + j].hidden1_to_hidden2 = self.generate_matrix(parent.hidden1_to_hidden2, partner.hidden1_to_hidden2, quotient)
                #offspring[i * partners.__len__() + j].hidden2_to_output = self.generate_matrix(parent.hidden2_to_output, partner.hidden2_to_output, quotient)
                
                matrix1 = self.generate_matrix(parent.input_to_hidden1, partner.input_to_hidden1, quotient)
                matrix2 = self.generate_matrix(parent.hidden1_to_hidden2, partner.hidden1_to_hidden2, quotient)
                matrix3 = self.generate_matrix(parent.hidden2_to_output, partner.hidden2_to_output, quotient)
                offspring.append(Snake(matrix1, matrix2, matrix3))
                logging.debug('done with one offspring\n\n\n')
        #clone = [Snake() for _ in range(10)]
        for i in range(2):
            for j in range(5):
                #clone[i * partners.__len__() + j].input_to_hidden1 = self.generate_matrix(parent.input_to_hidden1, partner.input_to_hidden1, quotient)
                #clone[i * partners.__len__() + j].hidden1_to_hidden2 = self.generate_matrix(parent.hidden1_to_hidden2, partner.hidden1_to_hidden2, quotient)
                #clone[i * partners.__len__() + j].hidden2_to_output = self.generate_matrix(parent.hidden2_to_output, partner.hidden2_to_output, quotient)
                matrix1 = self.generate_matrix( parents[i].input_to_hidden1,   parents[i].input_to_hidden1,   0.)
                matrix2 = self.generate_matrix( parents[i].hidden1_to_hidden2, parents[i].hidden1_to_hidden2, 0.)
                matrix3 = self.generate_matrix( parents[i].hidden2_to_output,  parents[i].hidden2_to_output,  0.)
                offspring.append(Snake(matrix1, matrix2, matrix3))
                #offspring.append(clone[i * partners.__len__() + j])
        fresh = [Snake() for _ in range(20)]
        for snek in fresh:
            offspring.append(snek)
        random.shuffle(offspring)
        self.population = offspring
        self.spawn_counter = 1
        self.current_snake = self.population[0]
        self.death_wall = 0
        self.death_self = 0
        self.death_time = 0
        self.start_run()
 
    def generate_matrix(self, p_matrix1, p_matrix2, _quotient):
        neurons2, neurons1 = p_matrix1.shape
        matrix_elter1 = p_matrix1.reshape(1, -1)
        matrix_elter2 = p_matrix2.reshape(1, -1)
        index = random.sample(range(matrix_elter1.size), int(_quotient * matrix_elter1.size))
        mask = np.zeros(matrix_elter1.size)
        for k in index:
            mask[k] = 1
        result = np.where(mask == 1, matrix_elter1, matrix_elter2)
        mutate = np.zeros(result.shape)
        mutation_spot = random.sample(range(mutate.size), int(np.ceil(mutate.size * self.mutation_rate)))
        for k in mutation_spot:
            mutate[0][k] = np.random.normal(0, self.mutation_deviation)
        result = np.add(result, mutate)
        #logging.debug('mutation_spot: {}, \nby: \n{}'.format(mutation_spot, mutate))
        # logging.debug('Fitness quotient: {},\n index: {},\n'.format(quotient, mask))
        return result.reshape((neurons2, neurons1))
 
    def start_run(self):
        logging.warning('STARTED RUN')
        self.grid = np.zeros((self.width, self.height))
        start_loc = np.random.randint(5, 25, size=2)
        self.spawn_snake(start_loc[0], start_loc[1])
        # self.fruit_loc = [(20, 10), (15, 20), (25, 5), (25, 10)]
        # set score, spawn fruit and set last fruit to 0
        self.score = 1
        self.last_fruit = 0
        self.spawn_fruit()
        self.update_distances()
 
    def spawn_snake(self, _start_x, _start_y):
        if _start_y < 15:
            self.append_snake(_start_x, _start_y)
            _start_y += 1
            self.append_snake(_start_x, _start_y)
            _start_y += 1
            self.append_snake(_start_x, _start_y)
        elif _start_y >= 15:
            self.append_snake(_start_x, _start_y)
            _start_y -= 1
            self.append_snake(_start_x, _start_y)
            _start_y -= 1
            self.append_snake(_start_x, _start_y)
 
    def append_snake(self, x, y):
        self.grid[x][y] = 1
        self.current_snake.append(x, y)
 
    def remove_last(self):
        self.grid[self.current_snake.body[0][0]][self.current_snake.body[0][1]] = 0
        self.current_snake.body = self.current_snake.body[1:]
 
    def update_distances(self):
        #y, x = self.current_snake.body[-1][0], self.current_snake.body[-1][1]
        #logging.debug('head pos, x: {}, y: {}'.format(x, y))


        body_array = np.array(self.current_snake.body)
        head       = np.array(self.current_snake.body[-1])
        logging.debug('head pos, x: %i, y: %i'%(head[1], head[0]))
        
        diff = body_array - head

        x_same = diff[ diff[:,1]==0, 0 ]
        up   = min( -x_same[x_same < 0] -1, default =    head[0])
        down = min(  x_same[x_same > 0] -1, default = 29-head[0])


        y_same = diff[ diff[:,0]==0, 1 ]
        left  = min( -y_same[y_same < 0] -1, default =    head[1])
        right = min(  y_same[y_same > 0] -1, default = 29-head[1])

 
        '''vertical, horizontal = [], []
        # vert = [self.grid[y][i] if self.grid[y][i] == 1 else None for i in range(self.width)]
        # logging.info('???: {}'.format(vert))
        for i in range(self.width):
            if self.grid[y][i] == 1 and i != x:
                horizontal.append(i)
        for i in range(self.height):
            if self.grid[i][x] == 1 and i != y:
                vertical.append(i)
        logging.debug('horizontal: {}, vertical: {}'.format(horizontal, vertical))
 
        hor_pre = [num for num in horizontal if num < x]
        hor_post = [num for num in horizontal if num > x]
        logging.debug('hor_pre: {}, hor_post: {}'.format(hor_pre, hor_post))
        ve_pre = [num for num in vertical if num < y]
        ve_post = [num for num in vertical if num > y]
        logging.debug('ver_pre: {}, ver_post: {}'.format(ve_pre, ve_post))
 
        right  = min(hor_post, default=30) - x - 1
        left   = x - max(hor_pre, default=-1) - 1
        up     = y - max(ve_pre, default=-1) - 1
        down   = min(ve_post, default=30) - y - 1'''
 
        fruit_horizontal   = self.fruit_pos[1] - head[1]
        fruit_vertical     = self.fruit_pos[0] - head[0]
        self.input = np.array([right, left, down, up, fruit_horizontal, fruit_vertical, 1])
        # if x < self.fruit_pos[1]:
        #     self.fruit_right    = self.fruit_pos[1] - x
        #     self.fruit_left     = 0
        # else:
        #     self.fruit_right    = 0
        #     self.fruit_left     = x - self.fruit_pos[1]
        # if y < self.fruit_pos[0]:
        #     self.fruit_up       = 0
        #     self.fruit_down     = self.fruit_pos[0] - y
        # else:
        #     self.fruit_up       = y - self.fruit_pos[0]
        #     self.fruit_down     = 0
        #
        # self.input = np.array([self.right, self.left, self.down, self.up,
        #                        self.fruit_right, self.fruit_left, self.fruit_down, self.fruit_up, 1])
        logging.debug('input: {}'.format(self.input))
 
    def update_score(self):
        # if self.last_fruit < 1:
        #     self.score += 1
        self.last_fruit += 1
        if self.last_fruit > 700:
            distance = np.array(self.fruit_pos) - np.array(self.current_snake.body[-1])
            distance = np.sqrt( np.sum( distance * distance))
            #self.score += int( 1 / distance ) * 200
            self.score += (200 - int(distance * 200 / 43))
            self.current_snake.dead = True
            logging.info('Snake died: timeout')
            self.death_time += 1
 
    def set_speed(self, _int):
        if self.speed + _int > 0:
            self.speed += _int
 
    def set_mutation_deviation(self, _float):
        if self.mutation_deviation + _float >= 0:
            self.mutation_deviation += _float
            self.mutation_deviation = np.round(self.mutation_deviation, decimals=1)
 
    def set_mutation_rate(self, _float):
        if self.mutation_rate + _float >= 0:
            self.mutation_rate += _float
            self.mutation_rate = np.round(self.mutation_rate, decimals=2)
 
    def save_population(self, _file):
        with open(_file, 'w') as f:
            f.write('Generation:{}\tSnake:{}\tMutation_Deviation:{}\t\n'.format(self.generation, self.spawn_counter, self.mutation_deviation))
            for snake in self.population:
                np.savetxt(f, snake.input_to_hidden1, delimiter=',')
                f.write('\n')
                np.savetxt(f, snake.hidden1_to_hidden2, delimiter=',')
                f.write('\n')
                np.savetxt(f, snake.hidden2_to_output, delimiter=',')
                f.write('\n')
 
    def auto_save(self):
        with open(self.autosave_file, 'w') as f:
            f.write('Generation:{}\tSnake:{}\tMutation_Deviation:{}\t\n'.format(self.generation, self.spawn_counter, self.mutation_deviation))
            for snake in self.population:
                np.savetxt(f, snake.input_to_hidden1, delimiter=',')
                f.write('\n')
                np.savetxt(f, snake.hidden1_to_hidden2, delimiter=',')
                f.write('\n')
                np.savetxt(f, snake.hidden2_to_output, delimiter=',')
                f.write('\n')
 
 
number_to_RGB = {
    0: (255, 255, 255),  # 0 = white
    1: (0, 0, 0),        # 1 = black
    2: (255, 0, 0)       # 2 = red
}
directions = {
    'UP':       (-1, 0),
    'DOWN':     (1, 0),
    'LEFT':     (0, -1),
    'RIGHT':    (0, 1)
}
int_to_dir = {
    0: 'RIGHT',
    1: 'LEFT',
    2: 'DOWN',
    3: 'UP'
}
 
if __name__ == "__main__":
    #setup()
    start_run(0.05, 0.3, 2500)
