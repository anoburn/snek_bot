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
 

class Rectangle:
    def __init__(self, _width, _height, _spacing):
        self.w = _width
        self.h = _height
        self.s = _spacing
 
 
class Snake:
    def __init__(self, m1=None, m2=None, m3=None):
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
        self.snake_score = 0
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
 
 
class GameHandler:  # starting population size, population, mutation rate, mutation deviation,
    def __init__(self, _game_grid, _mutation_rate, _mutation_dev, _max_generation, save_directory=None):
        # get the basic game grid and resulting width and height
        self.grid = _game_grid
        self.width, self.height = self.grid.shape
        # instantiate snake and append starting snake
        self.generation = 1
        self.population = [Snake() for _ in range(50)]
        self.spawn_counter = 1
        self.current_snake = self.population[0]
        self.score = 1
        self.last_fruit = 0
        self.mutation_rate = _mutation_rate
        self.mutation_deviation = _mutation_dev
        if _max_generation < 0:
            self.max_generation = True
        else:
            self.max_generation = _max_generation
        # generate folder
        if(save_directory is not None):
            self.folder = save_directory
        else:
            self.folder = './data/m_rate_{}_m_dev_{}/'.format(self.mutation_rate, self.mutation_deviation)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            os.makedirs(self.folder + 'autosave/')
 
        self.save_file = self.folder + 'gen{}.txt'.format(self.generation)
        self.autosave_file = self.folder + 'autosave/gen_{}.txt'.format(self.generation)
        self.figure_fitness_file = self.folder + 'fitness.png'
        self.figure_fitness_data_file = self.folder + 'fitness_data.txt'
        self.figure_snake_score_file = self.folder + 'score.png'
        self.figure_snake_score_data_file = self.folder + 'score_data.txt'
        self.parent_fitness = []
        self.parent_snake_scores = []
        self.fitness_means, self.fitness_maxs = [], []
        self.snake_scores_means, self.snake_scores_maxs = [], []
        self.death_wall = 0
        self.death_self = 0
        self.death_time = 0
        # self.fruit_loc = [(20, 10), (15, 20), (25, 5), (25, 10)]
        # game speed, running and pause
        self.speed = 8
        self.running = True
        self.show = False
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
        # images
        try:
            self.plot_fitness = pygame.image.load(os.path.join(self.figure_fitness_file)).convert()
        except:
            self.plot_fitness = pygame.Surface((100, 100))
        self.start_snake()
 
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

    def load_images(self):
        self.plot_fitness = pygame.image.load(os.path.join(self.figure_fitness_file)).convert()

    def get_RGB(self, _row, _column):
        return number_to_RGB[self.grid[_row][_column]]
 
    def spawn_fruit(self):
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
            logging.debug('Direction: {}'.format(self.current_snake.dir))
            # check upcoming block
            x, y = directions[direction]
            goal = (self.current_snake.body[-1][0] + x, self.current_snake.body[-1][1] + y)
            if 0 <= goal[0] < self.width and 0 <= goal[1] < self.height:
                b = self.grid[goal[0]][goal[1]]
                if b == 1:
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
                    self.current_snake.snake_score += 1
                    self.last_fruit = 0
                    logging.debug('collected fruit')
                # move snake to direction
                self.append_snake(goal[0], goal[1])
                logging.debug('Snake Body: {}'.format(self.current_snake.body))
                self.update_distances()
            else:
                self.death_wall += 1
                self.current_snake.dead = True
        else:
            self.respawn()
 
    def respawn(self):
        self.current_snake.fitness = self.score
        if self.spawn_counter < self.population.__len__():
            self.current_snake = self.population[self.spawn_counter]
            self.spawn_counter += 1
            self.start_snake()
        else:
            self.reproduce()
            self.generate_images()
 
    def generate_images(self):
        data = self.parent_fitness[-1]
        data = np.array(data)
        self.fitness_means.append(np.mean(data))
        self.fitness_maxs.append(np.max(data))
        data = self.parent_snake_scores[-1]
        data = np.array(data)
        self.snake_scores_means.append(np.mean(data))
        self.snake_scores_maxs.append(np.max(data))
        x = range(1, self.fitness_means.__len__() + 1)
        fig1 = plt.figure()
        fitness_mean, = plt.plot(x, self.fitness_means, color='green', linestyle='solid', linewidth=0.5)
        fitness_max, = plt.plot(x, self.fitness_maxs, color='red', marker=',', linewidth=0.2)
        plt.title('Mutationrate: {}, Mutationdeviation: {}'.format(self.mutation_rate, self.mutation_deviation))
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend([fitness_mean, fitness_max], ["mean", "max"])
        plt.savefig(self.figure_fitness_file)
        fig2 = plt.figure()
        snake_score_mean, = plt.plot(x, self.snake_scores_means, color='green', linestyle='solid', linewidth=0.5)
        snake_score_max, = plt.plot(x, self.snake_scores_maxs, color='red', marker=',', linewidth=0.2)
        plt.title('Mutationrate: {}, Mutationdeviation: {}'.format(self.mutation_rate, self.mutation_deviation))
        plt.xlabel('Generation')
        plt.ylabel('Snake Score')
        plt.legend([snake_score_mean, snake_score_max], ["mean", "max"])
        plt.savefig(self.figure_snake_score_file)
        plt.close(fig1)
        plt.close(fig2)
        self.load_images()

    def reproduce(self):
        logging.debug('\n\nGenerating Offspring')
        self.generation += 1
        # get fitness of all snakes, get top x fittest, get indices of top x fittest
        fitnesses = np.array([self.population[i].fitness for i in range(self.population.__len__())], dtype=int)
        top_indices = fitnesses.argsort()[-20:][::-1]
        top_fitness = sorted(fitnesses)[-20:][::-1]
        self.parent_snake_scores.append(np.array([self.population[i].snake_score for i in range(self.population.__len__())]))
        logging.debug('fitness of parents: {}\ntop 20 fitness: {}'.format(fitnesses, top_fitness))
        self.parent_fitness.append(top_fitness)
        # get list of parents and generate offspring
        parents = [self.population[index] for index in top_indices]
        number_of_partners = [8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2]
        offspring = []
        for i, parent in enumerate(parents):
            partners = [parents[index] for index in random.sample(range(parents.__len__()), number_of_partners[i])]
            for j, partner in enumerate(partners):
                quotient = parent.fitness / (parent.fitness + partner.fitness)
                matrix1 = self.generate_matrix(parent.input_to_hidden1, partner.input_to_hidden1, quotient)
                matrix2 = self.generate_matrix(parent.hidden1_to_hidden2, partner.hidden1_to_hidden2, quotient)
                matrix3 = self.generate_matrix(parent.hidden2_to_output, partner.hidden2_to_output, quotient)
                offspring.append(Snake(matrix1, matrix2, matrix3))
        for i in range(2):
            for j in range(5):
                matrix1 = self.generate_matrix(parents[i].input_to_hidden1,   parents[i].input_to_hidden1,   0.)
                matrix2 = self.generate_matrix(parents[i].hidden1_to_hidden2, parents[i].hidden1_to_hidden2, 0.)
                matrix3 = self.generate_matrix(parents[i].hidden2_to_output,  parents[i].hidden2_to_output,  0.)
                offspring.append(Snake(matrix1, matrix2, matrix3))
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
        self.start_snake()
 
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
        # logging.debug('mutation_spot: {}, \nby: \n{}'.format(mutation_spot, mutate))
        # logging.debug('Fitness quotient: {},\n index: {},\n'.format(quotient, mask))
        return result.reshape((neurons2, neurons1))
 
    def start_snake(self):
        logging.debug('STARTED RUN')
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
        body_array = np.array(self.current_snake.body)
        head       = np.array(self.current_snake.body[-1])
        logging.debug('head pos, x: %i, y: %i'%(head[1], head[0]))
        
        diff = body_array - head

        x_same = diff[diff[:, 1] == 0, 0]
        up   = min(-x_same[x_same < 0] - 1, default=    head[0])
        down = min( x_same[x_same > 0] - 1, default= 29-head[0])

        y_same = diff[diff[:, 0] == 0, 1]
        left  = min(-y_same[y_same < 0] - 1, default=    head[1])
        right = min( y_same[y_same > 0] - 1, default= 29-head[1])
 
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
            distance = np.sqrt(np.sum(distance * distance))
            # self.score += int( 1 / distance ) * 200
            self.score += (200 - int(distance * 200 / 43))
            self.current_snake.dead = True
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

    def run(self):
        lastgen = 0
        while self.running and lastgen <= self.max_generation:
            self.current_snake.calc_dir(self.input)
            # print(timer.timeit(10))
            # timer = timeit.Timer(functools.partial(game.generate_matrix, game.population[0].input_to_hidden1, game.population[1].input_to_hidden1, 0.5))
            # print(timer.timeit(10))
            self.move(game.current_snake.dir)
            self.update_score()
            if lastgen % 100 == 0:
                # TODO: self.save_population(self.autosave_file)
                pass
            if not self.show:
                if lastgen != self.generation:
                    lastgen = self.generation
                    self.update_messages()
                    pygame.display.flip()
                    screen.fill((0, 0, 0))
                    for text, recta in self.text_boxes:
                        screen.blit(text, recta)
                    screen.blit(self.plot_fitness, (350, 0))
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        self.show = not self.show
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.running = False
            else:
                self.update_messages()
                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        # if event.key == pygame.K_SPACE:
                        #     game.paused = not game.paused
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                            break
                        elif event.key == pygame.K_LEFT:
                            self.set_mutation_deviation(-0.1)
                        elif event.key == pygame.K_RIGHT:
                            self.set_mutation_deviation(0.1)
                        elif event.key == pygame.K_DOWN:
                            self.set_speed(-1)
                        elif event.key == pygame.K_UP:
                            self.set_speed(1)
                        elif event.key == pygame.K_w:
                            self.set_mutation_rate(0.01)
                        elif event.key == pygame.K_s:
                            self.set_mutation_rate(-0.01)
                        elif event.key == pygame.K_p:
                            self.show = not self.show
                # game.move(game.current_snake.dir)
                # game.current_snake.calc_dir(game.input)
                screen.fill((0, 0, 0))
                for text, recta in self.text_boxes:
                    screen.blit(text, recta)
                screen.blit(self.plot_fitness, (350, 0))
                for row in range(self.width):
                    for column in range(self.height):
                        color = self.get_RGB(row, column)
                        pygame.draw.rect(screen, color, [(rec.s + rec.w) * column + rec.s + 10,
                                                         (rec.s + rec.h) * row + rec.s + 10, rec.w, rec.h])
                clock.tick(self.speed * 10)
        self.save_population(self.save_file)
        pygame.quit()
 
 
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
    logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
    pygame.init()
    size = width, height = 1000, 600
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    rec = Rectangle(10, 10, 1)
    grid = np.zeros((30, 30))
    game = GameHandler(grid, 0.05, 0.3, 15)
    game.run()
