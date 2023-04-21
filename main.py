run0 = False
run1 = False
run2 = False
run3 = False
snakeAgentOn = False
bbAgentOn = False
run4 = True

while True:

    import pygame
    import sys
    import math
    import random
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython import display

    # initialize pygame
    pygame.init()

    # defining global variables
    SW, SH = 800, 800
    mainFont = pygame.font.SysFont("calibri", 40)
    snakeButtonText = mainFont.render("Snake!", True, "white")
    bbButtonText = mainFont.render("Brick Breaker", True, "white")
    snakeAIButtonText = mainFont.render("Snake AI", True, "white")
    bbAIButtonText = mainFont.render("Brick Breaker AI", True, "white")

    gameStatePause = False

    # creating window
    screen = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("Main Menu")

    # functions
    def menuText(text, font, color, x, y):
        img = font.render(text, True, color)
        screen.blit(img, (x, y))

    #classes
    class mainMenuButton:
        def __init__(self, buttonText, x, y):
            self.buttonText = buttonText
            self.rect = self.buttonText.get_rect(center=(x, y))

        def draw(self):
            global run0
            global run1
            global run2
            global run3
            global run4
            # get mouse position
            pos = pygame.mouse.get_pos()

            #check mouseover and click
            if self.rect.collidepoint(pos):
                if pygame.mouse.get_pressed()[0]:
                    if self.buttonText == snakeButtonText:
                        run0 = True
                    elif self.buttonText == bbButtonText:
                        run1 = True
                    elif self.buttonText == snakeAIButtonText:
                        run2 = True
                    elif self.buttonText == bbAIButtonText:
                        run3 = True

            screen.blit(self.buttonText, self.rect)

     #class pauseMenuButton():
     #   def __init__(self, buttonText, buttonFunction, x, y):
     #       self.buttonText = buttonText
     #       self.buttonFile = buttonFile
     #       self.rect =

     #objects

    snakeButton = mainMenuButton(snakeButtonText, 75, 450)
    bbButton = mainMenuButton(bbButtonText, 125, 500)
    snakeAIButton = mainMenuButton(snakeAIButtonText, 88, 550)
    bbAIButton = mainMenuButton(bbAIButtonText, 146, 600)

    plt.ion()

    def plot(scores, meanScores):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.plot(meanScores)
        plt.ylim(ymin=0)
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(meanScores) - 1, meanScores[-1], str(meanScores[-1]))
        plt.show(block=False)
        plt.pause(.1)

    class snakeLinearQNet(nn.Module):
        def __init__(self, inputSize, hiddenSize, outputSize):
            super().__init__()
            self.linear1 = nn.Linear(inputSize, hiddenSize)
            self.linear2 = nn.Linear(hiddenSize, outputSize)

        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return x

        def save(self, fileName="model.pth"):
            modelFolderPath = "./model"
            if not os.path.exists(modelFolderPath):
                os.makedirs(modelFolderPath)

            fileName = os.path.join(modelFolderPath, fileName)
            torch.save(self.state_dict(), fileName)

    class QTrainerSnake:
        def __init__(self, model, lr, gamma):
            self.lr = lr
            self.gamma = gamma
            self.model = model
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
            self.criterion = nn.MSELoss()

        def trainStep(self, state, action, reward, stateNext, gameOver):
            state = torch.tensor(state, dtype=torch.float)
            stateNext = torch.tensor(stateNext, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)
            # (n, x)

            if len(state.shape) == 1:
                    # (1, x)
                state = torch.unsqueeze(state, 0)
                stateNext = torch.unsqueeze(stateNext, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                gameOver = (gameOver, )

            # 1: predicted Q values in current state
            pred = self.model(state)

            target = pred.clone()
            for idx in range(len(gameOver)):
                QNew = reward[idx]
                if not gameOver[idx]:
                    QNew = reward[idx] + self.gamma * torch.max(self.model(stateNext[idx]))

                target[idx][torch.argmax(action).item()] = QNew

            #2: QNew = r + gamma * max(next predicted q value), only if not gameOver
            #pred.clone()
            #preds[argmax(action)] = QNew
            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()

            self.optimizer.step()

    while run0 == True:
        import pygame
        import random
        from enum import Enum
        from collections import namedtuple

        pygame.init()
        font = pygame.font.SysFont('calibri', 25)


        # font = pygame.font.SysFont('arial', 25)

        class Direction(Enum):
            RIGHT = 1
            LEFT = 2
            UP = 3
            DOWN = 4


        Point = namedtuple('Point', 'x, y')

        # rgb colors
        white = (255, 255, 255)
        red = (200, 0, 0)
        blue1 = (0, 0, 255)
        blue2 = (0, 100, 255)
        black = (0, 0, 0)

        BLOCK_SIZE = 20
        SPEED = 20


        class SnakeGame:

            def __init__(self, w=640, h=480):
                self.w = w
                self.h = h
                # init display
                self.display = pygame.display.set_mode((self.w, self.h))
                pygame.display.set_caption('Snake')
                self.clock = pygame.time.Clock()

                # init game state
                self.direction = Direction.RIGHT

                self.head = Point(self.w / 2, self.h / 2)
                self.snake = [self.head,
                              Point(self.head.x - BLOCK_SIZE, self.head.y),
                              Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

                self.score = 0
                self.food = None
                self._placeFood()

            def _placeFood(self):
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                self.food = Point(x, y)
                if self.food in self.snake:
                    self._placeFood()

            def playStep(self):
                # 1. collect user input
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            self.direction = Direction.LEFT
                        elif event.key == pygame.K_RIGHT:
                            self.direction = Direction.RIGHT
                        elif event.key == pygame.K_UP:
                            self.direction = Direction.UP
                        elif event.key == pygame.K_DOWN:
                            self.direction = Direction.DOWN

                # 2. move
                self._move(self.direction)  # update the head
                self.snake.insert(0, self.head)

                # 3. check if game over
                gameOver = False
                if self._isCollision():
                    gameOver = True
                    return gameOver, self.score

                # 4. place new food or just move
                if self.head == self.food:
                    self.score += 1
                    self._placeFood()
                else:
                    self.snake.pop()

                # 5. update ui and clock
                self._updateUI()
                self.clock.tick(SPEED)
                # 6. return game over and score
                return gameOver, self.score

            def _isCollision(self):
                # hits boundary
                if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
                    return True
                # hits itself
                if self.head in self.snake[1:]:
                    return True

                return False

            def _updateUI(self):
                self.display.fill(black)

                for pt in self.snake:
                    pygame.draw.rect(self.display, blue1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.display, blue2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

                pygame.draw.rect(self.display, red, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

                self.text = font.render("Score: " + str(self.score), True, white)
                self.display.blit(self.text, [0, 0])
                pygame.display.flip()

            def _move(self, direction):
                x = self.head.x
                y = self.head.y
                if direction == Direction.RIGHT:
                    x += BLOCK_SIZE
                elif direction == Direction.LEFT:
                    x -= BLOCK_SIZE
                elif direction == Direction.DOWN:
                    y += BLOCK_SIZE
                elif direction == Direction.UP:
                    y -= BLOCK_SIZE

                self.head = Point(x, y)


        while run0 == True:
            game = SnakeGame()

            # game loop
            while True:
                gameOver, score = game.playStep()

                if gameOver == True:
                    break

            print('Final Score', score)

            pygame.quit()

    while run1 == True:
        pygame.init()

        # colors
        white = (255, 255, 255)
        black = (0, 0, 0)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (50, 100, 200)
        gray = (220, 220, 220)


        # classes
        class Paddle(pygame.sprite.Sprite):
            def __init__(self):
                super().__init__()
                black = (0, 0, 0)

                self.image = pygame.Surface([120, 20])
                pygame.draw.rect(self.image, black, [0, 0, 120, 20])
                self.rect = self.image.get_rect()

            def moveRight(self):
                self.rect.x += 6
                if self.rect.x >= (SW - 120):
                    self.rect.x = SW - 120

            def moveLeft(self):
                self.rect.x -= 6
                if self.rect.x <= 0:
                    self.rect.x = 0


        class Ball(pygame.sprite.Sprite):
            def __init__(self):
                super().__init__()
                red = (255, 0, 0)

                self.image = pygame.Surface([20, 20])
                pygame.draw.rect(self.image, red, [0, 0, 20, 20])
                self.rect = self.image.get_rect()

                self.velocity = [random.randint(3, 7), random.randint(4, 8)]

            def initPos(self):
                self.rect.x = (SW / 2) - 10
                self.rect.y = SH - 300

            def update(self):
                self.rect.x += self.velocity[0]
                self.rect.y += self.velocity[1]

            def bounceX(self):
                self.velocity[0] = -self.velocity[0]

            def bounceY(self):
                self.velocity[1] = -self.velocity[1]


        class Brick(pygame.sprite.Sprite):
            def __init__(self, color):
                super().__init__()

                self.image = pygame.Surface([100, 20])
                pygame.draw.rect(self.image, color, [0, 0, 100, 20])
                self.rect = self.image.get_rect()


        # functions
        def createBricks(c, r):
            for i in range(c):
                for j in range(r):
                    brick = Brick(blue)
                    brick.rect.x = 20 + i * 110
                    brick.rect.y = 20 + j * 30
                    allBricks.add(brick)
                    allSprites.add(brick)


        def killAllBricks():
            for brick in allBricks:
                brick.kill()


        # display
        screen = pygame.display.set_mode((SW, SH))
        pygame.display.set_caption("Brick Breaker")

        # clock
        FPS = 60
        clock = pygame.time.Clock()

        # create paddle
        paddle = Paddle()
        paddle.rect.x = (SW / 2) - 60
        paddle.rect.y = (SH) - 60

        # create ball
        ball = Ball()
        ball.initPos()

        # group of sprites
        allSprites = pygame.sprite.Group()
        allSprites.add(paddle)
        allSprites.add(ball)

        # bricks
        allBricks = pygame.sprite.Group()

        # create bricks
        createBricks(7, 9)

        score = 0
        lives = 3
        start = False
        while True:
            # event listeners
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # fill screen
            screen.fill(gray)

            # draw the sprites
            allSprites.draw(screen)

            # handle key events
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                start = True
            if keys[pygame.K_RIGHT]:
                paddle.moveRight()
            if keys[pygame.K_LEFT]:
                paddle.moveLeft()

            # Score
            font = pygame.font.Font(None, 40)
            text = font.render(f'Score: {score}', 1, black)
            screen.blit(text, (20, SH - 30))
            # Lives
            font = pygame.font.Font(None, 40)
            text = font.render(f'Lives: {lives}', 1, black)
            screen.blit(text, (SW - 120, SH - 30))

            # game over message
            if lives == 0:
                font = pygame.font.Font(None, 40)
                text = font.render(f'GAME OVER', 1, black)
                screen.blit(text, (SW / 2.5, SH / 2))
                font = pygame.font.Font(None, 40)
                text = font.render(f'PRESS SPACE TO REPLAY', 1, black)
                screen.blit(text, (SW / 3.3, SH / 1.5))
            if len(allBricks) == 0:
                font = pygame.font.Font(None, 40)
                text = font.render(f'YOU WIN!!', 1, black)
                screen.blit(text, (SW / 2.5, SH / 2))
                ball.initPos()
                score = 0
                start = False

            if start:
                # update the Sprites
                allSprites.update()
                if lives == 0:
                    lives = 3
                    score = 0
                    killAllBricks()
                    createBricks(7, 9)
                if ball.rect.right >= SW or ball.rect.left < 0:
                    ball.bounceX()
                if ball.rect.top <= 0:
                    ball.bounceY()
                if ball.rect.y >= SH - 40:
                    ball.initPos()
                    lives -= 1
                    start = False

                # collisions with paddle
                if ball.rect.colliderect(paddle.rect):
                    if abs(paddle.rect.top - ball.rect.bottom) < 9:
                        ball.bounceY()
                    if abs(paddle.rect.left - ball.rect.right) < 9 or abs(paddle.rect.right - ball.rect.left) < 9:
                        ball.bounceX()

                # collisions with bricks
                ballHitList = pygame.sprite.spritecollide(ball, allBricks, False)
                for brick in ballHitList:
                    ball.bounceY()
                    brick.kill()
                    score += 1

            # draw a line
            pygame.draw.line(screen, black, [0, SH - 35], [SW, SH - 35], 5)

            # update display
            pygame.display.update()

            # tick clock
            clock.tick(FPS)

    from collections import deque
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LR = 0.001

    class snakeAgent:

        def __init__(self):
            self.numGames = 0
            self.epsilon = 0  # randomness
            self.gamma = 0.9  # discount rate, must be smaller than 1
            self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
            self.model = snakeLinearQNet(11, 256, 3)
            self.trainer = QTrainerSnake(self.model, lr=LR, gamma=self.gamma)

        def getState(self, game):
            head = game.snake[0]
            pointL = Point(head.x - 20, head.y)
            pointR = Point(head.x + 20, head.y)
            pointU = Point(head.x, head.y - 20)
            pointD = Point(head.x, head.y + 20)

            dirL = game.direction == Direction.LEFT
            dirR = game.direction == Direction.RIGHT
            dirU = game.direction == Direction.UP
            dirD = game.direction == Direction.DOWN

            state = [
                # Danger straight
                (dirR and game.isCollision(pointR)) or
                (dirL and game.isCollision(pointL)) or
                (dirU and game.isCollision(pointU)) or
                (dirD and game.isCollision(pointD)),

                # Danger right
                (dirU and game.isCollision(pointR)) or
                (dirD and game.isCollision(pointL)) or
                (dirL and game.isCollision(pointU)) or
                (dirR and game.isCollision(pointD)),

                # Danger left
                (dirD and game.isCollision(pointR)) or
                (dirU and game.isCollision(pointL)) or
                (dirR and game.isCollision(pointU)) or
                (dirL and game.isCollision(pointD)),

                # Move direction
                dirL,
                dirR,
                dirU,
                dirD,

                # Food location
                game.food.x < game.head.x,  # food left
                game.food.x > game.head.x,  # food right
                game.food.y < game.head.y,  # food up
                game.food.y > game.head.y  # food down
                ]
            return np.array(state, dtype=int)

        def remember(self, state, action, reward, nextState, gameOver):
            self.memory.append((state, action, reward, nextState, gameOver)) # popleft if MAX_MEMORY is reached

        def trainLongMemory(self):
            if len(self.memory) > BATCH_SIZE:
                miniSample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
            else:
                miniSample = self.memory

            states, actions, rewards, nextStates, gameOvers = zip(*miniSample)
            self.trainer.trainStep(states, actions, rewards, nextStates, gameOvers)
        def trainShortMemory(self, state, action, reward, nextState, gameOver):
            self.trainer.trainStep(state, action, reward, nextState, gameOver)

        def getAction(self, state):
            # random moves: tradeoff exploration / exploitation
            self.epsilon = 80 - self.numGames
            finalMove = [0,0,0]
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 2)
                finalMove[move] = 1
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                finalMove[move] = 1

            return finalMove

    def snakeTrain():
        global numGames
        plotScores = []
        plotMeanScores = []
        totalScore = 0
        record = 0
        agent = snakeAgent()
        game = SnakeAI()
        while True:
            # get old state
            stateOld = agent.getState(game)

            # get move
            finalMove = agent.getAction(stateOld)

            # perform move and get new state
            reward, gameOver, score = game.playStep(finalMove)
            stateNew = agent.getState(game)

            # train short term memory
            agent.trainShortMemory(stateOld, finalMove, reward, stateNew, gameOver)

            # remember
            agent.remember(stateOld, finalMove, reward, stateNew, gameOver)

            if gameOver:
                # train long memory
                game.reset()
                agent.numGames += 1
                agent.trainLongMemory()

                if score > record:
                    record = score
                    agent.model.save()

                print("Game", agent.numGames, "Score", score, "Record:", record)

                plotScores.append(score)
                totalScore += score
                meanScore = totalScore / agent.numGames
                plotMeanScores.append(meanScore)
                plot(plotScores, plotMeanScores)

    def turnOnSnakeAgent():
        global snakeAgentOn
        snakeAgentOn = True

    def turnOnbbAgent():
        global bbAgentOn
        bbAgentOn = True

    while run2 == True:
        import pygame
        import random
        from enum import Enum
        from collections import namedtuple

        pygame.init()
        font = pygame.font.SysFont('calibri', 25)


        # font = pygame.font.SysFont('arial', 25)

        class Direction(Enum):
            RIGHT = 1
            LEFT = 2
            UP = 3
            DOWN = 4


        Point = namedtuple('Point', 'x, y')

        # rgb colors
        white = (255, 255, 255)
        red = (200, 0, 0)
        blue1 = (0, 0, 255)
        blue2 = (0, 100, 255)
        black = (0, 0, 0)

        BLOCK_SIZE = 20
        SPEED = 120 #snake game for player at 20


        class SnakeAI:

            def __init__(self, w=640, h=480):
                self.w = w
                self.h = h
                # init display
                self.display = pygame.display.set_mode((self.w, self.h))
                pygame.display.set_caption('Snake')
                self.clock = pygame.time.Clock()

                # init game state
                self.direction = Direction.RIGHT

                self.head = Point(self.w / 2, self.h / 2)
                self.snake = [self.head,
                              Point(self.head.x - BLOCK_SIZE, self.head.y),
                              Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

                self.score = 0
                self.food = None
                self._placeFood()
                self.frameIteration = 0

            def reset(self):
                # init game state
                self.direction = Direction.RIGHT

                self.head = Point(self.w / 2, self.h / 2)
                self.snake = [self.head,
                              Point(self.head.x - BLOCK_SIZE, self.head.y),
                              Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

                self.score = 0
                self.food = None
                self._placeFood()
                self.frameIteration = 0

            def _placeFood(self):
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                self.food = Point(x, y)
                if self.food in self.snake:
                    self._placeFood()

            def playStep(self, action):
                self.frameIteration += 1
                # 1. collect user input
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

                # 2. move
                self._move(action)  # update the head
                self.snake.insert(0, self.head)

                # 3. check if game over
                reward = 0
                gameOver = False
                if self.isCollision() or self.frameIteration > 100*len(self.snake):
                    gameOver = True
                    reward = -10
                    return reward, gameOver, self.score

                # 4. place new food or just move
                if self.head == self.food:
                    self.score += 1
                    reward = 10
                    self._placeFood()
                else:
                    self.snake.pop()

                # 5. update ui and clock
                self._updateUI()
                self.clock.tick(SPEED)
                # 6. return game over and score
                return reward, gameOver, self.score

            def isCollision(self, pt=None):
                if pt is None:
                    pt = self.head
                # hits boundary
                if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
                    return True
                # hits itself
                if pt in self.snake[1:]:
                    return True

                return False

            def _updateUI(self):
                self.display.fill(black)

                for pt in self.snake:
                    pygame.draw.rect(self.display, blue1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.display, blue2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

                pygame.draw.rect(self.display, red, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

                text = font.render("Score: " + str(self.score), True, white)
                self.display.blit(text, [0, 0])
                pygame.display.flip()

            def _move(self, action):
                # [straight, right, left]

                clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
                idx = clockwise.index(self.direction)

                if np.array_equal(action, [1, 0, 0]):
                    newDir = clockwise[idx]
                elif np.array_equal(action, [0, 1, 0]):
                    nextIdx = (idx + 1) % 4 #clockwise turn
                    newDir = clockwise[nextIdx]
                else:
                    nextIdx = (idx - 1) % 4 #counterclockwise turn
                    newDir = clockwise[nextIdx]

                self.direction = newDir


                x = self.head.x
                y = self.head.y
                if self.direction == Direction.RIGHT:
                    x += BLOCK_SIZE
                elif self.direction == Direction.LEFT:
                    x -= BLOCK_SIZE
                elif self.direction == Direction.DOWN:
                    y += BLOCK_SIZE
                elif self.direction == Direction.UP:
                    y -= BLOCK_SIZE

                self.head = Point(x, y)

        turnOnSnakeAgent()
        while snakeAgentOn == True:
            snakeTrain()

    while run3 == True:
        import neat
        import os
        import pickle
        import time

        pygame.init()

        # colors
        white = (255, 255, 255)
        black = (0, 0, 0)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (50, 100, 200)
        gray = (220, 220, 220)

        # group of sprites
        allSprites = pygame.sprite.Group()
        # bricks
        allBricks = pygame.sprite.Group()


        # classes
        class Paddle(pygame.sprite.Sprite):

            def __init__(self):
                super().__init__()
                black = (0, 0, 0)

                self.image = pygame.Surface([200, 20])
                pygame.draw.rect(self.image, black, [0, 0, 120, 20])
                self.rect = self.image.get_rect()

            def moveRight(self):
                self.rect.x += 7
                if self.rect.x >= (SW - 120):
                    self.rect.x = SW - 120

            def moveLeft(self):
                self.rect.x -= 7
                if self.rect.x <= 0:
                    self.rect.x = 0

        class Ball(pygame.sprite.Sprite):
            def __init__(self):
                super().__init__()
                red = (255, 0, 0)

                self.image = pygame.Surface([20, 20])
                pygame.draw.rect(self.image, red, [0, 0, 20, 20])
                self.rect = self.image.get_rect()

                self.velocity = [random.randint(-1, 1) * 5, 5]
                if self.velocity[0] == 0:
                    self.velocity[0] = 1

            def initPos(self):
                self.rect.x = (SW / 2) - 10
                self.rect.y = SH - 300

            def update(self):
                self.rect.x += self.velocity[0]
                self.rect.y += self.velocity[1]

            def bounceX(self):
                self.velocity[0] = -self.velocity[0]

            def bounceY(self):
                self.velocity[1] = -self.velocity[1]


        class Brick(pygame.sprite.Sprite):
            def __init__(self, color):
                super().__init__()

                self.image = pygame.Surface([100, 20])
                pygame.draw.rect(self.image, color, [0, 0, 100, 20])
                self.rect = self.image.get_rect()


        # functions
        def createBricks(c, r):
            global allSprites
            global allBricks
            for i in range(c):
                for j in range(r):
                    brick = Brick(blue)
                    brick.rect.x = 20 + i * 110
                    brick.rect.y = 20 + j * 30
                    allBricks.add(brick)
                    allSprites.add(brick)


        def killAllBricks():
            for brick in allBricks:
                brick.kill()


        class bbGameInfo:
            def __init__(self, score, fail, quit):
                self.score = score
                self.fail = fail
                self.quit = quit

        class bbGame:

            def __init__(self, SW, SH, screen):
                # display
                self.SW = SW
                self.SH = SH
                self.screen = screen
                pygame.display.set_caption("Brick Breaker")

                # create paddle
                self.paddle = Paddle()
                self.paddle.rect.x = (self.SW / 2) - 60
                self.paddle.rect.y = (self.SH) - 60

                # create ball
                self.ball = Ball()
                self.ball.initPos()

                # clock
                self.FPS = 600
                self.clock = pygame.time.Clock()

                # make sprites
                allSprites.add(self.paddle)
                allSprites.add(self.ball)

                # create bricks
                createBricks(7, 9)

                self.score = 0.0
                self.fail = 0
                self.quit = False

                #self.lives = 3

                self.cur = 0
                self.prev = 0

                self.start = True

            def _collisions(self):
                # collisions with paddle
                if self.ball.rect.colliderect(self.paddle.rect):
                    if abs(self.paddle.rect.top - self.ball.rect.bottom) < 9:
                        self.ball.bounceY()
                    if abs(self.paddle.rect.left - self.ball.rect.right) < 9 or abs(
                            self.paddle.rect.right - self.ball.rect.left) < 9:
                        self.ball.bounceX()

                # collisions with bricks
                self.ballHitList = pygame.sprite.spritecollide(self.ball, allBricks, False)
                for brick in self.ballHitList:
                    self.ball.bounceY()
                    brick.kill()
                    self.score += 1

            def resetPos(self):
                killAllBricks()
                createBricks(7, 9)
                self.ball.initPos()

            def loop(self):
                # event listeners
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    self.quit = True
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                # fill screen
                self.screen.fill(gray)

                # draw the sprites
                allSprites.draw(screen)

                # handle key events
                #keys = pygame.key.get_pressed()
                #if keys[pygame.K_RIGHT]:
                #    self.paddle.moveRight()
                #if keys[pygame.K_LEFT]:
                #    self.paddle.moveLeft()

                # Score
                font = pygame.font.Font(None, 40)
                text = font.render(f'Score: {self.score}', 1, black)
                screen.blit(text, (20, self.SH - 30))
                # Lives
                #font = pygame.font.Font(None, 40)
                #text = font.render(f'Lives: {self.lives}', 1, black)
                #screen.blit(text, (self.SW - 120, self.SH - 30))

                # game over message
                #if self.lives == 0:
                #    font = pygame.font.Font(None, 40)
                #    text = font.render(f'GAME OVER', 1, black)
                #    self.screen.blit(text, (self.SW / 2.5, self.SH / 2))
                #    font = pygame.font.Font(None, 40)
                #    text = font.render(f'PRESS SPACE TO REPLAY', 1, black)
                #    self.screen.blit(text, (self.SW / 3.3, self.SH / 1.5))
                #if len(allBricks) == 0:
                #    font = pygame.font.Font(None, 40)
                #    text = font.render(f'YOU WIN!!', 1, black)
                #    self.screen.blit(text, (self.SW / 2.5, self.SH / 2))
                #    self.ball.initPos()
                #    self.score = 0
                #    self.start = False

                if self.start:
                    # update the Sprites
                    allSprites.update()
                    allBricks.update()
                    #if self.lives == 0:
                    #    self.lives = 3
                    #    self.score = 0
                    #    killAllBricks()
                    #    createBricks(7, 9)
                    if self.ball.rect.right >= self.SW or self.ball.rect.left < 0:
                        self.ball.bounceX()
                    if self.ball.rect.top <= 0:
                        self.ball.bounceY()
                    if self.ball.rect.y >= self.SH - 40:
                        self.fail = 1
                        killAllBricks()
                        self.paddle.kill()
                        gameInfo = bbGameInfo(self.score, self.fail, self.quit)

                        return gameInfo
                        #self.resetPos()
                        #self.lives -= 1

                    self._collisions()

                # draw a line
                pygame.draw.line(self.screen, black, [0, self.SH - 35], [SW, self.SH - 35], 5)

                # update display
                pygame.display.update()
                allSprites.update()
                allBricks.update()
                allSprites.draw(screen)

                # tick clock
                self.clock.tick(self.FPS)

                gameInfo = bbGameInfo(self.score, self.fail, self.quit)

                return gameInfo

            def moveAI(self, net):
                player = [(self.genome, net, self.paddle)]
                for (genome, net, paddle) in player:
                    output = net.activate(
                        (self.paddle.rect.x, self.ball.rect.x, abs(self.paddle.rect.y - self.ball.rect.y)))
                    decision = output.index(max(output))

                    if decision == 0:  # Don't move
                        self.genome.fitness -= 0.01  # we want to discourage this
                    elif decision == 1:
                        self.paddle.moveRight()
                    else:  # Move down
                        self.paddle.moveLeft()

            def bbGameTrainAI(self, game, genome, config):
                startTime = time.time()

                net = neat.nn.FeedForwardNetwork.create(genome, config)
                self.genome = genome

                while True:

                    gameInfo = game.loop()

                    self.moveAI(net)

                    duration = time.time() - startTime

                    if gameInfo.fail >= 1 or gameInfo.score >= 62 or gameInfo.quit:
                        killAllBricks()
                        self.paddle.kill()
                        self.calcFitness(gameInfo.score, duration)
                        break

            def calcFitness(self, gameInfoScore, duration):
                self.genome.fitness += gameInfoScore + duration

        def evalGenomes(genomes, config):
            #screen = pygame.display.set_mode((SW, SH))
            #game = bbGame(800, 800, screen)

            for i, (genomeID, genome) in enumerate(genomes):
                screen = pygame.display.set_mode((SW, SH))
                killAllBricks()
                game = bbGame(800, 800, screen)
                genome.fitness = 0
                if i == len(genomes) - 1:
                    killAllBricks()
                    game.paddle.kill()
                    break

                game.bbGameTrainAI(game, genome, config)

                #for j, (genomeID1, genome1) in enumerate(genomes):
                #    genome1.fitness = 0
                #    if i == len(genomes) - 1:
                #        break

                #    game.bbGameTrainAI(game, genome1, config)

        def runNeat(config):
            #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-7')
            p = neat.Population(config)
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(1))

            winner = p.run(evalGenomes, 50)

            with open("best.pickle", "wb") as f:
                pickle.dump(winner, f)

        bbAgentOn = True
        while bbAgentOn:
            localDir = os.path.dirname(__file__)
            configPath = os.path.join(localDir, "config.txt")

            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 configPath)

            runNeat(config)

    run4 = True
    while run4:
        screen.fill("purple")

        snakeButton.draw()
        bbButton.draw()
        snakeAIButton.draw()
        bbAIButton.draw()
        mainMenuText = mainFont.render("Games Played By AI!!", True, "white")
        mainMenuTextRect = mainMenuText.get_rect(center=(377, 200))
        screen.blit(mainMenuText, mainMenuTextRect)

        # event handler
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    gameStatePause = True
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
            # todo add event listeners for click mainMenu buttons
        if run0 + run1 + run2 + run3 != 0:
            run4 = False

        if run4 == True:
            pygame.display.update()
