import numpy as np

BOARD_ROWS = 5
BOARD_COLS = 10
WIN_STATE = (1, 1)
WIN_STATE2 = (4, 8)
START = (2, 3)
DETERMINISTIC = True


class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[1, 1] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC
        

    def giveReward(self):
        return self.reward
        # if self.state == WIN_STATE and not self.win1:
        #     self.win1 = True
            
        #     return 20
        # elif self.state == WIN_STATE2 and not self.win2:
        #     self.win2 = True
        #     return 20
        # elif self.state == LOSE_STATE:
        #     return -1
        # else:
        #     return -1

    def isEndFunc(self,win1,win2,reward):
    
        if self.state == WIN_STATE and not win1:
            #reward = reward + 20
            win1 = True
            if (win1 and win2):
                self.isEnd = True
                reward = reward + 40
                if reward < 0:
                    reward = 0
            return win1, win2, reward
        
        elif self.state == WIN_STATE2 and not win2:
            #reward = reward + 20
            win2 = True
            if (win1 and win2):
                self.isEnd = True
                reward = reward + 40
                if reward < 0:
                    reward = 0
            return win1, win2, reward
        else:
            reward = reward 
            if (win1 and win2):
                self.isEnd = True
                reward = reward + 40
                if reward < 0:
                    reward = 0
            return win1, win2, reward

    def _chooseActionProb(self, action):
        if action == "up":
            return np.random.choice(["up","down","left", "right"], p=[0.25, 0.25, 0.25, .25])
        if action == "down":
            return np.random.choice(["up","down","left", "right"], p=[0.25, 0.25, 0.25,.25])
        if action == "left":
            return np.random.choice(["up", "down","left","right"], p=[0.25, 0.25, 0.25,.25])
        if action == "right":
            return np.random.choice(["up", "down","left", "right"], p=[0.25, 0.25, 0.25,.25])

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position on board
        """
        if self.determine:
            if action == "up":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            # non-deterministic
            action = self._chooseActionProb(action)
            self.determine = True
            nxtState = self.nxtPosition(action)

        # if next state is legal
        if (nxtState[0] >= 0) and (nxtState[0] <= 4):
            if (nxtState[1] >= 0) and (nxtState[1] <= 9):
                return nxtState
        return self.state

    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------------')


class Agent:

    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.isEnd = self.State.isEnd
        self.lr = .2 #learning rate, takes longer when low .2
        self.exp_rate = 0.5 #exploration rate .7 .75 .5
        self.decay_gamma = 0.85 #Reward decay does not work well when 1 .85 .8 .85
        self.restart = False
        self.count = 0
        self.win1 = False
        self.win2 = False
        self.reward = 0


        # initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        # update State
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd
        self.restart = True
        self.win1 = False
        self.win2 = False
        self.reward = 0

    def play(self, i):
        #i = 0
        #while i < rounds:
            # to the end of game back propagate reward
        if self.State.isEnd:
            # back propagate
            reward = self.reward
            for a in self.actions:
                self.Q_values[self.State.state][a] = reward
            print("Game End Reward", reward)
            for s in reversed(self.states):
                current_q_value = self.Q_values[s[0]][s[1]]
                reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                self.Q_values[s[0]][s[1]] = round(reward, 3)
            self.reset()
            i += 1
        else:
            action = self.chooseAction()
            # append trace
            self.states.append([(self.State.state), action])
            #print("current position {} action {}".format(self.State.state, action))
            # by taking the action, it reaches the next state
            self.State = self.takeAction(action)
            # mark is end
            self.win1,self.win2,self.reward = self.State.isEndFunc(self.win1,self.win2,self.reward)
            #print("nxt state", self.State.state)
            #print("---------------------")
            self.isEnd = self.State.isEnd
            self.restart = False
        return(i, self)


if __name__ == "__main__":
    ag = Agent()
    print("initial Q-values ... \n")
    print(ag.Q_values)
    ag2 = Agent()
    print("initial Q-values ... \n")
    print(ag2.Q_values)
    i = 0
    j = 0
    while i<30:
        i, ag  = ag.play(i)

        if ag.win1:
            ag2.win1 = True
        if ag.win2:
            ag2.win2 = True
        j, ag2 = ag2.play(j)
        if ag2.win1:
            ag.win1 = True
        if ag2.win2:
            ag.win2 = True
        
    print("latest Q-values ... \n")
    print(ag.Q_values)
    print("latest Q-values ... \n")
    print(ag2.Q_values)
    
    
    Board = np.zeros([BOARD_ROWS,BOARD_COLS])
    
    print("\n\nAGENT 1:")

    for i in range (0, BOARD_ROWS):
        for j in range (0, BOARD_COLS):
            Board[i,j] = np.array([ag.Q_values[(i, j)]['up'],ag.Q_values[(i, j)]['down'],ag.Q_values[(i, j)]['left'],ag.Q_values[(i, j)]['right']]).argmax()
    
    Board[WIN_STATE] = 10
    Board[WIN_STATE2] = 10
    Board[START] = 20+Board[START]

    for i in range(0, BOARD_ROWS):
        print('--------------------------------')
        out = '| '
        for j in range(0, BOARD_COLS):
            if Board[i,j] == 10:
                token = '*'
            elif Board[i,j] == 20:
                token = '+↑'
            elif Board[i,j] == 21:
                token = '+↓'
            elif Board[i,j] == 22:
                token = '←+'
            elif Board[i,j] == 23:
                token = '+→'
            elif Board[i,j] == 0:
                token = '↑'
            elif Board[i,j] == 1:
                token = '↓'
            elif Board[i,j] == 2:
                token = '←'
            elif Board[i,j] == 3:
                token = '→'
            else:
                token = ' '
            out += token + ' | '
        print(out)
    print('--------------------------------')

    print("\n\nAGENT 2:")
    
    for i in range (0, BOARD_ROWS):
        for j in range (0, BOARD_COLS):
            Board[i,j] = np.array([ag2.Q_values[(i, j)]['up'],ag2.Q_values[(i, j)]['down'],ag2.Q_values[(i, j)]['left'],ag2.Q_values[(i, j)]['right']]).argmax()
    
    Board[WIN_STATE] = 10
    Board[WIN_STATE2] = 10
    Board[START] = 20+Board[START]

    for i in range(0, BOARD_ROWS):
        print('--------------------------------')
        out = '| '
        for j in range(0, BOARD_COLS):
            if Board[i,j] == 10:
                token = '*'
            elif Board[i,j] == 20:
                token = '+↑'
            elif Board[i,j] == 21:
                token = '+↓'
            elif Board[i,j] == 22:
                token = '←+'
            elif Board[i,j] == 23:
                token = '+→'
            elif Board[i,j] == 0:
                token = '↑'
            elif Board[i,j] == 1:
                token = '↓'
            elif Board[i,j] == 2:
                token = '←'
            elif Board[i,j] == 3:
                token = '→'
            else:
                token = ' '
            out += token + ' | '
        print(out)
    print('--------------------------------')
     
    #self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict
    #np.array([ag.Q_values[(1, 8)]['up'],ag.Q_values[(1, 8)]['down'],ag.Q_values[(1, 8)]['left'],ag.Q_values[(1, 8)]['right']]).argmax()
    # BOARD_ROWS = 5
    # BOARD_COLS = 10
    # WIN_STATE = (1, 1)
    # LOSE_STATE = (1, 3)
    # START = (2,3)
    # DETERMINISTIC = False



    # self.board[self.state] = 1
    #     for i in range(0, BOARD_ROWS):
    #         print('-----------------')
    #         out = '| '
    #         for j in range(0, BOARD_COLS):
    #             if self.board[i, j] == 1:
    #                 token = '*'
    #             if self.board[i, j] == -1:
    #                 token = 'z'
    #             if self.board[i, j] == 0:
    #                 token = '0'
    #             out += token + ' | '
    #         print(out)
    #     print('-----------------')
