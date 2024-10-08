import tkinter as tk
from tkinter import messagebox
import random
import pickle
import os

class TicTacToeGUI:
    def __init__(self, master, agent):
        self.master = master
        self.master.title("XOX Oyunu")
        self.agent = agent
        self.board = [' '] * 9
        self.buttons = []
        self.create_board()
        self.player_symbol = 'O'
        self.agent_symbol = 'X'
        self.game_over = False

    def create_board(self):
        for i in range(9):
            button = tk.Button(self.master, text=' ', font=('Arial', 40), width=3, height=1,
                               command=lambda i=i: self.player_move(i))
            button.grid(row=i // 3, column=i % 3)
            self.buttons.append(button)

    def player_move(self, position):
        if self.board[position] == ' ' and not self.game_over:
            self.board[position] = self.player_symbol
            self.update_button(position)
            if self.check_winner(self.player_symbol):
                self.game_over = True
                messagebox.showinfo("Oyun Bitti", "Tebrikler, kazandınız!")
                self.agent.learn(self.prev_state, self.prev_action, -1, ''.join(self.board), True)
                self.reset_game()
            elif self.is_draw():
                self.game_over = True
                messagebox.showinfo("Oyun Bitti", "Oyun berabere!")
                self.agent.learn(self.prev_state, self.prev_action, 0, ''.join(self.board), True)
                self.reset_game()
            else:
                self.agent_move()

    def agent_move(self):
        state = ''.join(self.board)
        available_actions = [i for i, x in enumerate(self.board) if x == ' ']
        action = self.agent.choose_action(state, available_actions)
        self.prev_state = state
        self.prev_action = action
        self.board[action] = self.agent_symbol
        self.update_button(action)
        if self.check_winner(self.agent_symbol):
            self.game_over = True
            messagebox.showinfo("Oyun Bitti", "Maalesef, bilgisayar kazandı!")
            self.agent.learn(state, action, 1, ''.join(self.board), True)
            self.reset_game()
        elif self.is_draw():
            self.game_over = True
            messagebox.showinfo("Oyun Bitti", "Oyun berabere!")
            self.agent.learn(state, action, 0, ''.join(self.board), True)
            self.reset_game()
        else:
            next_state = ''.join(self.board)
            self.agent.learn(state, action, 0, next_state, False)

    def update_button(self, position):
        self.buttons[position]['text'] = self.board[position]

    def check_winner(self, player):
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Yatay
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Dikey
            (0, 4, 8), (2, 4, 6)              # Çapraz
        ]
        return any(all(self.board[i] == player for i in combo) for combo in win_conditions)

    def is_draw(self):
        return ' ' not in self.board

    def reset_game(self):
        self.board = [' '] * 9
        for button in self.buttons:
            button['text'] = ' '
        self.game_over = False

class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.175):
        self.q_table = {}  # Q-değerlerini tutan tablo
        self.alpha = alpha  # Öğrenme oranı
        self.gamma = gamma  # İndirim faktörü
        self.epsilon = epsilon  # Keşif oranı
        self.load_q_table()

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        q_values = [self.get_q_value(state, a) for a in available_actions]
        max_q = max(q_values)
        max_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(max_actions)

    def learn(self, state, action, reward, next_state, done):
        old_q = self.get_q_value(state, action)
        if done:
            target = reward
        else:
            next_available_actions = [i for i in range(9) if next_state[i] == ' ']
            if next_available_actions:
                next_q_values = [self.get_q_value(next_state, a) for a in next_available_actions]
                target = reward + self.gamma * max(next_q_values)
            else:
                target = reward
        self.q_table[(state, action)] = old_q + self.alpha * (target - old_q)
        self.save_q_table()

    def save_q_table(self):
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self):
        if os.path.exists('q_table.pkl'):
            with open('q_table.pkl', 'rb') as f:
                self.q_table = pickle.load(f)
        else:
            self.q_table = {}

def train(agent, episodes=10000):
    for _ in range(episodes):
        env = [' '] * 9
        state = ''.join(env)
        done = False
        while not done:
            available_actions = [i for i, x in enumerate(env) if x == ' ']
            action = agent.choose_action(state, available_actions)
            env[action] = 'X'
            if check_winner(env, 'X'):
                agent.learn(state, action, 1, ''.join(env), True)
                done = True
                continue
            elif ' ' not in env:
                agent.learn(state, action, 0, ''.join(env), True)
                done = True
                continue

            opponent_action = random.choice([i for i, x in enumerate(env) if x == ' '])
            env[opponent_action] = 'O'

            if check_winner(env, 'O'):
                agent.learn(state, action, -1, ''.join(env), True)
                done = True
                continue
            elif ' ' not in env:
                agent.learn(state, action, 0, ''.join(env), True)
                done = True
                continue

            next_state = ''.join(env)
            agent.learn(state, action, 0, next_state, False)
            state = next_state

def check_winner(board, player):
    win_conditions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Yatay
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Dikey
        (0, 4, 8), (2, 4, 6)              # Çapraz
    ]
    return any(all(board[i] == player for i in combo) for combo in win_conditions)

if __name__ == "__main__":
    agent = QLearningAgent()
    train(agent)
    # Grafik arayüzünü 
    root = tk.Tk()
    game = TicTacToeGUI(root, agent)
    root.mainloop()
