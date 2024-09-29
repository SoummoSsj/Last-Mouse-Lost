import tkinter as tk
from tkinter import messagebox
from board import Board
from player import HumanPlayer, RandomPlayer, SmartPlayer, FuzzyPlayer, MinMaxPlayer, GeneticAlgorithmPlayer, AStarPlayer

class GameUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Last Mouse Lost Game")

        self.board = Board()
        self.players = []
        self.current_player_index = 0

        self.master.configure(bg='#2C3E50')  # Set background color for the main window

        self.show_welcome_screen()

    def show_welcome_screen(self):
        self.welcome_frame = tk.Frame(self.master, bg='#2C3E50')
        self.welcome_frame.pack(pady=20)

        tk.Label(self.welcome_frame, text="Welcome to Last Mouse Lost!", font=('Helvetica', 18, 'bold'), fg='#ECF0F1', bg='#2C3E50').pack(pady=10)

        tk.Label(self.welcome_frame, text="Choose your opponent:", font=('Helvetica', 14), fg='#ECF0F1', bg='#2C3E50').pack(pady=10)

        self.player_type_var = tk.IntVar()
        player_types = [("Human", 1), ("Random", 2), ("Smart", 3), ("Fuzzy", 4), ("MinMax", 5), ("Genetic Algorithm", 6)]
        for text, value in player_types:
            tk.Radiobutton(self.welcome_frame, text=text, variable=self.player_type_var, value=value, font=('Helvetica', 12), fg='#ECF0F1', bg='#34495E', selectcolor='#2980B9', activebackground='#2980B9').pack(anchor=tk.W)

        self.start_button = tk.Button(self.welcome_frame, text="Start Game", command=self.start_game, font=('Helvetica', 12, 'bold'), bg='#2980B9', fg='#ECF0F1')
        self.start_button.pack(pady=10)

    def start_game(self):
        opponent_choice = self.player_type_var.get()

        if opponent_choice == 1:
            self.players = [HumanPlayer(self.board), HumanPlayer(self.board)]
        elif opponent_choice == 2:
            self.players = [HumanPlayer(self.board), RandomPlayer(self.board)]
        elif opponent_choice == 3:
            self.players = [HumanPlayer(self.board), SmartPlayer(self.board, 0)]
        elif opponent_choice == 4:
            self.players = [HumanPlayer(self.board), FuzzyPlayer(self.board)]
        elif opponent_choice == 5:
            self.players = [HumanPlayer(self.board), MinMaxPlayer(self.board)]
        elif opponent_choice == 6:
            self.players = [HumanPlayer(self.board), GeneticAlgorithmPlayer(self.board)]
        else:
            messagebox.showerror("Error", "Please select a valid option.")
            return

        self.welcome_frame.destroy()
        self.create_board()
        self.update_board()
        self.status_label.config(text="Game started! Player 1's turn.")

        if isinstance(self.players[self.current_player_index], HumanPlayer):
            self.status_label.config(text="Player 1 (Human) starts.")
        else:
            self.master.after(1000, self.ai_move)

    def create_board(self):
        self.board_frame = tk.Frame(self.master, bg='#2C3E50')
        self.board_frame.pack(pady=20)

        self.board_buttons = []
        for r in range(len(self.board.b)):
            row_frame = tk.Frame(self.board_frame, bg='#2C3E50')
            row_frame.pack(side=tk.TOP)

            # Display row number with uniform width
            tk.Label(row_frame, text=f"Row {r}: ", font=('Helvetica', 12), fg='#ECF0F1', bg='#2C3E50', width=8, anchor='w').pack(side=tk.LEFT, padx=5)

            row_buttons = []
            for c in range(len(self.board.b[r])):
                btn = tk.Button(row_frame, text='o', width=4, height=2, font=('Helvetica', 12, 'bold'), bg='#3498DB', fg='#ECF0F1', activebackground='#2980B9', activeforeground='#ECF0F1')
                btn.pack(side=tk.LEFT, padx=2, pady=2)
                row_buttons.append(btn)
            self.board_buttons.append(row_buttons)

        self.input_frame = tk.Frame(self.master, bg='#2C3E50')
        self.input_frame.pack(pady=10)

        tk.Label(self.input_frame, text="Row:", font=('Helvetica', 12), fg='#ECF0F1', bg='#2C3E50').pack(side=tk.LEFT)
        self.row_var = tk.IntVar(value=0)
        self.row_menu = tk.OptionMenu(self.input_frame, self.row_var, *range(6))
        self.row_menu.config(width=5, font=('Helvetica', 12), bg='#2980B9', fg='#ECF0F1', activebackground='#2980B9', activeforeground='#ECF0F1')
        self.row_menu.pack(side=tk.LEFT, padx=5)

        tk.Label(self.input_frame, text="Amount:", font=('Helvetica', 12), fg='#ECF0F1', bg='#2C3E50').pack(side=tk.LEFT)
        self.amount_slider = tk.Scale(self.input_frame, from_=1, to=6, orient=tk.HORIZONTAL, font=('Helvetica', 12), bg='#34495E', fg='#ECF0F1', troughcolor='#2980B9', activebackground='#2980B9')
        self.amount_slider.pack(side=tk.LEFT, padx=5)

        self.submit_button = tk.Button(self.input_frame, text="Submit Move", command=self.make_move, font=('Helvetica', 12, 'bold'), bg='#27AE60', fg='#ECF0F1')
        self.submit_button.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(self.master, text="", font=('Helvetica', 14, 'bold'), fg='#ECF0F1', bg='#2C3E50')
        self.status_label.pack(pady=10)

    def make_move(self):
        player = self.players[self.current_player_index]
        if isinstance(player, HumanPlayer):
            try:
                r = self.row_var.get()
                a = self.amount_slider.get()
                if r < 0 or r >= len(self.board.b) or a <= 0 or self.board.row_empty(r):
                    raise ValueError
                self.board.update_b(r, a)
                self.status_label.config(text=f"Player {self.current_player_index + 1} (Human) made move: Row {r}, Amount {a}")
                self.next_turn()
            except ValueError:
                messagebox.showwarning("Invalid Move", "Please enter a valid row and amount.")
                self.row_var.set(0)
                self.amount_slider.set(1)

    def ai_move(self):
        player = self.players[self.current_player_index]
        move = player.move()
        self.board.update_b(move[0], move[1])
        self.update_board()  # Update the board before displaying the move
        self.status_label.config(text=f"Player {self.current_player_index + 1} (AI) made move: Row {move[0]}, Amount {move[1]}")
        self.master.after(500, self.next_turn)  # Pause before moving to the next turn

    def next_turn(self):
        if self.board.g_o():
            self.status_label.config(text=f"Player {self.current_player_index + 1} Lost!")
            self.end_game_prompt()
            return

        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        next_player = self.players[self.current_player_index]

        if isinstance(next_player, HumanPlayer):
            self.status_label.config(text=f"Player {self.current_player_index + 1} (Human)'s turn.")
        else:
            self.status_label.config(text=f"Player {self.current_player_index + 1} (AI) is thinking...")
            self.master.after(1000, self.ai_move)

    def end_game_prompt(self):
        response = messagebox.askquestion("Game Over", "Would you like to play again?")
        if response == 'yes':
            self.restart_game()
        else:
            self.master.quit()

    def restart_game(self):
        self.board_frame.destroy()
        self.input_frame.destroy()
        self.status_label.destroy()
        self.__init__(self.master)

    def update_board(self):
        for r in range(len(self.board.b)):
            for c in range(len(self.board.b[r])):
                if self.board.b[r][c] == 'x':
                    self.board_buttons[r][c].config(text='x', state=tk.DISABLED, bg='#E74C3C')
                else:
                    self.board_buttons[r][c].config(text='o', state=tk.NORMAL, bg='#3498DB')

if __name__ == '__main__':
    root = tk.Tk()
    app = GameUI(root)
    root.mainloop()
