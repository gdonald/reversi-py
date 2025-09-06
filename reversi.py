import os
import sys
import random
import time
from enum import Enum

class Player(Enum):
    BLACK = 1
    WHITE = 2
    EMPTY = 0

class GameMode(Enum):
    ZERO_PLAYER = 0
    ONE_PLAYER = 1

class ReversiGame:
    def __init__(self):
        self.board = [[Player.EMPTY for _ in range(8)] for _ in range(8)]
        self.current_player = Player.BLACK
        self.cursor_x = 4
        self.cursor_y = 4
        self.game_mode = GameMode.ONE_PLAYER
        self.game_over = False
        self.initialize_board()

    def initialize_board(self):
        self.board[3][3] = Player.WHITE
        self.board[3][4] = Player.BLACK
        self.board[4][3] = Player.BLACK
        self.board[4][4] = Player.WHITE

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def display_board(self):
        self.clear_screen()
        print("Reversi - Terminal Edition")
        print("Use WASD to move, Enter to place piece, Q to quit\n")
        print(f"Current Player: {'BLACK' if self.current_player == Player.BLACK else 'WHITE'}")
        print(f"Game Mode: {'Computer vs Computer' if self.game_mode == GameMode.ZERO_PLAYER else 'Human vs Computer'}")
        print()
        
        print("   A B C D E F G H")
        for row in range(8):
            print(f"{row + 1}  ", end="")
            for col in range(8):
                if row == self.cursor_y and col == self.cursor_x:
                    if self.board[row][col] == Player.EMPTY:
                        print("[·]", end="")
                    elif self.board[row][col] == Player.BLACK:
                        print("[●]", end="")
                    else:
                        print("[○]", end="")
                else:
                    if self.board[row][col] == Player.EMPTY:
                        print(" · ", end="")
                    elif self.board[row][col] == Player.BLACK:
                        print(" ● ", end="")
                    else:
                        print(" ○ ", end="")
            print()

    def is_valid_move(self, row, col, player):
        if self.board[row][col] != Player.EMPTY:
            return False
        
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        opponent = Player.WHITE if player == Player.BLACK else Player.BLACK
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            found_opponent = False
            
            while 0 <= r < 8 and 0 <= c < 8:
                if self.board[r][c] == opponent:
                    found_opponent = True
                elif self.board[r][c] == player and found_opponent:
                    return True
                else:
                    break
                r += dr
                c += dc
        
        return False

    def get_valid_moves(self, player):
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self.is_valid_move(row, col, player):
                    valid_moves.append((row, col))
        return valid_moves

    def make_move(self, row, col, player):
        if not self.is_valid_move(row, col, player):
            return False
        
        self.board[row][col] = player
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        opponent = Player.WHITE if player == Player.BLACK else Player.BLACK
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            pieces_to_flip = []
            
            while 0 <= r < 8 and 0 <= c < 8:
                if self.board[r][c] == opponent:
                    pieces_to_flip.append((r, c))
                elif self.board[r][c] == player:
                    for flip_r, flip_c in pieces_to_flip:
                        self.board[flip_r][flip_c] = player
                    break
                else:
                    break
                r += dr
                c += dc
        
        return True

    def get_computer_move(self):
        valid_moves = self.get_valid_moves(self.current_player)
        if valid_moves:
            return random.choice(valid_moves)
        return None

    def count_pieces(self):
        black_count = sum(row.count(Player.BLACK) for row in self.board)
        white_count = sum(row.count(Player.WHITE) for row in self.board)
        return black_count, white_count

    def is_game_over(self):
        black_moves = self.get_valid_moves(Player.BLACK)
        white_moves = self.get_valid_moves(Player.WHITE)
        return len(black_moves) == 0 and len(white_moves) == 0

    def get_winner(self):
        black_count, white_count = self.count_pieces()
        if black_count > white_count:
            return Player.BLACK
        elif white_count > black_count:
            return Player.WHITE
        else:
            return None

    def play_game(self):
        while not self.game_over:
            self.display_board()
            
            valid_moves = self.get_valid_moves(self.current_player)
            
            if not valid_moves:
                opponent = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
                opponent_moves = self.get_valid_moves(opponent)
                
                if not opponent_moves:
                    self.game_over = True
                    break
                else:
                    print(f"\nNo valid moves for {'BLACK' if self.current_player == Player.BLACK else 'WHITE'}. Skipping turn.")
                    if self.game_mode == GameMode.ONE_PLAYER or self.current_player == Player.WHITE:
                        time.sleep(2)
                    else:
                        input("Press Enter to continue...")
                    self.current_player = opponent
                    continue

            if self.game_mode == GameMode.ZERO_PLAYER or (self.game_mode == GameMode.ONE_PLAYER and self.current_player == Player.WHITE):
                move = self.get_computer_move()
                if move:
                    row, col = move
                    self.make_move(row, col, self.current_player)
                    self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
                time.sleep(1)
            else:
                if self.handle_player_input():
                    self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK

        self.show_game_over()

    def handle_player_input(self):
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt
                    key = msvcrt.getch()
                    if isinstance(key, bytes):
                        key = key.decode('utf-8')
                else:
                    import termios
                    import tty
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setraw(fd)
                        key = sys.stdin.read(1)
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                if key.lower() == 'w':
                    self.cursor_y = max(0, self.cursor_y - 1)
                    self.display_board()
                elif key.lower() == 's':
                    self.cursor_y = min(7, self.cursor_y + 1)
                    self.display_board()
                elif key.lower() == 'a':
                    self.cursor_x = max(0, self.cursor_x - 1)
                    self.display_board()
                elif key.lower() == 'd':
                    self.cursor_x = min(7, self.cursor_x + 1)
                    self.display_board()
                elif key == '\r' or key == '\n':
                    if self.is_valid_move(self.cursor_y, self.cursor_x, self.current_player):
                        self.make_move(self.cursor_y, self.cursor_x, self.current_player)
                        return True
                    else:
                        print("\nInvalid move! Press any key to continue...")
                        if os.name == 'nt':
                            msvcrt.getch()
                        else:
                            sys.stdin.read(1)
                        self.display_board()
                elif key.lower() == 'q':
                    sys.exit(0)
            except KeyboardInterrupt:
                sys.exit(0)

    def show_game_over(self):
        self.display_board()
        black_count, white_count = self.count_pieces()
        winner = self.get_winner()
        
        print(f"\nGame Over!")
        print(f"Black: {black_count} pieces")
        print(f"White: {white_count} pieces")
        
        if winner is None:
            print("It's a tie!")
        else:
            print(f"Winner: {'BLACK' if winner == Player.BLACK else 'WHITE'}!")

    def start_new_game(self):
        while True:
            print("\nSelect game mode:")
            print("1. Play against computer")
            print("2. Watch computer vs computer")
            print("q. Quit")
            
            choice = input("Enter your choice: ").lower()
            
            if choice == '1':
                self.game_mode = GameMode.ONE_PLAYER
                break
            elif choice == '2':
                self.game_mode = GameMode.ZERO_PLAYER
                break
            elif choice == 'q':
                sys.exit(0)
            else:
                print("Invalid choice. Please try again.")
        
        self.board = [[Player.EMPTY for _ in range(8)] for _ in range(8)]
        self.current_player = Player.BLACK
        self.cursor_x = 4
        self.cursor_y = 4
        self.game_over = False
        self.initialize_board()
        self.play_game()

def main():
    game = ReversiGame()
    
    while True:
        game.start_new_game()
        
        while True:
            play_again = input("\nPlay again? (y/n): ").lower()
            if play_again == 'y':
                break
            elif play_again == 'n':
                print("Thanks for playing!")
                sys.exit(0)
            else:
                print("Please enter 'y' for yes or 'n' for no.")

if __name__ == "__main__":
    main()