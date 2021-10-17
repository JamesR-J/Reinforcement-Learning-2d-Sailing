import pandas as pd

df = pd.read_csv('./player_moves.csv')
moves = df['Move'].tolist()
#print(moves)