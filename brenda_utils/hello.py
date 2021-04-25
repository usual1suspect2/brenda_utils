import pandas as pd


def hello_word():
    print("hello world mthfck")

    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    print(df)


if __name__ == "__main__":
    hello_word()