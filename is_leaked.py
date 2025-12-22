import pandas as pd

df = pd.read_csv("E:/New SW PR/backend/Ignored datasets/rockyou_dataset_cleaned.csv", usecols=['password'])

password_set = set(df['password'].astype(str))

def password_exists(password):

    return password in password_set